"""Simulate normal operation and physical equipment faults in two water tanks."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from plotting import save_plots


TANK_CAPACITY_L = 1_000.0
TIMESTEP_MIN = 1.0
INITIAL_TANK1_L = 600.0
INITIAL_TANK2_L = 600.0
START_TIME = pd.Timestamp("2026-01-01 00:00:00")
DEFAULT_STEPS = 10_080
DEFAULT_SEED = 3015

INLET_RATE_LPM = 28.0
TRANSFER_RATE_LPM = 20.0
OUTLET_RATE_LPM = 25.0
FLOW_TARGET_INTERVAL_MIN = 120
FLOW_SMOOTHING_ALPHA = 1.0 / 30.0
DEMAND_PERIOD_MIN = 24.0 * 60.0

ANOMALY_TYPES = (
    "transfer_pump_failure",
    "inlet_valve_stuck_open",
    "outlet_pump_failure",
)
EVENTS_PER_TYPE = 5
STABILISATION_MIN = 360
FINAL_RECOVERY_RESERVE_MIN = 240
RECOVERY_CONFIRM_MIN = 15
CONTINUOUS_COLUMNS = (
    "tank1_level", "tank2_level", "inlet_flow", "transfer_flow",
    "outlet_flow", "pressure",
)


@dataclass(frozen=True)
class Event:
    """A planned fault; times are elapsed minutes from the simulation start."""

    event_id: int
    anomaly_type: str
    planned_start: int
    duration: int


def generate_event_plan(steps: int, seed: int) -> list[Event]:
    """Distribute five faults of each kind across separate time windows."""
    # A separate random stream preserves Stage 1's process/noise sequence.
    rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
    event_count = EVENTS_PER_TYPE * len(ANOMALY_TYPES)
    available = steps - STABILISATION_MIN - FINAL_RECOVERY_RESERVE_MIN
    if available < event_count * 360:
        raise ValueError("Stage 2 needs at least 6,000 steps to space 15 events.")

    edges = np.linspace(
        STABILISATION_MIN, steps - FINAL_RECOVERY_RESERVE_MIN,
        event_count + 1, dtype=int,
    )
    # Each group of three windows contains one event of each kind.
    kinds = [
        str(kind)
        for _ in range(EVENTS_PER_TYPE)
        for kind in rng.permutation(ANOMALY_TYPES)
    ]
    return [
        Event(
            event_id=index + 1,
            anomaly_type=kind,
            planned_start=int(edges[index] + rng.integers(30, 91)),
            duration=int(rng.integers(60, 161)),
        )
        for index, kind in enumerate(kinds)
    ]


def can_start_event(kind: str, valve: bool, pump1: bool, pump2: bool) -> bool:
    """Wait for a phase where the fault has a meaningful physical effect."""
    if kind == "inlet_valve_stuck_open":
        return not valve and pump1
    # Both pumps and the inlet are operating: a failed transfer pump removes
    # inflow to Tank 2; a failed outlet pump removes its discharge instead.
    return valve and pump1 and pump2


def simulate(
    steps: int = DEFAULT_STEPS,
    seed: int = DEFAULT_SEED,
    events: list[Event] | None = None,
) -> pd.DataFrame:
    """Return end-of-minute readings; events=None reproduces Stage 1."""
    if steps < 1:
        raise ValueError("steps must be at least 1")
    plans = [] if events is None else list(events)
    if any(e.anomaly_type not in ANOMALY_TYPES or e.duration < 1 for e in plans):
        raise ValueError("Every event needs a supported type and positive duration.")
    if any(e.planned_start < 0 for e in plans):
        raise ValueError("Event start times cannot be negative.")
    if plans != sorted(plans, key=lambda e: e.planned_start):
        raise ValueError("Events must be ordered by their planned start time.")
    if len({e.event_id for e in plans}) != len(plans) or any(e.event_id <= 0 for e in plans):
        raise ValueError("Event IDs must be unique positive integers.")

    rng = np.random.default_rng(seed)
    tank1_l, tank2_l = INITIAL_TANK1_L, INITIAL_TANK2_L
    inlet_valve_command = pump1_command = pump2_command = False
    flow_targets = np.zeros(3)
    flow_offsets = np.zeros(3)
    next_event = 0
    active_event: Event | None = None
    recovery_event: Event | None = None
    active_until = 0
    recovered_minutes = 0
    rows: list[dict[str, object]] = []

    for index in range(steps):
        # These hysteresis rules and their memory are unchanged from Stage 1.
        # Commands remain independent of whether equipment actually operates.
        if inlet_valve_command and tank1_l >= 750.0:
            inlet_valve_command = False
        elif not inlet_valve_command and tank1_l <= 450.0:
            inlet_valve_command = True
        if pump1_command and (tank1_l <= 400.0 or tank2_l >= 780.0):
            pump1_command = False
        elif not pump1_command and tank1_l >= 500.0 and tank2_l <= 600.0:
            pump1_command = True
        if pump2_command and tank2_l <= 450.0:
            pump2_command = False
        elif not pump2_command and tank2_l >= 780.0:
            pump2_command = True

        # Remove the physical fault at its scheduled duration, then let normal
        # flows and the unchanged controller restore operation without resets.
        if active_event is not None and index >= active_until:
            recovery_event = active_event
            active_event = None
            recovered_minutes = 0
        if active_event is None and recovery_event is None and next_event < len(plans):
            candidate = plans[next_event]
            if index >= candidate.planned_start and can_start_event(
                candidate.anomaly_type, inlet_valve_command, pump1_command, pump2_command
            ):
                active_event = candidate
                active_until = index + candidate.duration
                next_event += 1

        kind = active_event.anomaly_type if active_event is not None else "normal"
        inlet_valve_actual = inlet_valve_command or kind == "inlet_valve_stuck_open"
        pump1_actual = pump1_command and kind != "transfer_pump_failure"
        pump2_actual = pump2_command and kind != "outlet_pump_failure"

        # Keep Stage 1's normal demand, bounded drift and random draw order.
        elapsed_min = index * TIMESTEP_MIN
        if elapsed_min % FLOW_TARGET_INTERVAL_MIN == 0:
            flow_targets = rng.uniform(-1.0, 1.0, size=3)
        flow_offsets += FLOW_SMOOTHING_ALPHA * (flow_targets - flow_offsets)
        inlet_condition = INLET_RATE_LPM + 0.5 * flow_offsets[0] + rng.uniform(-0.3, 0.3)
        transfer_condition = (
            TRANSFER_RATE_LPM + 0.3 * flow_offsets[1] + rng.uniform(-0.2, 0.2)
        )
        outlet_demand = (
            OUTLET_RATE_LPM
            + 0.75 * np.sin(2.0 * np.pi * elapsed_min / DEMAND_PERIOD_MIN)
            + 0.3 * flow_offsets[2]
            + rng.uniform(-0.2, 0.2)
        )
        requested_inlet = inlet_condition if inlet_valve_actual else 0.0
        requested_transfer = transfer_condition if pump1_actual else 0.0
        requested_outlet = outlet_demand if pump2_actual else 0.0

        # Limit real flows BEFORE the update; do not clip or reset tank levels.
        # A full tank accepts only water displaced by simultaneous outflow.
        # This idealises backpressure/blocked inflow and omits overflow paths.
        outlet_flow = min(requested_outlet, tank2_l / TIMESTEP_MIN)
        transfer_flow = min(
            requested_transfer, tank1_l / TIMESTEP_MIN,
            (TANK_CAPACITY_L - tank2_l) / TIMESTEP_MIN + outlet_flow,
        )
        inlet_flow = min(
            requested_inlet,
            (TANK_CAPACITY_L - tank1_l) / TIMESTEP_MIN + transfer_flow,
        )
        tank1_l += (inlet_flow - transfer_flow) * TIMESTEP_MIN
        tank2_l += (transfer_flow - outlet_flow) * TIMESTEP_MIN

        # Truthful pressure follows PHYSICAL transfer-pump operation, including
        # during a failure with an ON command and a stationary physical pump.
        pressure_bar = (2.25 if pump1_actual else 0.35) + rng.uniform(-0.04, 0.04)
        row = {
            "timestamp": START_TIME + pd.Timedelta(minutes=index + 1),
            "tank1_level": 100.0 * tank1_l / TANK_CAPACITY_L,
            "tank2_level": 100.0 * tank2_l / TANK_CAPACITY_L,
            "inlet_flow": inlet_flow,
            "transfer_flow": transfer_flow,
            "outlet_flow": outlet_flow,
            "pressure": pressure_bar,
            # Legacy state columns consistently represent actual behaviour.
            "inlet_valve": int(inlet_valve_actual),
            "pump1": int(pump1_actual),
            "pump2": int(pump2_actual),
        }
        if events is not None:
            labelled_event = active_event or recovery_event
            row.update({
                "inlet_valve_command": int(inlet_valve_command),
                "inlet_valve_actual": int(inlet_valve_actual),
                "pump1_command": int(pump1_command),
                "pump1_actual": int(pump1_actual),
                "pump2_command": int(pump2_command),
                "pump2_actual": int(pump2_actual),
                "label": int(labelled_event is not None),
                "anomaly_type": labelled_event.anomaly_type if labelled_event else "normal",
                "event_id": labelled_event.event_id if labelled_event else 0,
                "event_phase": "active" if active_event else "recovery" if recovery_event else "normal",
            })
        rows.append(row)

        # Recovery remains abnormal through 15 consecutive healthy minutes in
        # the normal level envelope. At these volumes no flow limit is active.
        # Requiring an operating device excludes the known all-off idle state.
        if recovery_event is not None:
            normal_state = (
                400.0 <= tank1_l <= 800.0 and 400.0 <= tank2_l <= 800.0
                and (inlet_valve_command or pump1_command or pump2_command)
            )
            recovered_minutes = recovered_minutes + 1 if normal_state else 0
            if recovered_minutes >= RECOVERY_CONFIRM_MIN:
                recovery_event = None

    return pd.DataFrame.from_records(rows)


def event_manifest(data: pd.DataFrame, plans: list[Event]) -> pd.DataFrame:
    """Report physical event intervals separately from labelled recovery."""
    records = []
    for plan in plans:
        group = data.loc[data.event_id.eq(plan.event_id)]
        active = group.loc[group.event_phase.eq("active")]
        recovery = group.loc[group.event_phase.eq("recovery")]
        tail = recovery.tail(RECOVERY_CONFIRM_MIN)
        settled = (
            len(tail) == RECOVERY_CONFIRM_MIN
            and tail.tank1_level.between(40, 80).all()
            and tail.tank2_level.between(40, 80).all()
            and tail[["inlet_valve_command", "pump1_command", "pump2_command"]].sum(axis=1).gt(0).all()
        )
        if len(active) != plan.duration or not settled:
            raise RuntimeError(f"Event {plan.event_id} did not complete and recover in this run.")
        records.append({
            "event_id": plan.event_id,
            "anomaly_type": plan.anomaly_type,
            "planned_start": START_TIME + pd.Timedelta(minutes=plan.planned_start),
            "start": active.timestamp.iloc[0] - pd.Timedelta(minutes=1),
            "end_exclusive": active.timestamp.iloc[-1],
            "duration_minutes": len(active),
            "recovery_end_exclusive": recovery.timestamp.iloc[-1],
            "recovery_minutes": len(recovery),
        })
    return pd.DataFrame.from_records(records)


def print_summary(data: pd.DataFrame) -> None:
    """Print complete counts and continuous-variable sample statistics."""
    count = len(data)
    abnormal = int(data.label.sum())
    print(f"Total rows: {count:,}")
    print(f"Normal rows: {count - abnormal:,} ({100 * (count - abnormal) / count:.2f}%)")
    print(f"Abnormal rows (including recovery): {abnormal:,} ({100 * abnormal / count:.2f}%)")
    summary = []
    for kind in ("normal", *ANOMALY_TYPES):
        group = data.loc[data.anomaly_type.eq(kind)]
        active = group.loc[group.event_phase.eq("active")]
        summary.append({
            "anomaly_type": kind, "rows": len(group),
            "percent": 100 * len(group) / count,
            "active_rows": len(active),
            "recovery_rows": int(group.event_phase.eq("recovery").sum()),
            "events": active.event_id.nunique(),
        })
    print("\nRows by type; event counts refer to physical fault intervals:")
    print(pd.DataFrame(summary).set_index("anomaly_type").to_string(float_format=lambda x: f"{x:.2f}"))
    stats = data[list(CONTINUOUS_COLUMNS)].agg(["min", "max", "mean", "std"]).T
    print("\nFull-dataset continuous statistics (sample standard deviation, ddof=1):")
    print(stats.to_string(float_format=lambda x: f"{x:.4f}"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=(1, 2), default=2)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or Path(__file__).resolve().parent / (
        "output/stage2" if args.stage == 2 else "output"
    )
    try:
        plans = generate_event_plan(args.steps, args.seed) if args.stage == 2 else None
        data = simulate(args.steps, args.seed, events=plans)
        manifest = event_manifest(data, plans) if plans is not None else None
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    output_dir.mkdir(parents=True, exist_ok=True)
    name = "water_treatment_stage2.csv" if args.stage == 2 else "normal_operation.csv"
    data.to_csv(output_dir / name, index=False)
    if manifest is not None:
        manifest.to_csv(output_dir / "events.csv", index=False)
    save_plots(data, output_dir)
    print(f"Wrote {len(data):,} one-minute rows to {output_dir / name}")
    print(f"Wrote {'seven' if args.stage == 2 else 'four'} plots to {output_dir}")
    if args.stage == 2:
        print_summary(data)
    else:
        stats = data.select_dtypes(include="number").agg(["min", "max", "mean", "std"]).T
        print("\nFull-dataset statistics (sample standard deviation, ddof=1):")
        print(stats.to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
