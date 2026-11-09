"""Overview and event-detail plots of truthful simulated process measurements."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd


EVENT_COLOURS = {
    "transfer_pump_failure": "tab:red",
    "inlet_valve_stuck_open": "tab:purple",
    "outlet_pump_failure": "tab:brown",
}


def shade_events(ax: plt.Axes, data: pd.DataFrame) -> None:
    """Use stronger shading for faults and lighter shading for their recovery."""
    if "event_phase" not in data.columns:
        return
    for (_, phase), group in data.loc[data.label.eq(1)].groupby(["event_id", "event_phase"]):
        ax.axvspan(
            group.timestamp.iloc[0] - pd.Timedelta(minutes=1),
            group.timestamp.iloc[-1],
            color=EVENT_COLOURS[group.anomaly_type.iloc[0]],
            alpha=0.20 if phase == "active" else 0.07,
            linewidth=0,
        )


def finish_plot(fig: plt.Figure, data: pd.DataFrame, path: Path) -> None:
    """Add an event colour key and write a complete, standalone figure."""
    if "event_phase" in data.columns:
        kinds = set(data.loc[data.label.eq(1), "anomaly_type"])
        handles = [
            Patch(facecolor=colour, alpha=0.20, label=kind.replace("_", " "))
            for kind, colour in EVENT_COLOURS.items() if kind in kinds
        ]
        handles.append(Patch(facecolor="grey", alpha=0.07, label="lighter shading = recovery"))
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1), ncol=2)
        # Reserve a fixed height for the key, including on tall event figures.
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.6 / fig.get_figheight()))
    else:
        fig.tight_layout()
    fig.autofmt_xdate()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_states(axes, data: pd.DataFrame) -> None:
    """Compare controller commands to physical actuator states."""
    for ax, device, title in zip(
        axes, ("inlet_valve", "pump1", "pump2"),
        ("Inlet valve", "Transfer pump", "Outlet pump"),
    ):
        if f"{device}_command" in data.columns:
            ax.step(data.timestamp, data[f"{device}_actual"], where="pre",
                    label="Actual", linewidth=1.2, color="tab:blue")
            ax.step(data.timestamp, data[f"{device}_command"], where="pre",
                    label="Command", linewidth=1, linestyle="--", color="tab:orange")
        else:
            ax.step(data.timestamp, data[device], where="pre", linewidth=0.8)
        ax.set(ylabel=title, ylim=(-0.15, 1.15), yticks=(0, 1))
        ax.grid(alpha=0.25)
        shade_events(ax, data)


def save_plots(data: pd.DataFrame, output_dir: Path) -> None:
    """Save four overview plots and one before/during/after view per fault."""
    time = data.timestamp
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(time, data.tank1_level, label="Tank 1", linewidth=0.8)
    ax.plot(time, data.tank2_level, label="Tank 2", linewidth=0.8)
    ax.set(title="Tank levels", ylabel="Level (% of capacity)", ylim=(0, 103))
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25)
    shade_events(ax, data)
    finish_plot(fig, data, output_dir / "levels.png")

    fig, ax = plt.subplots(figsize=(14, 5))
    for column, label in (("inlet_flow", "Inlet"), ("transfer_flow", "Transfer"),
                          ("outlet_flow", "Outlet")):
        ax.plot(time, data[column], label=label, linewidth=0.8)
    ax.set(title="Water flows", ylabel="Flow (L/min)")
    ax.set_ylim(bottom=-0.5)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25)
    shade_events(ax, data)
    finish_plot(fig, data, output_dir / "flows.png")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(time, data.pressure, color="tab:green", linewidth=0.8)
    ax.set(title="Transfer-line pressure", ylabel="Pressure (bar)")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    shade_events(ax, data)
    finish_plot(fig, data, output_dir / "pressure.png")

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    plot_states(axes, data)
    axes[0].set_title("Actuator states (0 = off/closed, 1 = on/open)")
    if "event_phase" in data.columns:
        axes[0].legend(loc="upper right")
    finish_plot(fig, data, output_dir / "actuators.png")

    if "event_phase" in data.columns:
        for kind in EVENT_COLOURS:
            save_event_zoom(data, kind, output_dir)


def save_event_zoom(data: pd.DataFrame, kind: str, output_dir: Path) -> None:
    """Show the first event of a type, including its complete recovery."""
    matches = data.loc[data.anomaly_type.eq(kind)]
    if matches.empty:
        return
    event_id = matches.event_id.iloc[0]
    event = matches.loc[matches.event_id.eq(event_id)]
    start = event.timestamp.iloc[0] - pd.Timedelta(minutes=60)
    end = event.timestamp.iloc[-1] + pd.Timedelta(minutes=60)
    window = data.loc[data.timestamp.between(start, end)]
    fig, axes = plt.subplots(6, 1, figsize=(14, 13), sharex=True)
    axes[0].plot(window.timestamp, window.tank1_level, label="Tank 1")
    axes[0].plot(window.timestamp, window.tank2_level, label="Tank 2")
    axes[0].set(ylabel="Level (%)", ylim=(0, 103),
                title=f"{kind.replace('_', ' ').title()} — event {event_id}")
    axes[0].legend(loc="upper right")
    for column, label in (("inlet_flow", "Inlet"), ("transfer_flow", "Transfer"),
                          ("outlet_flow", "Outlet")):
        axes[1].plot(window.timestamp, window[column], label=label)
    axes[1].set(ylabel="Flow (L/min)", ylim=(-0.5, 31))
    axes[1].legend(loc="upper right")
    axes[2].plot(window.timestamp, window.pressure, color="tab:green")
    axes[2].set(ylabel="Pressure (bar)", ylim=(0, 2.5))
    for ax in axes[:3]:
        ax.grid(alpha=0.25)
        shade_events(ax, window)
    plot_states(axes[3:], window)
    axes[3].legend(loc="upper right")
    axes[-1].set_xlabel("Time (one-minute samples)")
    finish_plot(fig, window, output_dir / f"zoom_{kind}.png")
