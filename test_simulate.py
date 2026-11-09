"""Physical and labelling checks for the default Stage 2 experiment."""

import unittest

import numpy as np
import pandas as pd

from simulate import (
    ANOMALY_TYPES, DEFAULT_SEED, DEFAULT_STEPS, EVENTS_PER_TYPE,
    RECOVERY_CONFIRM_MIN, START_TIME, STABILISATION_MIN,
    event_manifest, generate_event_plan, simulate,
)


class Stage2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plans = generate_event_plan(DEFAULT_STEPS, DEFAULT_SEED)
        cls.data = simulate(events=cls.plans)
        cls.manifest = event_manifest(cls.data, cls.plans)

    def test_full_week_and_stabilisation(self):
        data = self.data
        self.assertEqual(len(data), 10080)
        self.assertEqual(data.timestamp.iloc[0], START_TIME + pd.Timedelta(minutes=1))
        self.assertEqual(data.timestamp.iloc[-1], START_TIME + pd.Timedelta(days=7))
        self.assertTrue(data.timestamp.diff().iloc[1:].eq(pd.Timedelta(minutes=1)).all())
        self.assertTrue(data.label.iloc[:STABILISATION_MIN].eq(0).all())
        self.assertTrue(np.isfinite(data.select_dtypes(include="number").to_numpy()).all())

    def test_conservation_and_capacity_during_faults(self):
        data = self.data
        for level, incoming, outgoing in (
            ("tank1_level", "inlet_flow", "transfer_flow"),
            ("tank2_level", "transfer_flow", "outlet_flow"),
        ):
            self.assertTrue(data[level].between(0, 100).all())
            changes = np.diff(np.r_[600.0, data[level].to_numpy() * 10])
            np.testing.assert_allclose(changes, data[incoming] - data[outgoing], atol=1e-9, rtol=0)
        self.assertTrue(data[["inlet_flow", "transfer_flow", "outlet_flow"]].ge(0).all().all())
        self.assertTrue(data.tank1_level.ge(99.99).any())

    def test_event_counts_durations_distribution_and_no_overlap(self):
        manifest = self.manifest
        self.assertEqual(len(manifest), 15)
        for kind in ANOMALY_TYPES:
            self.assertEqual(manifest.anomaly_type.eq(kind).sum(), EVENTS_PER_TYPE)
        self.assertTrue(manifest.duration_minutes.between(60, 180).all())
        self.assertEqual(manifest.start.dt.normalize().nunique(), 7)
        for previous, following in zip(manifest.itertuples(), manifest.iloc[1:].itertuples()):
            self.assertLessEqual(previous.recovery_end_exclusive, following.start)
        for plan, record in zip(self.plans, manifest.itertuples()):
            self.assertEqual(record.duration_minutes, plan.duration)
            self.assertGreaterEqual(record.start, record.planned_start)

    def test_command_actual_distinction_and_truthful_pressure(self):
        data = self.data
        for device, fault, failed_state, flow in (
            ("pump1", "transfer_pump_failure", 0, "transfer_flow"),
            ("pump2", "outlet_pump_failure", 0, "outlet_flow"),
            ("inlet_valve", "inlet_valve_stuck_open", 1, "inlet_flow"),
        ):
            failed = data.event_phase.eq("active") & data.anomaly_type.eq(fault)
            self.assertTrue(data.loc[failed, f"{device}_actual"].eq(failed_state).all())
            self.assertTrue(data.loc[~failed, f"{device}_actual"].eq(data.loc[~failed, f"{device}_command"]).all())
            self.assertTrue(data[device].eq(data[f"{device}_actual"]).all())
            self.assertTrue(data.loc[failed, f"{device}_command"].ne(data.loc[failed, f"{device}_actual"]).any())
            if failed_state == 0:
                self.assertTrue(data.loc[failed, flow].eq(0).all())
            self.assertTrue(data.loc[data[f"{device}_actual"].eq(0), flow].eq(0).all())
        running = data.pump1_actual.eq(1)
        self.assertTrue(data.loc[running, "pressure"].between(2.21, 2.29).all())
        self.assertTrue(data.loc[~running, "pressure"].between(0.31, 0.39).all())

    def test_each_event_has_expected_initial_physical_response(self):
        for _, group in self.data.loc[self.data.event_phase.eq("active")].groupby("event_id"):
            index = group.index[0]
            before, after = self.data.iloc[index - 1], self.data.iloc[index]
            if after.anomaly_type == "transfer_pump_failure":
                self.assertGreater(after.tank1_level, before.tank1_level)
                self.assertLess(after.tank2_level, before.tank2_level)
                self.assertLess(after.pressure, 0.4)
            elif after.anomaly_type == "outlet_pump_failure":
                self.assertGreater(after.tank2_level, before.tank2_level)
                self.assertGreater(after.transfer_flow, 0)
            else:
                self.assertGreater(after.tank1_level, before.tank1_level)
                self.assertEqual(after.inlet_valve_command, 0)
                self.assertEqual(after.inlet_valve_actual, 1)

    def test_recovery_and_normal_labels(self):
        data = self.data
        self.assertEqual(set(data.anomaly_type), {"normal", *ANOMALY_TYPES})
        self.assertTrue(data.label.eq(data.event_phase.ne("normal").astype(int)).all())
        normal = data.loc[data.label.eq(0)]
        self.assertTrue(normal.event_id.eq(0).all())
        self.assertTrue(normal.anomaly_type.eq("normal").all())
        self.assertTrue(normal[["tank1_level", "tank2_level"]].ge(40).all().all())
        self.assertTrue(normal[["tank1_level", "tank2_level"]].le(80).all().all())
        self.assertTrue(data[["inlet_valve_command", "pump1_command", "pump2_command"]].sum(axis=1).gt(0).all())
        self.assertTrue(0.10 <= data.label.mean() <= 0.20)
        for _, group in data.loc[data.label.eq(1)].groupby("event_id"):
            active = group.loc[group.event_phase.eq("active")]
            recovery = group.loc[group.event_phase.eq("recovery")]
            self.assertEqual(active.index[-1] + 1, recovery.index[0])
            self.assertEqual(group.anomaly_type.nunique(), 1)
            self.assertGreaterEqual(len(recovery), RECOVERY_CONFIRM_MIN)
            tail = recovery.tail(RECOVERY_CONFIRM_MIN)
            self.assertTrue(tail[["tank1_level", "tank2_level"]].ge(40).all().all())
            self.assertTrue(tail[["tank1_level", "tank2_level"]].le(80).all().all())
        residual = data.loc[data.event_phase.eq("recovery") & data.tank1_level.gt(80)]
        self.assertFalse(residual.empty)
        self.assertTrue(residual.label.eq(1).all())
        self.assertTrue(residual.anomaly_type.eq("inlet_valve_stuck_open").all())

    def test_fixed_seed_reproduces_events_and_dataset(self):
        plans = generate_event_plan(DEFAULT_STEPS, DEFAULT_SEED)
        self.assertEqual(plans, self.plans)
        self.assertEqual(simulate(events=plans).to_csv(index=False), self.data.to_csv(index=False))

    def test_normal_mode_unchanged_by_empty_fault_plan(self):
        normal = simulate()
        labelled = simulate(events=[])
        self.assertEqual(len(normal.columns), 10)
        pd.testing.assert_frame_equal(normal, labelled[normal.columns], check_exact=True)
        self.assertTrue(labelled.label.eq(0).all())


if __name__ == "__main__":
    unittest.main()
