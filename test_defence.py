"""Stage 5 checks; all Random Forest training is explicitly prohibited."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from evaluate_attacks import DEFAULT_DATA, DEFAULT_STAGE3, frozen_hashes, load_frozen_assets
from evaluate_defence import (
    DEFAULT_STAGE4, comparison_records, load_attack_archive, normal_training_inputs,
    reported_inputs, rule_records, stage4_hashes,
)
from physical_consistency import (
    BALANCE_CHECKS, QUANTILE, RULE_COLUMNS, RULES, STATE_CHECKS,
    apply_consistency, calibrate_thresholds, hybrid_prediction,
)
from train_baseline import FEATURES


class ConsistencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.no_fit = patch("sklearn.ensemble.RandomForestClassifier.fit",
                           side_effect=AssertionError("Stage 5 must not train the RF"))
        cls.fit_guard = cls.no_fit.start()
        cls.addClassCleanup(cls.no_fit.stop)
        cls.assets = load_frozen_assets(DEFAULT_DATA, DEFAULT_STAGE3)
        cls.stage4_before = stage4_hashes(DEFAULT_STAGE4)
        cls.normal = normal_training_inputs(cls.assets.train)
        cls.thresholds = calibrate_thresholds(cls.normal)
        cls.threshold_snapshot = cls.thresholds.copy(deep=True)
        cls.test_snapshot = cls.assets.test.copy(deep=True)
        cls.archive = load_attack_archive(cls.assets, DEFAULT_STAGE4)

    @classmethod
    def tearDownClass(cls):
        if cls.fit_guard.called:
            raise AssertionError("A forest training call occurred")
        if frozen_hashes(DEFAULT_DATA, DEFAULT_STAGE3) != cls.assets.hashes:
            raise AssertionError("A frozen Stage 2/3 artifact changed")
        if stage4_hashes(DEFAULT_STAGE4) != cls.stage4_before:
            raise AssertionError("A Stage 4 attack artifact changed")
        pd.testing.assert_frame_equal(cls.assets.test, cls.test_snapshot, check_exact=True)
        pd.testing.assert_frame_equal(cls.thresholds, cls.threshold_snapshot, check_exact=True)

    def fixture(self, rows=5, on=True):
        """Physically balanced stream at learned state medians, not test values."""
        frame = pd.DataFrame(0.0, index=range(rows), columns=FEATURES)
        frame.loc[:, list(FEATURES[6:])] = int(on)
        for rule, (feature, _, _) in STATE_CHECKS.items():
            threshold = self.thresholds.loc[self.thresholds.rule.eq(rule)
                                            & self.thresholds.actuator_state.eq(int(on))].iloc[0]
            frame[feature] = threshold.center
        frame["tank1_level"] = 60.0 + np.arange(rows) * 0.1 * (frame.inlet_flow - frame.transfer_flow)
        frame["tank2_level"] = 60.0 + np.arange(rows) * 0.1 * (frame.transfer_flow - frame.outlet_flow)
        return frame

    def group(self, experiment, configuration, alpha):
        return self.archive.loc[self.archive.experiment.eq(experiment)
                                & self.archive.configuration.eq(configuration) & self.archive.alpha.eq(alpha)]

    def test_calibration_is_training_only_with_fixed_99th_percentile(self):
        self.assertEqual(QUANTILE, 0.99)
        self.assertEqual(len(self.normal), 5576)
        self.assertEqual(len(self.thresholds), 10)
        train = self.assets.train.copy()
        train.loc[train.label.eq(1), list(FEATURES[:6])] = 1e6
        pd.testing.assert_frame_equal(calibrate_thresholds(normal_training_inputs(train)), self.thresholds)
        for _, row in self.thresholds.loc[self.thresholds.actuator.ne("none")].iterrows():
            sample = self.normal.loc[self.normal[row.actuator].eq(row.actuator_state), row.feature]
            self.assertEqual(row.training_samples, len(sample))
            self.assertEqual(row.center, sample.median())
            raw = (sample - row.center).abs().quantile(0.99)
            self.assertEqual(row.absolute_residual_quantile, raw)
            self.assertEqual(row.threshold, max(raw, row.numerical_floor))

    def test_balance_calibration_excludes_pairs_crossing_abnormal_intervals(self):
        train = self.assets.train
        normal = train.label.eq(0)
        pairs = normal & normal.shift(1, fill_value=False)
        for rule, (level, incoming, outgoing) in BALANCE_CHECKS.items():
            full = train[level].diff() - 0.1 * (train[incoming] - train[outgoing])
            raw = full.loc[pairs].abs().quantile(0.99)
            threshold = self.thresholds.loc[self.thresholds.rule.eq(rule)].iloc[0]
            self.assertEqual(threshold.training_samples, 5564)
            self.assertEqual(threshold.training_samples, pairs.sum())
            self.assertEqual(threshold.absolute_residual_quantile, raw)
        self.assertLess(pairs.sum(), len(self.normal) - 1)

    def test_metadata_and_labels_are_rejected_at_rule_engine_boundary(self):
        inputs = self.fixture()
        for forbidden in ("label", "anomaly_type", "event_id", "event_phase", "timestamp",
                          "inlet_valve_command", "pump1_command", "pump2_command"):
            with self.subTest(forbidden=forbidden):
                contaminated = inputs.assign(**{forbidden: 0})
                with self.assertRaises(ValueError):
                    apply_consistency(contaminated, self.thresholds)
                with self.assertRaises(ValueError):
                    calibrate_thresholds(contaminated)

    def test_off_actuators_cannot_report_on_flow_or_pressure(self):
        frame = self.fixture(on=False)
        self.assertTrue(apply_consistency(frame, self.thresholds).consistency_alert.eq(0).all())
        for rule, (feature, _, _) in STATE_CHECKS.items():
            modified = frame.copy()
            on_reference = self.thresholds.loc[self.thresholds.rule.eq(rule)
                                               & self.thresholds.actuator_state.eq(1), "center"].iloc[0]
            modified.loc[0, feature] = on_reference
            checked = apply_consistency(modified, self.thresholds)
            self.assertEqual(checked.loc[0, f"rule_{rule}"], 1)

    def test_on_actuators_use_their_own_conditional_distribution(self):
        frame = self.fixture()
        self.assertTrue(apply_consistency(frame, self.thresholds).consistency_alert.eq(0).all())
        for rule, (feature, _, _) in STATE_CHECKS.items():
            modified = frame.copy()
            off_reference = self.thresholds.loc[self.thresholds.rule.eq(rule)
                                                & self.thresholds.actuator_state.eq(0), "center"].iloc[0]
            modified.loc[0, feature] = off_reference
            self.assertEqual(apply_consistency(modified, self.thresholds).loc[0, f"rule_{rule}"], 1)

    def test_water_balance_uses_current_flows_and_correct_percent_conversion(self):
        frame = self.fixture()
        checked = apply_consistency(frame, self.thresholds)
        self.assertTrue(checked.consistency_alert.eq(0).all())
        expected = 0.1 * (frame.inlet_flow - frame.transfer_flow)
        np.testing.assert_allclose(checked.tank1_balance_expected_change, expected, atol=1e-14, rtol=0)
        modified = frame.copy()
        modified.loc[2, "tank1_level"] += 1.0
        changed = apply_consistency(modified, self.thresholds)
        self.assertAlmostEqual(changed.loc[2, "tank1_balance_residual"], 1.0)
        self.assertAlmostEqual(changed.loc[3, "tank1_balance_residual"], -1.0)
        self.assertEqual(changed.rule_tank1_balance.tolist(), [0, 0, 1, 1, 0])
        self.assertTrue(changed.rule_tank2_balance.eq(0).all())

    def test_balance_warmup_and_missing_samples_never_bridge_a_gap(self):
        frame = self.fixture(rows=3)
        frame.index = [0, 1, 4]
        frame.loc[4, "tank1_level"] = 95
        checked = apply_consistency(frame, self.thresholds)
        self.assertEqual(checked.balance_pair_available.tolist(), [0, 1, 0])
        self.assertTrue(np.isnan(checked.loc[0, "tank1_balance_residual"]))
        self.assertTrue(np.isnan(checked.loc[4, "tank1_balance_residual"]))
        self.assertTrue(checked.rule_tank1_balance.eq(0).all())

    def test_balance_uses_reported_adversarial_history_and_never_hidden_truth(self):
        rows = self.group("B", "top_6", 1)
        inputs = reported_inputs(rows)
        checked = apply_consistency(inputs, self.thresholds)
        expected = inputs.tank1_level.diff() - 0.1 * (inputs.inlet_flow - inputs.transfer_flow)
        np.testing.assert_allclose(checked.tank1_balance_residual, expected, atol=1e-14, rtol=0, equal_nan=True)
        consecutive_attacked = rows.attack_eligible.eq(1) & rows.attack_eligible.shift(1).eq(1)
        self.assertTrue(checked.loc[consecutive_attacked.to_numpy(), "tank1_balance_observed_change"].eq(0).all())

    def test_rule_evaluation_is_causal(self):
        frame = self.fixture()
        before = apply_consistency(frame, self.thresholds)
        frame.loc[4, "tank1_level"] = 100
        after = apply_consistency(frame, self.thresholds)
        pd.testing.assert_frame_equal(before.iloc[:4], after.iloc[:4], check_exact=True)

    def test_hybrid_is_or_and_asr_uses_original_rf_true_positives(self):
        test = pd.DataFrame({"label": [0, 0, 1, 1, 1, 1, 1],
                             "anomaly_type": ["normal"] * 2 + ["transfer_pump_failure"] * 5,
                             "event_phase": ["normal"] * 2 + ["active"] * 3 + ["recovery"] * 2})
        baseline = np.array([0, 1, 1, 1, 1, 0, 0])
        forest = np.array([0, 1, 0, 0, 1, 0, 1])
        alerts = np.array([1, 0, 1, 0, 0, 1, 0])
        defended = hybrid_prediction(forest, alerts)
        np.testing.assert_array_equal(defended, [1, 1, 1, 0, 1, 1, 1])
        records = comparison_records(test, baseline, forest, defended, test.label.eq(1).to_numpy(),
                                     pd.DataFrame({"consistency_alert": alerts}), {})
        result = pd.DataFrame(records).set_index("scope").loc["all_abnormal"]
        self.assertEqual(result.baseline_tp_attacked, 3)
        self.assertEqual(result.undefended_success_count, 2)
        self.assertEqual(result.defended_success_count, 1)
        self.assertAlmostEqual(result.defended_attack_success_rate, 1 / 3)
        self.assertEqual(result.rf_false_negatives_caught, 2)
        self.assertEqual(result.newly_flagged_normal_rows, 1)
        self.assertEqual(result.defended_fp, 2)
        self.assertEqual(result.defended_tp, 4)
        self.assertTrue(np.isnan(pd.DataFrame(records).set_index("scope").loc["recovery", "defended_attack_success_rate"]))

    def test_fault_specific_denominator_and_overlap_counts(self):
        rows = self.group("C", "inlet_valve_stuck_open", 1)
        checks = apply_consistency(reported_inputs(rows), self.thresholds)
        forest = rows.adversarial_prediction.to_numpy()
        eligible = rows.attack_eligible.to_numpy(dtype=bool)
        defended = hybrid_prediction(forest, checks.consistency_alert)
        records = comparison_records(self.assets.test, self.assets.baseline_prediction, forest,
                                     defended, eligible, checks, {})
        result = pd.DataFrame(records).set_index("scope")
        self.assertEqual(result.loc["all_abnormal", "baseline_tp_attacked"], 164)
        self.assertEqual(result.loc["inlet_valve_stuck_open", "baseline_tp_attacked"], 164)
        self.assertEqual(result.loc["outlet_pump_failure", "baseline_tp_attacked"], 0)
        self.assertTrue(np.isnan(result.loc["outlet_pump_failure", "defended_attack_success_rate"]))
        rules = pd.DataFrame(rule_records(self.assets.test, self.assets.baseline_prediction, forest,
                                          eligible, checks, {}, checks)).set_index(["scope", "rule"])
        multiple = rules.loc[("all_abnormal", "multiple_rules"), "targeted_abnormal_rows_caught"]
        self.assertEqual(multiple, np.count_nonzero(eligible & checks.consistency_rule_count.gt(1).to_numpy()))
        self.assertTrue(rules.new_rule_alerts_on_targeted_abnormal.eq(0).all())

    def test_every_zero_strength_control_is_the_same_truthful_evaluation(self):
        truthful = apply_consistency(self.assets.test.loc[:, list(FEATURES)], self.thresholds)
        controls = self.archive.loc[self.archive.alpha.eq(0)]
        self.assertEqual(controls.groupby(["experiment", "configuration"]).ngroups, 13)
        for _, rows in controls.groupby(["experiment", "configuration"], sort=False):
            checks = apply_consistency(reported_inputs(rows), self.thresholds)
            np.testing.assert_array_equal(checks[list(RULE_COLUMNS)], truthful[list(RULE_COLUMNS)])
            np.testing.assert_array_equal(rows.adversarial_prediction, self.assets.baseline_prediction)
            self.assertEqual(checks.balance_pair_available.iloc[0], 0)

    def test_inputs_and_thresholds_unchanged_and_evaluation_reproducible(self):
        rows = self.group("B", "top_6", 0.75)
        original = rows.copy(deep=True)
        inputs = reported_inputs(rows)
        input_snapshot = inputs.copy(deep=True)
        first = apply_consistency(inputs, self.thresholds)
        second = apply_consistency(inputs, self.thresholds)
        pd.testing.assert_frame_equal(first, second, check_exact=True)
        pd.testing.assert_frame_equal(inputs, input_snapshot, check_exact=True)
        pd.testing.assert_frame_equal(rows, original, check_exact=True)
        np.testing.assert_array_equal(first.consistency_rule_count, first[list(RULE_COLUMNS)].sum(axis=1))
        np.testing.assert_array_equal(first.consistency_alert, first.consistency_rule_count.gt(0).astype(int))


if __name__ == "__main__":
    unittest.main()
