"""Stage 4 checks: no training, no protected-field edits, correct evasion counts."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from evaluate_attacks import (
    ACTUATORS, ALPHAS, BOUNDS, DEFAULT_DATA, DEFAULT_STAGE3, FAULT_GROUPS,
    SENSORS, Attack, build_attacks, frozen_hashes, load_frozen_assets,
    manipulate, score_attack, training_references,
)
from train_baseline import FEATURES


class AttackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Fail if this stage accidentally invokes any forest training method.
        cls.no_training = patch(
            "sklearn.ensemble.RandomForestClassifier.fit",
            side_effect=AssertionError("Stage 4 must never train a Random Forest"),
        )
        cls.training_spy = cls.no_training.start()
        cls.addClassCleanup(cls.no_training.stop)
        cls.assets = load_frozen_assets(DEFAULT_DATA, DEFAULT_STAGE3)
        cls.original_test = cls.assets.test.copy(deep=True)
        cls.references = training_references(cls.assets.train).set_index("sensor").normal_median
        cls.attacks = build_attacks(cls.assets.ranking)

    @classmethod
    def tearDownClass(cls):
        if cls.training_spy.called:
            raise AssertionError("A training method was called")
        if frozen_hashes(DEFAULT_DATA, DEFAULT_STAGE3) != cls.assets.hashes:
            raise AssertionError("A frozen source file changed")
        pd.testing.assert_frame_equal(cls.assets.test, cls.original_test, check_exact=True)

    def test_all_prescribed_groups_and_strengths(self):
        self.assertEqual(ALPHAS, (0, 0.25, 0.5, 0.75, 1))
        self.assertEqual([sum(a.experiment == e for a in self.attacks) for e in "ABC"], [6, 4, 3])
        self.assertEqual(self.assets.ranking[:3], ("tank1_level", "outlet_flow", "tank2_level"))
        for attack in self.attacks:
            self.assertTrue(set(attack.sensors) <= set(SENSORS))
            if attack.experiment == "C":
                self.assertEqual(attack.sensors, FAULT_GROUPS[attack.target_type])

    def test_references_use_normal_training_rows_only(self):
        train = self.assets.train.copy()
        normal = train.label.eq(0)
        expected = train.loc[normal, list(SENSORS)].median()
        np.testing.assert_array_equal(self.references.loc[list(SENSORS)], expected)
        # Changing excluded abnormal training values cannot affect references.
        train.loc[~normal, list(SENSORS)] = 1e6
        result = training_references(train).set_index("sensor")
        np.testing.assert_array_equal(result.normal_median.loc[list(SENSORS)], expected)
        self.assertTrue(result.normal_training_rows.eq(5576).all())
        self.assertLess(self.assets.train.timestamp.max(), self.assets.test.timestamp.min())

    def test_zero_strength_is_exactly_the_original_baseline(self):
        attack = next(a for a in self.attacks if a.experiment == "B" and len(a.sensors) == 6)
        adversarial, _ = manipulate(self.assets.test, attack, self.references, 0)
        pd.testing.assert_frame_equal(adversarial, self.assets.test, check_exact=True)
        np.testing.assert_array_equal(
            self.assets.model.predict(adversarial[list(FEATURES)]), self.assets.baseline_prediction,
        )

    def test_interpolation_formula_and_exact_reference_endpoint(self):
        attack = Attack("A", "tank1_level", ("tank1_level",))
        test = self.assets.test
        manipulated, eligible = manipulate(test, attack, self.references, 0.25)
        expected = test.loc[eligible, "tank1_level"] + 0.25 * (
            self.references["tank1_level"] - test.loc[eligible, "tank1_level"]
        )
        np.testing.assert_array_equal(manipulated.loc[eligible, "tank1_level"], expected)
        full, _ = manipulate(test, attack, self.references, 1)
        self.assertTrue(full.loc[eligible, "tank1_level"].eq(self.references["tank1_level"]).all())

    def test_every_configuration_preserves_states_labels_order_and_untargeted_values(self):
        test = self.assets.test
        for attack in self.attacks:
            for alpha in ALPHAS:
                with self.subTest(attack=attack.configuration, alpha=alpha):
                    adversarial, eligible = manipulate(test, attack, self.references, alpha)
                    protected = [c for c in test if c not in attack.sensors]
                    pd.testing.assert_frame_equal(adversarial[protected], test[protected], check_exact=True)
                    pd.testing.assert_frame_equal(adversarial.loc[~eligible], test.loc[~eligible], check_exact=True)
                    np.testing.assert_array_equal(adversarial.index, test.index)
                    for sensor, (low, high) in BOUNDS.items():
                        self.assertTrue(adversarial[sensor].between(low, high).all())
                    if attack.target_type is not None:
                        self.assertTrue(test.loc[eligible, "anomaly_type"].eq(attack.target_type).all())

    def test_forbidden_features_and_invalid_strengths_are_rejected(self):
        for name in (*ACTUATORS, "label", "event_id", "timestamp"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                manipulate(self.assets.test, Attack("A", name, (name,)), self.references, 0.5)
        for alpha in (-0.01, 1.01):
            with self.assertRaises(ValueError):
                manipulate(self.assets.test, self.attacks[0], self.references, alpha)

    def test_asr_excludes_preexisting_misses_and_records_newly_detected_rows(self):
        test = pd.DataFrame({
            "label": [0] * 3 + [1] * 6,
            "anomaly_type": ["normal"] * 3 + ["transfer_pump_failure"] * 6,
            "event_phase": ["normal"] * 3 + ["active"] * 4 + ["recovery"] * 2,
        })
        baseline = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0])
        prediction = np.array([0, 0, 1, 0, 0, 0, 1, 1, 0])
        eligible = test.label.eq(1).to_numpy()
        records = score_attack(test, baseline, prediction, eligible, eligible,
                               self.attacks[0], 0.75)
        by_scope = pd.DataFrame(records).set_index("scope")
        row = by_scope.loc["all_abnormal"]
        self.assertEqual(row.baseline_tp_attacked, 4)
        self.assertEqual(row.previously_detected_became_fn, 3)
        self.assertEqual(row.attack_success_rate, 0.75)
        self.assertEqual(row.newly_detected_abnormal_rows, 1)
        self.assertEqual(row.fn, 4)
        self.assertAlmostEqual(row.false_negative_rate, 4 / 6)
        self.assertEqual(row.precision, 2 / 3)
        self.assertTrue(np.isnan(by_scope.loc["recovery", "attack_success_rate"]))

    def test_fault_specific_asr_uses_only_targeted_baseline_true_positives(self):
        attack = next(a for a in self.attacks if a.target_type == "inlet_valve_stuck_open")
        _, eligible = manipulate(self.assets.test, attack, self.references, 0)
        records = score_attack(
            self.assets.test, self.assets.baseline_prediction, self.assets.baseline_prediction,
            eligible, np.zeros(len(eligible), dtype=bool), attack, 0,
        )
        by_scope = pd.DataFrame(records).set_index("scope")
        self.assertEqual(by_scope.loc["all_abnormal", "baseline_tp_attacked"], 164)
        self.assertEqual(by_scope.loc["all_abnormal", "attack_success_rate"], 0)
        self.assertEqual(by_scope.loc["transfer_pump_failure", "baseline_tp_attacked"], 0)
        self.assertTrue(np.isnan(by_scope.loc["transfer_pump_failure", "attack_success_rate"]))

    def test_attack_and_predictions_are_reproducible_without_training(self):
        attack = next(a for a in self.attacks if a.experiment == "B" and len(a.sensors) == 6)
        first, _ = manipulate(self.assets.test, attack, self.references, 0.75)
        second, _ = manipulate(self.assets.test, attack, self.references, 0.75)
        pd.testing.assert_frame_equal(first, second, check_exact=True)
        np.testing.assert_array_equal(
            self.assets.model.predict(first[list(FEATURES)]),
            self.assets.model.predict(second[list(FEATURES)]),
        )


if __name__ == "__main__":
    unittest.main()
