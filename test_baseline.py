"""Checks for leakage prevention, metric definitions and model reproducibility."""

import unittest

import numpy as np
import pandas as pd

from train_baseline import (
    ANOMALY_TYPES, COMMAND_FEATURES, DEFAULT_DATA, FEATURES, FORBIDDEN_FEATURES,
    binary_metrics, event_aware_split, fit_forest, load_dataset,
    per_anomaly_metrics, phase_metrics, validate_dataset,
)


def small_dataset():
    """Three contiguous events, including recovery, around possible cut points."""
    data = pd.DataFrame({name: np.zeros(30) for name in FEATURES + COMMAND_FEATURES})
    data["timestamp"] = pd.date_range("2026-01-01", periods=30, freq="min")
    data["label"] = 0
    data["event_id"] = 0
    data["anomaly_type"] = "normal"
    data["event_phase"] = "normal"
    for event_id, start, end, active_end in [(1, 3, 5, 4), (2, 19, 22, 20), (3, 27, 28, 27)]:
        data.loc[start:end, ["label", "event_id"]] = [1, event_id]
        data.loc[start:end, "anomaly_type"] = "transfer_pump_failure"
        data.loc[start:active_end, "event_phase"] = "active"
        data.loc[active_end + 1:end, "event_phase"] = "recovery"
    return data


class BaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_dataset(DEFAULT_DATA)
        cls.train, cls.test = event_aware_split(cls.data)
        cls.model = fit_forest(cls.train)

    def test_default_split_keeps_whole_events_and_recovery(self):
        self.assertEqual((len(self.train), len(self.test)), (7056, 3024))
        self.assertLess(self.train.timestamp.max(), self.test.timestamp.min())
        train_ids = set(self.train.loc[self.train.event_id.gt(0), "event_id"])
        test_ids = set(self.test.loc[self.test.event_id.gt(0), "event_id"])
        self.assertEqual(train_ids, set(range(1, 12)))
        self.assertEqual(test_ids, {12, 13, 14, 15})
        self.assertFalse(train_ids & test_ids)
        for part in (self.train, self.test):
            self.assertTrue(set(ANOMALY_TYPES) <= set(part.anomaly_type))
        self.assertEqual(len(self.test.loc[self.test.event_id.eq(15)]), 176)

    def test_cut_moves_before_an_event_instead_of_splitting_it(self):
        data = small_dataset()
        validate_dataset(data)
        train, test = event_aware_split(data, train_fraction=20 / 30)
        self.assertEqual(len(train), 19)
        self.assertTrue(test.loc[test.event_id.eq(2), "event_phase"].eq("recovery").any())
        self.assertFalse(train.event_id.eq(2).any())

    def test_cut_can_move_past_recovery_as_one_event(self):
        train, test = event_aware_split(small_dataset(), train_fraction=22 / 30)
        self.assertEqual(len(train), 23)
        self.assertEqual(train.event_id.eq(2).sum(), 4)
        self.assertFalse(test.event_id.eq(2).any())

    def test_validation_rejects_repeated_event_blocks(self):
        data = small_dataset()
        data.loc[10, ["label", "event_id", "anomaly_type", "event_phase"]] = [
            1, 1, "transfer_pump_failure", "active",
        ]
        with self.assertRaisesRegex(ValueError, "contiguous"):
            validate_dataset(data)

    def test_validation_rejects_reversed_time(self):
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            validate_dataset(small_dataset().iloc[::-1].reset_index(drop=True))

    def test_features_cannot_include_ground_truth_or_unknown_columns(self):
        for name in (*FORBIDDEN_FEATURES, "pump1", "source_row"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                fit_forest(self.train, FEATURES + (name,))
        self.assertEqual(tuple(self.model.feature_names_in_), FEATURES)
        self.assertEqual(self.model.n_features_in_, 9)
        self.assertTrue(set(FEATURES).isdisjoint(FORBIDDEN_FEATURES))
        self.assertTrue(set(FEATURES).isdisjoint(COMMAND_FEATURES))

    def test_metric_denominators_and_confusion_matrix_orientation(self):
        truth = [0] * 10 + [1] * 10
        prediction = [0] * 8 + [1] * 2 + [0] * 3 + [1] * 7
        result = binary_metrics(truth, prediction)
        self.assertEqual([result[x] for x in ("tn", "fp", "fn", "tp")], [8, 2, 3, 7])
        for name, expected in {
            "accuracy": 0.75, "precision": 7 / 9, "recall": 0.7,
            "f1": 14 / 19, "false_positive_rate": 0.2, "false_negative_rate": 0.3,
        }.items():
            self.assertAlmostEqual(result[name], expected)

    def test_per_anomaly_uses_shared_normals_without_relabelling_other_faults(self):
        data = small_dataset()
        data.loc[data.event_id.eq(2), "anomaly_type"] = "inlet_valve_stuck_open"
        data.loc[data.event_id.eq(3), "anomaly_type"] = "outlet_pump_failure"
        prediction = data.label.to_numpy().copy()
        prediction[0] = 1  # One shared normal false alarm.
        prediction[[5, 22, 28]] = 0  # One recovery miss for each type.
        metrics = per_anomaly_metrics(data, prediction).set_index("anomaly_type")
        for kind, positive_rows in zip(ANOMALY_TYPES, [3, 4, 2]):
            self.assertEqual(metrics.loc[kind, "fp"], 1)
            self.assertEqual(metrics.loc[kind, "tn"], 20)
            self.assertEqual(metrics.loc[kind, "fn"], 1)
            self.assertEqual(metrics.loc[kind, "positive_rows"], positive_rows)
            self.assertAlmostEqual(metrics.loc[kind, "recall"], (positive_rows - 1) / positive_rows)
        phase = phase_metrics(data, prediction).set_index("event_phase")
        self.assertEqual(phase.loc["active", "recall"], 1)
        self.assertEqual(phase.loc["recovery", "recall"], 0.25)

    def test_random_seed_reproduces_predictions(self):
        second = fit_forest(self.train)
        inputs = self.test[list(FEATURES)]
        np.testing.assert_array_equal(self.model.predict(inputs), second.predict(inputs))
        # Parallel accumulation can change floating point sums by tiny amounts.
        np.testing.assert_allclose(self.model.predict_proba(inputs), second.predict_proba(inputs),
                                   atol=1e-12, rtol=0)


if __name__ == "__main__":
    unittest.main()
