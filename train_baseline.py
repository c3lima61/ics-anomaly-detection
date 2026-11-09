"""Stage 3: train and audit a supervised Random Forest on truthful process data."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


FEATURES = (
    "tank1_level", "tank2_level", "inlet_flow", "transfer_flow",
    "outlet_flow", "pressure", "inlet_valve_actual", "pump1_actual", "pump2_actual",
)
COMMAND_FEATURES = ("inlet_valve_command", "pump1_command", "pump2_command")
FORBIDDEN_FEATURES = frozenset({"label", "anomaly_type", "event_id", "event_phase", "timestamp"})
ANOMALY_TYPES = (
    "transfer_pump_failure", "inlet_valve_stuck_open", "outlet_pump_failure",
)
MODEL_PARAMETERS = {
    "n_estimators": 200,
    "random_state": 3015,
    "class_weight": "balanced",
    "min_samples_leaf": 2,
    "max_depth": None,
    "max_features": "sqrt",
    "bootstrap": True,
    "n_jobs": -1,
}
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = PROJECT_DIR / "output/stage2/water_treatment_stage2.csv"
DEFAULT_OUTPUT = PROJECT_DIR / "output/stage3"


def validate_dataset(data: pd.DataFrame) -> None:
    """Check ordering and event integrity before selecting model inputs."""
    required = set(FEATURES + COMMAND_FEATURES) | FORBIDDEN_FEATURES
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    if len(data) < 2 or data[list(required)].isna().any().any():
        raise ValueError("Dataset must contain complete, nonempty records.")
    if not data.timestamp.is_monotonic_increasing or not data.timestamp.is_unique:
        raise ValueError("Timestamps must be strictly increasing; rows are never shuffled.")
    if not data.timestamp.diff().iloc[1:].eq(pd.Timedelta(minutes=1)).all():
        raise ValueError("Expected consecutive one-minute records.")
    if set(data.label.unique()) != {0, 1}:
        raise ValueError("Dataset must contain both binary labels 0 and 1.")
    if not np.isfinite(data[list(FEATURES + COMMAND_FEATURES)].to_numpy(dtype=float)).all():
        raise ValueError("Model inputs must be finite numeric values.")
    for name in FEATURES[6:] + COMMAND_FEATURES:
        if not data[name].isin([0, 1]).all():
            raise ValueError(f"Actuator column {name} must be binary.")
    normal = data.label.eq(0)
    if not (data.loc[normal, "anomaly_type"].eq("normal").all()
            and data.loc[normal, "event_id"].eq(0).all()
            and data.loc[normal, "event_phase"].eq("normal").all()):
        raise ValueError("Normal-row labels and event metadata disagree.")
    abnormal = data.loc[~normal]
    if not (abnormal.event_id.gt(0).all()
            and abnormal.anomaly_type.isin(ANOMALY_TYPES).all()
            and abnormal.event_phase.isin(["active", "recovery"]).all()):
        raise ValueError("Abnormal rows need a physical event ID, type and phase.")
    for event_id, group in abnormal.groupby("event_id", sort=False):
        # Recovery shares the event ID and therefore must stay in the same set.
        if group.anomaly_type.nunique() != 1 or not np.all(np.diff(group.index) == 1):
            raise ValueError(f"Event {event_id} must be one contiguous block of one type.")


def load_dataset(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, parse_dates=["timestamp"])
    validate_dataset(data)
    return data


def event_aware_split(
    data: pd.DataFrame, train_fraction: float = 0.70,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Choose the closest chronological boundary that cannot divide an event."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")
    # A cut k puts rows [0, k) in training and [k, n) in testing.
    valid = np.ones(len(data) + 1, dtype=bool)
    valid[0] = valid[-1] = False
    for _, group in data.loc[data.event_id.gt(0)].groupby("event_id"):
        positions = data.index.get_indexer(group.index)
        valid[positions.min() + 1:positions.max() + 1] = False
    cuts = np.flatnonzero(valid)
    # Both classes must be available in both sets for a meaningful evaluation.
    for label in (0, 1):
        positions = np.flatnonzero(data.label.eq(label).to_numpy())
        if not len(positions):
            raise ValueError("Both labels must occur in the dataset.")
        cuts = cuts[(cuts > positions.min()) & (cuts <= positions.max())]
    if not len(cuts):
        raise ValueError("No chronological event-preserving split contains both classes.")

    both_types = np.ones(len(cuts), dtype=bool)
    test_types = np.ones(len(cuts), dtype=bool)
    for kind in data.loc[data.label.eq(1), "anomaly_type"].unique():
        positions = np.flatnonzero(data.anomaly_type.eq(kind).to_numpy())
        both_types &= (cuts > positions.min()) & (cuts <= positions.max())
        test_types &= cuts <= positions.max()
    # Prefer all observed types in both sets; otherwise prefer test coverage.
    # This choice uses event metadata only, never features or model performance.
    if both_types.any():
        cuts = cuts[both_types]
    elif test_types.any():
        cuts = cuts[test_types]
    target = int(np.floor(len(data) * train_fraction))
    boundary = int(cuts[np.argmin(np.abs(cuts - target))])
    train, test = data.iloc[:boundary].copy(), data.iloc[boundary:].copy()
    train_events = set(train.loc[train.event_id.gt(0), "event_id"])
    test_events = set(test.loc[test.event_id.gt(0), "event_id"])
    if train_events & test_events or train.timestamp.max() >= test.timestamp.min():
        raise RuntimeError("Split integrity check failed.")
    return train, test


def fit_forest(train: pd.DataFrame, features: tuple[str, ...] = FEATURES) -> RandomForestClassifier:
    """Fit only named, allowed inputs and the training labels."""
    allowed = set(FEATURES + COMMAND_FEATURES)
    if set(features) & FORBIDDEN_FEATURES or not set(features) <= allowed:
        raise ValueError("Features must use the explicit process/actuator allowlist.")
    if len(set(features)) != len(features) or not features:
        raise ValueError("Feature list must be nonempty and contain no duplicates.")
    model = RandomForestClassifier(**MODEL_PARAMETERS)
    model.fit(train[list(features)], train.label)
    return model


def ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def binary_metrics(truth, predictions) -> dict[str, float | int]:
    """Class 1 is positive; rows of the confusion matrix are true classes."""
    tn, fp, fn, tp = (int(value) for value in confusion_matrix(truth, predictions, labels=[0, 1]).ravel())
    return {
        "accuracy": ratio(tp + tn, tp + tn + fp + fn),
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
        "false_positive_rate": ratio(fp, fp + tn),
        "false_negative_rate": ratio(fn, fn + tp),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def positive_group_metrics(data: pd.DataFrame, predictions, mask) -> dict:
    """Recall for an abnormal-only subset; no meaningless subset precision."""
    values = np.asarray(predictions)[np.asarray(mask)]
    tp, count = int(np.count_nonzero(values == 1)), len(values)
    return {"positive_rows": count, "tp": tp, "fn": count - tp,
            "recall": ratio(tp, count), "false_negative_rate": ratio(count - tp, count)}


def phase_metrics(data: pd.DataFrame, predictions) -> pd.DataFrame:
    return pd.DataFrame([
        {"event_phase": phase, **positive_group_metrics(data, predictions, data.event_phase.eq(phase))}
        for phase in ("active", "recovery")
    ])


def per_anomaly_metrics(data: pd.DataFrame, predictions) -> pd.DataFrame:
    records = []
    for kind in ANOMALY_TYPES:
        positives = data.anomaly_type.eq(kind)
        subset = data.label.eq(0) | positives
        # Every type is compared with the same normal test rows. Other anomaly
        # types are excluded, never relabelled as normal for this binary model.
        metrics = binary_metrics(data.loc[subset, "label"], np.asarray(predictions)[subset.to_numpy()])
        active = positive_group_metrics(data, predictions, positives & data.event_phase.eq("active"))
        recovery = positive_group_metrics(data, predictions, positives & data.event_phase.eq("recovery"))
        records.append({
            "anomaly_type": kind,
            "test_events": data.loc[positives, "event_id"].nunique(),
            "positive_rows": int(positives.sum()),
            "normal_rows": int(data.label.eq(0).sum()),
            **metrics,
            "active_rows": active["positive_rows"], "active_recall": active["recall"],
            "recovery_rows": recovery["positive_rows"], "recovery_recall": recovery["recall"],
        })
    return pd.DataFrame(records)


def split_summary(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([
        {"split": name, "rows": len(part),
         "normal_rows": int(part.label.eq(0).sum()),
         "abnormal_rows": int(part.label.eq(1).sum()),
         "normal_percent": 100 * part.label.eq(0).mean(),
         "abnormal_percent": 100 * part.label.eq(1).mean(),
         "start": part.timestamp.iloc[0], "end": part.timestamp.iloc[-1]}
        for name, part in (("train", train), ("test", test))
    ])


def save_evaluation_plots(model, test: pd.DataFrame, predictions, output: Path) -> pd.DataFrame:
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        test.label, predictions, labels=[0, 1],
        display_labels=["Normal (0)", "Abnormal (1)"],
        cmap="Blues", colorbar=False, values_format="d", ax=ax,
    )
    ax.set_title("Random Forest: held-out chronological test set")
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    importance = pd.DataFrame({
        "feature": model.feature_names_in_, "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ordered = importance.sort_values("importance")
    ax.barh(ordered.feature, ordered.importance, color="tab:blue")
    ax.set(xlabel="Mean decrease in impurity (training data)",
           title="Random Forest feature importance")
    fig.tight_layout()
    fig.savefig(output / "feature_importance.png", dpi=160)
    plt.close(fig)
    return importance


def run_baseline(input_path: Path, output: Path) -> dict:
    data = load_dataset(input_path)
    train, test = event_aware_split(data)
    model = fit_forest(train)
    prediction = model.predict(test[list(FEATURES)])
    probability = model.predict_proba(test[list(FEATURES)])[:, list(model.classes_).index(1)]
    metrics = binary_metrics(test.label, prediction)
    training_metrics = binary_metrics(train.label, model.predict(train[list(FEATURES)]))
    by_type = per_anomaly_metrics(test, prediction)
    by_phase = phase_metrics(test, prediction)

    # One predeclared diagnostic comparison, not a parameter search or a
    # replacement selected on the test score. Primary inputs remain fixed.
    command_features = FEATURES + COMMAND_FEATURES
    command_model = fit_forest(train, command_features)
    command_prediction = command_model.predict(test[list(command_features)])
    comparison = []
    for name, values in (("primary_actual_states", prediction), ("with_commands_diagnostic", command_prediction)):
        phases = phase_metrics(test, values).set_index("event_phase")
        comparison.append({
            "model": name, **binary_metrics(test.label, values),
            "active_recall": phases.loc["active", "recall"],
            "recovery_recall": phases.loc["recovery", "recall"],
        })
    comparison = pd.DataFrame(comparison)
    mismatch = np.zeros(len(test), dtype=bool)
    for device in ("inlet_valve", "pump1", "pump2"):
        mismatch |= test[f"{device}_command"].ne(test[f"{device}_actual"]).to_numpy()
    mismatch_summary = pd.DataFrame([
        {"event_phase": phase, "rows": int(test.event_phase.eq(phase).sum()),
         "mismatch_rows": int(mismatch[test.event_phase.eq(phase).to_numpy()].sum())}
        for phase in ("normal", "active", "recovery")
    ])

    train_ids = sorted(int(x) for x in train.loc[train.event_id.gt(0), "event_id"].unique())
    test_ids = sorted(int(x) for x in test.loc[test.event_id.gt(0), "event_id"].unique())
    split = split_summary(train, test)
    near_perfect_groups = [
        f"{row.anomaly_type}/{scope}"
        for row in by_type.itertuples()
        for scope, value in (("all", row.recall), ("active", row.active_recall),
                             ("recovery", row.recovery_recall))
        if value >= 0.99
    ]
    audit = {
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "features": list(FEATURES), "diagnostic_additional_features": list(COMMAND_FEATURES),
        "forbidden_feature_overlap": sorted(set(model.feature_names_in_) & FORBIDDEN_FEATURES),
        "train_event_ids": train_ids, "test_event_ids": test_ids,
        "overlapping_event_ids": sorted(set(train_ids) & set(test_ids)),
        "split_boundary_row_zero_based": len(train),
        "train_fraction": len(train) / len(data),
        "train_last_timestamp": str(train.timestamp.iloc[-1]),
        "test_first_timestamp": str(test.timestamp.iloc[0]),
        "missing_train_anomaly_types": sorted(set(ANOMALY_TYPES) - set(train.anomaly_type)),
        "missing_test_anomaly_types": sorted(set(ANOMALY_TYPES) - set(test.anomaly_type)),
        "near_perfect_accuracy_or_recall": bool(metrics["accuracy"] >= 0.99 or metrics["recall"] >= 0.99),
        "near_perfect_recall_groups": near_perfect_groups,
        "model_parameters": model.get_params(), "training_metrics": training_metrics,
        "test_metrics": metrics,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
                     "matplotlib": matplotlib.__version__, "joblib": joblib.__version__},
    }

    lines = [
        "Stage 3 Random Forest baseline; abnormal label 1 is the positive class.",
        "Split: earliest 70% for training and latest 30% for testing; move the boundary",
        "to the nearest valid point if needed, keeping each event AND recovery together.",
        "Prefer all anomaly types in both sets; if impossible, prefer test coverage.",
        "No row shuffling, temporal feature engineering, scaling or test-set tuning.",
        "", split.to_string(index=False),
        "", f"Train event IDs: {train_ids}", f"Test event IDs: {test_ids}",
        "", "Primary features:", *[f"  {name}" for name in FEATURES],
        "", "Model parameters:", json.dumps(model.get_params(), indent=2),
        "", "Held-out metrics:",
        *[f"  {name}: {metrics[name]:.4%}" for name in (
            "accuracy", "precision", "recall", "f1", "false_positive_rate", "false_negative_rate")],
        f"Confusion matrix [[TN, FP], [FN, TP]]: [[{metrics['tn']}, {metrics['fp']}], [{metrics['fn']}, {metrics['tp']}]]",
        "", "Per-anomaly performance (same normal test rows, other faults excluded):",
        by_type[["anomaly_type", "test_events", "positive_rows", "precision", "recall", "f1",
                 "active_recall", "recovery_recall"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "", "Active versus recovery recall:", by_phase.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "", "Command-state diagnostic comparison (same split and fixed parameters):",
        comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "", "Command/actual mismatch in test data:", mismatch_summary.to_string(index=False),
    ]
    sanity = [
        "Sanity audit",
        f"Near-perfect accuracy or recall (>=99%): {audit['near_perfect_accuracy_or_recall']}",
        f"Anomaly groups with recall >=99%: {near_perfect_groups}",
        f"Forbidden inputs in fitted primary model: {audit['forbidden_feature_overlap']}",
        f"Events shared between training and test, including recovery: {audit['overlapping_event_ids']}",
        f"Missing training anomaly types: {audit['missing_train_anomaly_types']}",
        f"Missing test anomaly types: {audit['missing_test_anomaly_types']}",
        "Only X_train and y_train are passed to fit; no preprocessing uses future rows.",
        "Command columns and legacy duplicate states are absent from primary features.",
        "The diagnostic command model does not change the primary model or its settings.",
        "High active-fault recall is plausible: complete pump stoppages, disjoint pressure",
        "bands and sustained 100% Tank 1 levels make these simulated faults distinctive.",
        "Command/actual mismatch further exposes physical actuator failures directly.",
        "Recovery includes 15 healthy confirmation minutes that still have abnormal labels;",
        "instantaneous inputs can overlap normal operation, making recovery harder.",
        "No labels, feature values or model scores are changed to reduce performance.",
        "The split prevents the same event from entering both sets; temporal dependence",
        "and shared simulator rules remain. Only four held-out events occur in this run.",
        "One synthetic week does not establish generalisation to real plants or robustness",
        "to adversarial sensor manipulation. Feature importance is descriptive, not causal.",
        "", "Training-set diagnostics:", json.dumps(training_metrics, indent=2),
    ]

    output.mkdir(parents=True, exist_ok=True)
    result = test[["timestamp", *FEATURES, "label", "anomaly_type", "event_id", "event_phase"]].copy()
    result.insert(0, "source_row", test.index.to_numpy())
    result.rename(columns={"label": "true_label"}, inplace=True)
    result["predicted_label"] = prediction
    result["abnormal_probability"] = probability
    result.to_csv(output / "test_predictions.csv", index=False)
    pd.DataFrame([metrics]).to_csv(output / "overall_metrics.csv", index=False)
    by_type.to_csv(output / "per_anomaly_metrics.csv", index=False)
    by_phase.to_csv(output / "phase_metrics.csv", index=False)
    split.to_csv(output / "split_summary.csv", index=False)
    comparison.to_csv(output / "command_comparison.csv", index=False)
    mismatch_summary.to_csv(output / "command_mismatch_by_phase.csv", index=False)
    importance = save_evaluation_plots(model, test, prediction, output)
    importance.to_csv(output / "feature_importance.csv", index=False)
    joblib.dump(model, output / "random_forest.joblib", compress=3)
    (output / "run_metadata.json").write_text(json.dumps(audit, indent=2) + "\n")
    (output / "summary.txt").write_text("\n".join(lines) + "\n")
    (output / "sanity_checks.txt").write_text("\n".join(sanity) + "\n")
    print("\n".join(lines))
    print("\n" + "\n".join(sanity))
    print(f"\nSaved predictions, metrics, plots and primary model to {output}")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        run_baseline(args.data, args.output_dir)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
