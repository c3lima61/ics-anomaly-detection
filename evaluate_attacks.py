"""Stage 4: deterministic sensor-only evasion experiments on the frozen baseline."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from attack_plotting import save_attack_plots
from train_baseline import ANOMALY_TYPES, FEATURES, PROJECT_DIR, binary_metrics, load_dataset


SENSORS = (
    "tank1_level", "tank2_level", "inlet_flow", "transfer_flow", "outlet_flow", "pressure",
)
ACTUATORS = ("inlet_valve_actual", "pump1_actual", "pump2_actual")
ALPHAS = (0.0, 0.25, 0.50, 0.75, 1.0)
# Per-sensor engineering bounds, fixed from the simulator design, not fitted to
# test data. Flow bounds include the original normal process variation.
BOUNDS = {
    "tank1_level": (0.0, 100.0), "tank2_level": (0.0, 100.0),
    "inlet_flow": (0.0, 28.8), "transfer_flow": (0.0, 20.5),
    "outlet_flow": (0.0, 26.25), "pressure": (0.0, 3.0),
}
FAULT_GROUPS = {
    "transfer_pump_failure": ("transfer_flow", "pressure", "tank1_level", "tank2_level"),
    "inlet_valve_stuck_open": ("tank1_level", "inlet_flow"),
    "outlet_pump_failure": ("tank2_level", "outlet_flow"),
}
DEFAULT_DATA = PROJECT_DIR / "output/stage2/water_treatment_stage2.csv"
DEFAULT_STAGE3 = PROJECT_DIR / "output/stage3"
DEFAULT_OUTPUT = PROJECT_DIR / "output/stage4"


@dataclass(frozen=True)
class Attack:
    experiment: str
    configuration: str
    sensors: tuple[str, ...]
    target_type: str | None = None


@dataclass
class FrozenAssets:
    train: pd.DataFrame
    test: pd.DataFrame
    model: object
    baseline_prediction: np.ndarray
    ranking: tuple[str, ...]
    metadata: dict
    hashes: dict[str, str]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_hashes(data_path: Path, stage3: Path) -> dict[str, str]:
    paths = [data_path, *sorted(path for path in stage3.iterdir() if path.is_file())]
    return {f"data/{path.name}" if path == data_path else f"stage3/{path.name}": sha256(path)
            for path in paths}


def load_frozen_assets(data_path: Path, stage3: Path) -> FrozenAssets:
    """Restore the exact saved split and model; there is no training fallback."""
    hashes = frozen_hashes(data_path, stage3)
    metadata = json.loads((stage3 / "run_metadata.json").read_text())
    if sha256(data_path) != metadata["input_sha256"]:
        raise ValueError("Stage 2 CSV differs from the data used for Stage 3.")
    data = load_dataset(data_path)
    saved = pd.read_csv(stage3 / "test_predictions.csv", parse_dates=["timestamp"])
    cut = int(metadata["split_boundary_row_zero_based"])
    if not np.array_equal(saved.source_row, np.arange(cut, len(data))):
        raise ValueError("Saved Stage 3 test rows do not match the recorded chronological split.")
    train, test = data.iloc[:cut].copy(), data.iloc[saved.source_row.to_numpy()].copy()
    for source, recorded in (("label", "true_label"), ("timestamp", "timestamp"),
                             ("anomaly_type", "anomaly_type"), ("event_id", "event_id"),
                             ("event_phase", "event_phase")):
        if not np.array_equal(test[source], saved[recorded]):
            raise ValueError(f"Saved test metadata disagrees with source column {source}.")
    if not np.allclose(test[list(FEATURES)], saved[list(FEATURES)], atol=1e-12, rtol=0):
        raise ValueError("Saved test inputs disagree with the truthful source data.")
    train_ids = sorted(int(x) for x in train.loc[train.event_id.gt(0), "event_id"].unique())
    test_ids = sorted(int(x) for x in test.loc[test.event_id.gt(0), "event_id"].unique())
    if (train_ids != metadata["train_event_ids"] or test_ids != metadata["test_event_ids"]
            or set(train_ids) & set(test_ids)):
        raise ValueError("Training/test event IDs do not match the saved Stage 3 split.")

    # Joblib uses pickle: load only locally generated or independently trusted models.
    model = joblib.load(stage3 / "random_forest.joblib")
    if tuple(model.feature_names_in_) != FEATURES or list(FEATURES) != metadata["features"]:
        raise ValueError("Frozen model feature schema changed.")
    if model.get_params() != metadata["model_parameters"] or model.n_estimators != 200:
        raise ValueError("Frozen model parameters differ from Stage 3.")
    prediction = model.predict(test[list(FEATURES)])
    if not np.array_equal(prediction, saved.predicted_label):
        raise ValueError("Frozen model does not reproduce Stage 3 baseline predictions.")
    importance = pd.read_csv(stage3 / "feature_importance.csv")
    expected = pd.Series(model.feature_importances_, index=model.feature_names_in_)
    if set(importance.feature) != set(FEATURES) or not np.allclose(
        importance.importance, importance.feature.map(expected), atol=1e-12, rtol=0,
    ):
        raise ValueError("Stage 3 feature importance does not match the saved model.")
    ranked = importance.loc[importance.feature.isin(SENSORS)].sort_values(
        "importance", ascending=False, kind="stable",
    )
    return FrozenAssets(train, test, model, prediction, tuple(ranked.feature), metadata, hashes)


def training_references(train: pd.DataFrame) -> pd.DataFrame:
    """Use ONLY normal training rows; test data is not an argument."""
    normal = train.loc[train.label.eq(0), list(SENSORS)]
    if normal.empty:
        raise ValueError("Normal training rows are required to construct references.")
    records = []
    for sensor in SENSORS:
        lower, upper = BOUNDS[sensor]
        median = float(normal[sensor].median())
        if not lower <= median <= upper:
            raise ValueError(f"Training median for {sensor} is outside the fixed physical bounds.")
        records.append({
            "sensor": sensor, "normal_training_rows": len(normal),
            "normal_median": median, "normal_q05": normal[sensor].quantile(0.05),
            "normal_q95": normal[sensor].quantile(0.95),
            "physical_min": lower, "physical_max": upper,
        })
    return pd.DataFrame(records)


def build_attacks(ranking: tuple[str, ...]) -> list[Attack]:
    if len(ranking) != 6 or set(ranking) != set(SENSORS):
        raise ValueError("Ranking must contain exactly the six permitted sensors.")
    individual = [Attack("A", name, (name,)) for name in SENSORS]
    multi = [Attack("B", f"top_{count}", ranking[:count]) for count in (1, 2, 3, 6)]
    specific = [Attack("C", kind, group, kind) for kind, group in FAULT_GROUPS.items()]
    return individual + multi + specific


def manipulate(
    test: pd.DataFrame, attack: Attack, references: pd.Series, alpha: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Copy inputs and interpolate permitted sensors on targeted abnormal rows."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must lie between 0 and 1.")
    if (not attack.sensors or len(set(attack.sensors)) != len(attack.sensors)
            or not set(attack.sensors) <= set(SENSORS)):
        raise ValueError("Only the six allowlisted continuous sensors can be manipulated.")
    if attack.target_type is not None and attack.target_type not in ANOMALY_TYPES:
        raise ValueError("Unknown target anomaly type.")
    eligible = test.label.eq(1).to_numpy(copy=True)
    if attack.target_type is not None:
        eligible &= test.anomaly_type.eq(attack.target_type).to_numpy()
    adversarial = test.copy(deep=True)
    if alpha != 0:
        for sensor in attack.sensors:
            reference = float(references[sensor])
            lower, upper = BOUNDS[sensor]
            original = test.loc[eligible, sensor].to_numpy()
            # Assign the exact endpoint at alpha=1, avoiding cancellation error.
            values = np.full(len(original), reference) if alpha == 1 else (
                original + alpha * (reference - original)
            )
            adversarial.loc[eligible, sensor] = np.clip(values, lower, upper)
    protected = [name for name in test.columns if name not in SENSORS]
    if (not adversarial[protected].equals(test[protected])
            or not adversarial.loc[~eligible].equals(test.loc[~eligible])):
        raise RuntimeError("An attack changed a protected field or untargeted row.")
    return adversarial, eligible


def scope_masks(test: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "all_abnormal": test.label.eq(1).to_numpy(),
        "active": test.event_phase.eq("active").to_numpy(),
        "recovery": test.event_phase.eq("recovery").to_numpy(),
        **{kind: test.anomaly_type.eq(kind).to_numpy() for kind in ANOMALY_TYPES},
        **{
            f"{kind}/{phase}": (test.anomaly_type.eq(kind) & test.event_phase.eq(phase)).to_numpy()
            for kind in ANOMALY_TYPES for phase in ("active", "recovery")
        },
    }


def score_attack(
    test: pd.DataFrame, baseline: np.ndarray, prediction: np.ndarray,
    eligible: np.ndarray, changed: np.ndarray, attack: Attack, alpha: float,
) -> list[dict]:
    records = []
    normal = test.label.eq(0).to_numpy()
    if not np.array_equal(prediction[normal], baseline[normal]):
        raise RuntimeError("Predictions on untouched normal rows changed.")
    for scope, positives in scope_masks(test).items():
        subset = normal | positives
        metrics = binary_metrics(test.label.to_numpy()[subset], prediction[subset])
        before = binary_metrics(test.label.to_numpy()[subset], baseline[subset])
        targeted_tp = positives & eligible & (baseline == 1)
        evaded = int(np.count_nonzero(targeted_tp & (prediction == 0)))
        denominator = int(targeted_tp.sum())
        records.append({
            "experiment": attack.experiment, "configuration": attack.configuration,
            "target_anomaly_type": attack.target_type or "all", "alpha": alpha,
            "selected_sensors": "|".join(attack.sensors), "sensor_count": len(attack.sensors),
            "scope": scope, "normal_rows": int(normal.sum()),
            "abnormal_rows": int(positives.sum()),
            "attacked_abnormal_rows": int(np.count_nonzero(eligible & positives)),
            "changed_abnormal_rows": int(np.count_nonzero(changed & positives)),
            "baseline_tp": before["tp"], "baseline_fn": before["fn"],
            "baseline_recall": before["recall"], **metrics,
            "recall_change": metrics["recall"] - before["recall"],
            "baseline_tp_attacked": denominator,
            "previously_detected_became_fn": evaded,
            "attack_success_rate": evaded / denominator if denominator else float("nan"),
            "newly_detected_abnormal_rows": int(np.count_nonzero(
                positives & eligible & (baseline == 0) & (prediction == 1)
            )),
        })
    return records


def row_archive(
    test: pd.DataFrame, adversarial: pd.DataFrame, baseline, prediction,
    eligible, changed_matrix, attack: Attack, alpha: float,
) -> pd.DataFrame:
    """Original columns stay truthful; separate *_adv columns show model inputs."""
    rows = test.copy()
    rows.insert(0, "source_row", test.index.to_numpy())
    rows["experiment"] = attack.experiment
    rows["configuration"] = attack.configuration
    rows["target_anomaly_type"] = attack.target_type or "all"
    rows["alpha"] = alpha
    rows["selected_sensors"] = "|".join(attack.sensors)
    rows["attack_eligible"] = eligible.astype(int)
    for sensor in SENSORS:
        rows[f"{sensor}_adv"] = adversarial[sensor].to_numpy()
    rows["changed_sensors"] = [
        "|".join(sensor for sensor, is_changed in zip(SENSORS, flags) if is_changed)
        for flags in changed_matrix
    ]
    rows["baseline_prediction"] = baseline
    rows["adversarial_prediction"] = prediction
    return rows


def actuator_diagnostic(rows: pd.DataFrame) -> pd.DataFrame:
    """At all-six/alpha=1, only the untouched actuator inputs vary by row."""
    records = []
    for states, group in rows.loc[rows.label.eq(1)].groupby(list(ACTUATORS)):
        if group.adversarial_prediction.nunique() != 1:
            raise RuntimeError("Identical median/state inputs produced inconsistent predictions.")
        detected = group.baseline_prediction.eq(1)
        records.append({
            **dict(zip(ACTUATORS, states)), "abnormal_rows": len(group),
            "baseline_tp": int(detected.sum()),
            "evaded_baseline_tp": int((detected & group.adversarial_prediction.eq(0)).sum()),
            "remaining_detected_rows": int(group.adversarial_prediction.eq(1).sum()),
            "adversarial_prediction": int(group.adversarial_prediction.iloc[0]),
        })
    return pd.DataFrame(records)


def choose_example(rows: pd.DataFrame) -> pd.Series:
    target = rows.attack_eligible.eq(1)
    evaded = target & rows.baseline_prediction.eq(1) & rows.adversarial_prediction.eq(0)
    for mask, reason in (
        (evaded & rows.event_phase.eq("active") & rows.anomaly_type.eq("inlet_valve_stuck_open")
         & rows.tank1_level.ge(99.99), "first full-tank active-fault evasion at alpha=1"),
        (evaded & rows.event_phase.eq("active"), "first successful active-fault evasion at alpha=1"),
        (evaded, "first successful evasion at alpha=1"),
        (target & rows.event_phase.eq("active"), "first active row; no successful evasion at alpha=1"),
    ):
        if mask.any():
            example = rows.loc[mask].iloc[0].copy()
            example["example_selection"] = reason
            return example
    raise ValueError("No rows of the targeted fault occur in the test set.")


def interpret(summary: pd.DataFrame, states: pd.DataFrame) -> str:
    individual = summary.loc[summary.experiment.eq("A")]
    strongest = individual.loc[individual.alpha.gt(0)].sort_values(
        ["recall", "alpha", "configuration"], kind="stable",
    ).iloc[0]
    endpoint = summary.loc[summary.alpha.eq(1)]
    multi = endpoint.loc[endpoint.experiment.eq("B")]
    specific = endpoint.loc[endpoint.experiment.eq("C")]
    easiest = specific.sort_values("attack_success_rate", ascending=False, kind="stable").iloc[0]
    nonmonotone = []
    recall_increases = []
    for (experiment, name), group in summary.groupby(["experiment", "configuration"], sort=False):
        if group.sort_values("alpha").attack_success_rate.diff().lt(-1e-12).any():
            nonmonotone.append(f"{experiment}/{name}")
        if group.sort_values("alpha").recall.diff().gt(1e-12).any():
            recall_increases.append(f"{experiment}/{name}")
    best_single_end = endpoint.loc[endpoint.experiment.eq("A"), "recall"].min()
    all_six = multi.loc[multi.configuration.eq("top_6")].iloc[0]
    lines = [
        f"Greatest observed single-sensor recall loss: {strongest.configuration}, alpha={strongest.alpha:.2f};",
        f"  recall {strongest.baseline_recall:.2%} -> {strongest.recall:.2%}, ASR {strongest.attack_success_rate:.2%}.",
        f"At alpha=1, best single-sensor recall is {best_single_end:.2%}; all-six recall is {all_six.recall:.2%}.",
        f"ASR decreases at a stronger step in {len(nonmonotone)} of 13 configurations: {nonmonotone}.",
        "In this grid ASR is nondecreasing for every group, including plateaus."
        if not nonmonotone else "Stronger manipulation did not always increase evasion.",
        f"Recall increases at a stronger step in {len(recall_increases)} configurations: {recall_increases}.",
        "This can happen when previously missed abnormal rows become detected; they are",
        "counted separately and do not cancel successful evasion in the ASR numerator.",
        "Neither monotonicity nor a benefit from adding sensors is guaranteed by a forest.",
        f"At alpha=1, easiest fault-specific target by ASR is {easiest.target_anomaly_type}:",
        f"  ASR {easiest.attack_success_rate:.2%}; recall {easiest.baseline_recall:.2%} -> {easiest.recall:.2%}.",
        "This compares baseline true positives, rather than crediting pre-existing misses.",
        "", "All-six/alpha=1 predictions grouped by unchanged actual actuator states:",
        states.to_string(index=False),
    ]
    remaining = int(states.remaining_detected_rows.sum())
    if remaining:
        lines += [
            f"{remaining} abnormal rows remain detected with every continuous sensor at its median.",
            "Here the six continuous inputs are identical, so differences across these groups",
            "are due to the three truthful actuator inputs. Their values were never modified.",
        ]
    else:
        lines += [
            "No abnormal rows remain detected in the all-six/alpha=1 configuration; truthful",
            "actuator states alone did not prevent this particular median-based evasion.",
        ]
    lines += [
        "", "Interpretation limits:",
        "The primary experiment uses oracle abnormal-row selection, and experiment C also",
        "uses the true fault type to choose its prescribed group; labels never enter model inputs.",
        "One unconditional normal median per sensor mixes different operating modes and",
        "is not necessarily a jointly observed normal state. Reported flows may contradict",
        "truthful actuator states or water balance, although each sensor stays in range.",
        "The physical process and truthful archive remain unchanged; no simulator is rerun.",
        "These are fixed interpolation attacks, not an optimisation of RF settings or thresholds.",
        "Feature importance chooses B groups as a practical heuristic, not a causal vulnerability measure.",
        "One synthetic week and four held-out events limit generalisation of the results.",
    ]
    return "\n".join(lines)


def run_attacks(data_path: Path, stage3: Path, output: Path) -> pd.DataFrame:
    if output.resolve().is_relative_to(stage3.resolve()):
        raise ValueError("Stage 4 output must be outside the frozen Stage 3 directory.")
    assets = load_frozen_assets(data_path, stage3)
    test = assets.test
    references = training_references(assets.train)
    medians = references.set_index("sensor").normal_median
    for sensor, (low, high) in BOUNDS.items():
        if not test[sensor].between(low, high).all():
            raise ValueError(f"Original {sensor} values exceed the fixed sensor bounds.")
    attacks = build_attacks(assets.ranking)
    scores, archives, examples = [], [], []
    state_table = None
    last_experiment = None
    for attack in attacks:
        if attack.experiment != last_experiment:
            print(f"Running experiment {attack.experiment} on the saved test set...", flush=True)
            last_experiment = attack.experiment
        for alpha in ALPHAS:
            adversarial, eligible = manipulate(test, attack, medians, alpha)
            changed_matrix = np.column_stack([
                adversarial[sensor].to_numpy() != test[sensor].to_numpy() for sensor in SENSORS
            ])
            prediction = assets.model.predict(adversarial[list(FEATURES)])
            scores.extend(score_attack(
                test, assets.baseline_prediction, prediction, eligible,
                changed_matrix.any(axis=1), attack, alpha,
            ))
            rows = row_archive(test, adversarial, assets.baseline_prediction, prediction,
                               eligible, changed_matrix, attack, alpha)
            archives.append(rows)
            if attack.experiment == "C" and alpha == 1:
                examples.append(choose_example(rows))
            if attack.experiment == "B" and len(attack.sensors) == 6 and alpha == 1:
                state_table = actuator_diagnostic(rows)

    results = pd.DataFrame(scores)
    primary_scope = (
        results.experiment.ne("C") & results.scope.eq("all_abnormal")
    ) | (results.experiment.eq("C") & results.scope.eq(results.target_anomaly_type))
    summary = results.loc[primary_scope].copy()
    if frozen_hashes(data_path, stage3) != assets.hashes:
        raise RuntimeError("A frozen model/data/Stage 3 file changed during evaluation.")
    output.mkdir(parents=True, exist_ok=True)
    references.to_csv(output / "normal_training_references.csv", index=False)
    results.to_csv(output / "adversarial_results.csv", index=False)
    summary.to_csv(output / "attack_success_summary.csv", index=False)
    for experiment, name in (("A", "per_sensor_attacks.csv"), ("B", "multi_sensor_attacks.csv"),
                             ("C", "anomaly_specific_attacks.csv")):
        summary.loc[summary.experiment.eq(experiment)].to_csv(output / name, index=False)
    truthful = test.copy()
    truthful.insert(0, "source_row", test.index.to_numpy())
    truthful.to_csv(output / "test_set_truthful.csv", index=False)
    pd.concat(archives, ignore_index=True).to_csv(
        output / "adversarial_rows.csv.gz", index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    example_table = pd.DataFrame(examples)
    example_table.to_csv(output / "row_level_examples.csv", index=False)
    values = pd.DataFrame([
        {"configuration": row.configuration, "source_row": row.source_row,
         "sensor": sensor, "truthful": row[sensor], "manipulated": row[f"{sensor}_adv"],
         "changed": row[sensor] != row[f"{sensor}_adv"],
         "baseline_prediction": row.baseline_prediction,
         "adversarial_prediction": row.adversarial_prediction}
        for _, row in example_table.iterrows() for sensor in SENSORS
    ])
    values.to_csv(output / "row_level_example_values.csv", index=False)
    state_table.to_csv(output / "actuator_state_diagnostic.csv", index=False)
    save_attack_plots(results, summary, output)
    baseline = binary_metrics(test.label, assets.baseline_prediction)
    text = "\n".join([
        "Stage 4: frozen-model sensor manipulation", "",
        f"Test rows: {len(test)}; normal: {test.label.eq(0).sum()}; abnormal: {test.label.eq(1).sum()}",
        f"Baseline TP={baseline['tp']}, FN={baseline['fn']}, FP={baseline['fp']}, TN={baseline['tn']}",
        f"Normal training rows used for every reference: {len(assets.train.loc[assets.train.label.eq(0)])}",
        "Normal reference statistics:", references.to_string(index=False),
        "", f"Continuous sensor ranking from Stage 3: {list(assets.ranking)}",
        f"13 configurations x 5 strengths = 65 runs; {len(scope_masks(test))} scopes each = {len(results)} metric rows.",
        "ASR = baseline true positives changed to 0 / targeted baseline true positives.",
        "The same target cohort is used at alpha=0, giving a zero-success control.",
        "Untargeted scopes have undefined ASR (blank/NaN), not a false zero success rate.",
        "Every scope uses the same untouched normal test rows for precision and F1.",
        f"Normal predictions unchanged: FP={baseline['fp']} and TN={baseline['tn']} for every run.",
        "", "Results at alpha=1 (C reports its targeted fault only):",
        summary.loc[summary.alpha.eq(1), [
            "experiment", "configuration", "baseline_recall", "recall", "false_negative_rate",
            "precision", "f1", "baseline_tp_attacked", "previously_detected_became_fn", "attack_success_rate",
        ]].to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "", interpret(summary, state_table),
        "", "Concrete examples (retrospective illustrations, not parameter tuning):",
        example_table[["configuration", "source_row", "timestamp", "event_phase", "alpha",
                       "baseline_prediction", "adversarial_prediction", "changed_sensors"]].to_string(index=False),
        "", values.to_string(index=False, float_format=lambda x: f"{x:.6f}"),
        "", "Frozen model and all Stage 3 file hashes checked unchanged.",
    ])
    (output / "summary.txt").write_text(text + "\n")
    metadata = {
        "frozen_model_sha256": sha256(stage3 / "random_forest.joblib"),
        "frozen_input_hashes_before": assets.hashes,
        "frozen_input_hashes_after": frozen_hashes(data_path, stage3),
        "model_retrained": False, "model_threshold": "unchanged native predict",
        "model_features": list(FEATURES), "sensor_ranking": list(assets.ranking),
        "normal_training_rows": int(assets.train.label.eq(0).sum()),
        "reference_method": "unconditional median of normal Stage 3 training rows only",
        "train_test_boundary": assets.metadata["split_boundary_row_zero_based"],
        "test_event_ids": assets.metadata["test_event_ids"], "test_rows": len(test),
        "alphas": ALPHAS, "bounds": BOUNDS, "configurations": [asdict(a) for a in attacks],
        "aggregate_result_rows": len(results), "row_archive_rows": len(test) * len(attacks) * len(ALPHAS),
        "normal_rows_untouched": True, "protected_columns": [c for c in test if c not in SENSORS],
        "oracle_row_selection": True, "oracle_fault_type_for_C": True,
        "randomness": "none; references, groups, strengths and row order are fixed",
        "software_versions": assets.metadata["versions"],
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("\n" + text)
    print(f"\nSaved all Stage 4 results to {output}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--stage3-dir", type=Path, default=DEFAULT_STAGE3)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        run_attacks(args.data, args.stage3_dir, args.output_dir)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
