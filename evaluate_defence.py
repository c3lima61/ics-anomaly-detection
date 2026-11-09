"""Stage 5: evaluate one physical-consistency layer on the frozen Stage 4 grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from defence_plotting import save_defence_plots
from evaluate_attacks import (
    ALPHAS, DEFAULT_DATA, DEFAULT_STAGE3, SENSORS, build_attacks,
    frozen_hashes, load_frozen_assets, scope_masks, sha256,
)
from physical_consistency import (
    BALANCE_CHECKS, QUANTILE, RULE_COLUMNS, RULES, STATE_CHECKS,
    apply_consistency, calibrate_thresholds, hybrid_prediction,
)
from train_baseline import FEATURES, PROJECT_DIR, binary_metrics


DEFAULT_STAGE4 = PROJECT_DIR / "output/stage4"
DEFAULT_OUTPUT = PROJECT_DIR / "output/stage5"


def stage4_hashes(stage4: Path) -> dict[str, str]:
    return {f"stage4/{path.name}": sha256(path) for path in sorted(stage4.iterdir()) if path.is_file()}


def normal_training_inputs(train: pd.DataFrame) -> pd.DataFrame:
    # Labels are used ONLY here to select normal TRAINING records. The rule
    # engine receives no labels, event information, commands or test data.
    return train.loc[train.label.eq(0), list(FEATURES)].copy()


def reported_inputs(rows: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct the exact nine RF/defence inputs stored in the Stage 4 CSV."""
    inputs = pd.DataFrame({
        name: rows[f"{name}_adv"].to_numpy() if name in SENSORS else rows[name].to_numpy()
        for name in FEATURES
    }, index=rows.source_row.to_numpy())
    return inputs


def load_attack_archive(assets, stage4: Path) -> pd.DataFrame:
    metadata = json.loads((stage4 / "run_metadata.json").read_text())
    model_hash = next(digest for path, digest in assets.hashes.items()
                      if Path(path).name == "random_forest.joblib")
    if (metadata["frozen_input_hashes_after"] != assets.hashes
            or metadata["frozen_model_sha256"] != model_hash):
        raise ValueError("Stage 4 used different frozen Stage 3 assets.")
    if (metadata["model_features"] != list(FEATURES)
            or metadata["alphas"] != list(ALPHAS)
            or metadata["sensor_ranking"] != list(assets.ranking)):
        raise ValueError("Stage 4 feature schema, grid or sensor ranking changed.")
    # Round-trip parsing preserves the exact float64 values exported by Stage 4
    # instead of introducing extra CSV-decoding rounding into attack values.
    rows = pd.read_csv(stage4 / "adversarial_rows.csv.gz", parse_dates=["timestamp"],
                       float_precision="round_trip")
    attacks = build_attacks(assets.ranking)
    expected_rows = len(assets.test) * len(attacks) * len(ALPHAS)
    if len(rows) != expected_rows or not rows.source_row.isin(assets.test.index).all():
        raise ValueError("Stage 4 archive does not contain the complete saved test grid.")
    original = assets.test.loc[rows.source_row].reset_index(drop=True)
    for name in assets.test:
        if pd.api.types.is_float_dtype(original[name]):
            if not np.allclose(rows[name], original[name], atol=1e-12, rtol=0):
                raise ValueError(f"Truthful archive values changed: {name}.")
        elif not np.array_equal(rows[name], original[name]):
            raise ValueError(f"Protected archive values changed: {name}.")
    groups = rows.groupby(["experiment", "configuration", "alpha"], sort=False)
    expected_keys = {(a.experiment, a.configuration, alpha) for a in attacks for alpha in ALPHAS}
    if set(groups.groups) != expected_keys:
        raise ValueError("Stage 4 archive configurations differ from the prescribed grid.")
    for attack in attacks:
        for alpha in ALPHAS:
            group = groups.get_group((attack.experiment, attack.configuration, alpha))
            if (not np.array_equal(group.source_row, assets.test.index)
                    or not group.selected_sensors.eq("|".join(attack.sensors)).all()
                    or not group.target_anomaly_type.eq(attack.target_type or "all").all()):
                raise ValueError("Stage 4 test ordering or attack groups changed.")
            eligible = assets.test.label.eq(1)
            if attack.target_type is not None:
                eligible &= assets.test.anomaly_type.eq(attack.target_type)
            if not np.array_equal(group.attack_eligible, eligible.astype(int)):
                raise ValueError("Stage 4 attack eligibility changed.")
            untouched = ~eligible.to_numpy()
            for sensor in SENSORS:
                if not np.allclose(group.loc[untouched, f"{sensor}_adv"],
                                   group.loc[untouched, sensor], atol=1e-12, rtol=0):
                    raise ValueError("Stage 4 modified an untargeted row.")
    baseline = assets.baseline_prediction[rows.source_row.to_numpy() - assets.test.index[0]]
    if not np.array_equal(rows.baseline_prediction, baseline):
        raise ValueError("Archived truthful RF predictions disagree with Stage 3.")
    # Verify saved predictions once using native predict; never train or change
    # the forest, its probability threshold, or any archived attack value.
    full_inputs = pd.DataFrame({
        name: rows[f"{name}_adv"] if name in SENSORS else rows[name] for name in FEATURES
    })
    if not np.array_equal(assets.model.predict(full_inputs), rows.adversarial_prediction):
        raise ValueError("Frozen RF predictions disagree with Stage 4 adversarial predictions.")
    return rows


def truthful_metrics(test, baseline, defended, diagnostics) -> pd.DataFrame:
    normal = test.label.eq(0).to_numpy()
    records = []
    newly_flagged = int(np.count_nonzero(normal & (baseline == 0) & (defended == 1)))
    for scope, positives in scope_masks(test).items():
        subset = normal | positives
        for model, prediction in (("random_forest", baseline),
                                  ("random_forest_plus_consistency", defended)):
            records.append({
                "model": model, "scope": scope, "normal_rows": int(normal.sum()),
                "abnormal_rows": int(positives.sum()),
                **binary_metrics(test.label.to_numpy()[subset], prediction[subset]),
                "newly_flagged_normal_rows": newly_flagged if model != "random_forest" else 0,
                "consistency_alert_normal_rows": int(diagnostics.consistency_alert.to_numpy()[normal].sum()),
                "consistency_alert_abnormal_rows": int(diagnostics.consistency_alert.to_numpy()[positives].sum()),
            })
    return pd.DataFrame(records)


def comparison_records(test, baseline, forest, defended, eligible, diagnostics, details) -> list[dict]:
    normal = test.label.eq(0).to_numpy()
    alerts = diagnostics.consistency_alert.to_numpy(dtype=bool)
    records = []
    for scope, positives in scope_masks(test).items():
        subset = normal | positives
        targeted_tp = positives & eligible & (baseline == 1)
        denominator = int(targeted_tp.sum())
        forest_misses = targeted_tp & (forest == 0)
        defended_misses = targeted_tp & (defended == 0)
        records.append({
            **details, "scope": scope, "normal_rows": int(normal.sum()),
            "abnormal_rows": int(positives.sum()),
            "attacked_abnormal_rows": int(np.count_nonzero(positives & eligible)),
            "baseline_tp_attacked": denominator,
            "baseline_recall": binary_metrics(test.label.to_numpy()[subset], baseline[subset])["recall"],
            **{f"undefended_{k}": v for k, v in binary_metrics(
                test.label.to_numpy()[subset], forest[subset]).items()},
            **{f"defended_{k}": v for k, v in binary_metrics(
                test.label.to_numpy()[subset], defended[subset]).items()},
            "undefended_success_count": int(forest_misses.sum()),
            "defended_success_count": int(defended_misses.sum()),
            "undefended_attack_success_rate": float(forest_misses.sum() / denominator) if denominator else np.nan,
            "defended_attack_success_rate": float(defended_misses.sum() / denominator) if denominator else np.nan,
            "baseline_tp_saved_by_defence": int(np.count_nonzero(forest_misses & alerts)),
            "rf_false_negatives_caught": int(np.count_nonzero(positives & (forest == 0) & alerts)),
            "consistency_alert_abnormal_rows": int(np.count_nonzero(positives & alerts)),
            "consistency_alert_normal_rows": int(np.count_nonzero(normal & alerts)),
            "newly_flagged_normal_rows": int(np.count_nonzero(normal & (forest == 0) & alerts)),
        })
    return records


def rule_records(test, baseline, forest, eligible, diagnostics, details, truthful_diagnostics=None) -> list[dict]:
    """Counts overlap intentionally; exclusive and multiple-rule catches are explicit."""
    normal = test.label.eq(0).to_numpy()
    counts = diagnostics.consistency_rule_count.to_numpy()
    flags = {rule: diagnostics[f"rule_{rule}"].to_numpy(dtype=bool) for rule in RULES}
    flags.update(any_rule=counts > 0, multiple_rules=counts > 1)
    reference = None
    if truthful_diagnostics is not None:
        reference = {rule: truthful_diagnostics[f"rule_{rule}"].to_numpy(dtype=bool) for rule in RULES}
        original_counts = truthful_diagnostics.consistency_rule_count.to_numpy()
        reference.update(any_rule=original_counts > 0, multiple_rules=original_counts > 1)
    records = []
    for scope, positives in scope_masks(test).items():
        targeted = positives & eligible
        for rule, triggered in flags.items():
            records.append({
                **details, "scope": scope, "rule": rule,
                "rule_family": "water_balance" if rule in BALANCE_CHECKS else (
                    "actuator_flow_pressure" if rule in STATE_CHECKS else "combined"),
                "triggered_normal_rows": int(np.count_nonzero(normal & triggered)),
                "triggered_abnormal_rows": int(np.count_nonzero(positives & triggered)),
                "targeted_abnormal_rows_caught": int(np.count_nonzero(targeted & triggered)),
                "rf_false_negatives_caught": int(np.count_nonzero(positives & (forest == 0) & triggered)),
                "targeted_baseline_tp_saved": int(np.count_nonzero(
                    targeted & (baseline == 1) & (forest == 0) & triggered)),
                "exclusive_rf_fn_catches": int(np.count_nonzero(
                    positives & (forest == 0) & triggered & (counts == 1))),
                "new_rule_alerts_on_targeted_abnormal": int(np.count_nonzero(
                    targeted & triggered & ~reference[rule])) if reference is not None else np.nan,
                "new_rule_alerts_on_normal": int(np.count_nonzero(
                    normal & triggered & ~reference[rule])) if reference is not None else np.nan,
                "baseline_tp_saved_by_new_rule_alert": int(np.count_nonzero(
                    targeted & (baseline == 1) & (forest == 0) & triggered & ~reference[rule]
                )) if reference is not None else np.nan,
            })
    return records


def primary_rows(results: pd.DataFrame) -> pd.DataFrame:
    selected = (results.experiment.ne("C") & results.scope.eq("all_abnormal")) | (
        results.experiment.eq("C") & results.scope.eq(results.target_anomaly_type))
    return results.loc[selected].copy()


def leading_rules(summary: pd.DataFrame, rules: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in summary.iterrows():
        group = rules.loc[
            rules.experiment.eq(row.experiment) & rules.configuration.eq(row.configuration)
            & rules.alpha.eq(row.alpha) & rules.scope.eq(row.scope)
        ]
        individuals = group.loc[group.rule.isin(RULES)]
        maximum = int(individuals.targeted_baseline_tp_saved.max())
        winners = individuals.loc[individuals.targeted_baseline_tp_saved.eq(maximum), "rule"]
        records.append({
            "experiment": row.experiment, "configuration": row.configuration,
            "alpha": row.alpha, "scope": row.scope,
            "most_useful_rules": "|".join(winners) if maximum else "none",
            "baseline_tp_saved_by_leading_rule": maximum,
            "baseline_tp_saved_by_any_rule": row.baseline_tp_saved_by_defence,
            "targeted_abnormal_caught_by_multiple_rules": int(group.loc[
                group.rule.eq("multiple_rules"), "targeted_abnormal_rows_caught"].iloc[0]),
        })
    return pd.DataFrame(records)


def example_rule_values(examples: pd.DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in examples.iterrows():
        for rule in RULES:
            values = thresholds.loc[thresholds.rule.eq(rule)]
            if rule in STATE_CHECKS:
                actuator = STATE_CHECKS[rule][1]
                values = values.loc[values.actuator_state.eq(row[actuator])]
            bound = values.iloc[0]
            records.append({
                "experiment": row.experiment, "configuration": row.configuration,
                "source_row": row.source_row, "rule": rule,
                "residual": row[f"{rule}_residual"], "threshold": bound.threshold,
                "triggered": row[f"rule_{rule}"], "units": bound.units,
            })
    return pd.DataFrame(records)


def make_report(truth, summary, leaders, thresholds, examples) -> str:
    overall = truth.loc[truth.scope.eq("all_abnormal")].set_index("model")
    before, after = overall.loc["random_forest"], overall.loc["random_forest_plus_consistency"]
    endpoints = summary.loc[summary.alpha.eq(1)]
    all_six = endpoints.loc[endpoints.experiment.eq("B") & endpoints.configuration.eq("top_6")].iloc[0]
    still_effective = endpoints.loc[endpoints.defended_success_count.gt(0), "configuration"].tolist()
    training_rows = int(thresholds.loc[thresholds.rule.eq("inlet_flow_state"), "training_samples"].sum())
    training_pairs = int(thresholds.loc[thresholds.rule.eq("tank1_balance"), "training_samples"].iloc[0])
    lines = [
        "Stage 5: frozen Random Forest + one physical-consistency layer", "",
        f"Calibration uses {training_rows:,} normal Stage 3 training rows, including {training_pairs:,} consecutive normal pairs.",
        "No test data or test labels determine thresholds. Quantile is fixed at 0.99 before evaluation.",
        "State residual = reported value - normal-training median conditional on actual state.",
        "Balance residual = level[t] - level[t-1] - 0.1 * (inflow[t] - outflow[t]).",
        "Effective tolerance = max(99th percentile of absolute residual, eight-ULP numerical floor).",
        "The tiny balance tolerances reflect exact conservation in this simplified simulator, not realistic sensor accuracy.",
        "", "Every threshold (full precision also saved in defence_thresholds.csv):",
        thresholds.to_string(index=False, float_format=lambda x: f"{x:.12g}"),
        "", "Truthful held-out performance (positive = abnormal):",
        overall[["accuracy", "precision", "recall", "f1", "false_positive_rate", "false_negative_rate",
                 "tn", "fp", "fn", "tp", "newly_flagged_normal_rows"]].to_string(float_format=lambda x: f"{x:.6f}"),
        "Confusion matrices use [[TN, FP], [FN, TP]].",
        f"RF: [[{int(before.tn)}, {int(before.fp)}], [{int(before.fn)}, {int(before.tp)}]]",
        f"Hybrid: [[{int(after.tn)}, {int(after.fp)}], [{int(after.fn)}, {int(after.tp)}]]",
        f"New normal rows flagged by the layer: {int(after.newly_flagged_normal_rows)} / {int(after.normal_rows)}.",
        "Per-rule 99% intervals do not imply a 1% combined false-positive rate; the rules are OR-combined.",
        "", "Attack endpoints at alpha=1 (C evaluates only its targeted type):",
        endpoints[["experiment", "configuration", "undefended_recall", "defended_recall",
                   "undefended_false_negative_rate", "defended_false_negative_rate",
                   "undefended_attack_success_rate", "defended_attack_success_rate",
                   "defended_false_positive_rate", "baseline_tp_saved_by_defence"]].to_string(
                       index=False, float_format=lambda x: f"{x:.6f}"),
        "", f"All-six recall restored from {all_six.undefended_recall:.2%} to {all_six.defended_recall:.2%};",
        f"ASR changes from {all_six.undefended_attack_success_rate:.2%} to {all_six.defended_attack_success_rate:.2%}.",
        f"Configurations with remaining evasion at alpha=1: {still_effective}",
        "", "Most useful rules by baseline true positives saved at alpha=1 (ties shown):",
        leaders.loc[leaders.alpha.eq(1)].to_string(index=False),
        "Rule counts overlap; multiple-rule and exclusive catches are saved separately.",
        "Some state-rule alerts already occur on truthful physical faults. Rule summaries also",
        "separate new alerts introduced by manipulation from alerts present on truthful inputs.",
        "", "ASR uses the SAME original Stage 3 RF true positives and attacked cohort as Stage 4.",
        "Defended numerator: targeted original RF true positives still predicted 0 by the hybrid.",
        "Pre-existing RF misses are excluded from both ASR denominators, but included in recall.",
        "", "Causal evaluation and limits:",
        "The rule engine sees only nine sensor/actual-state inputs, not labels or event metadata.",
        "Every normal and abnormal row is checked. Labels/types are used only for reporting",
        "and Stage 4's already-fixed attack eligibility, never to decide consistency alerts.",
        "Each of the 65 runs starts with no previous test sample; balance is unavailable only",
        "for its first sample. Subsequent checks use the previous REPORTED level, never hidden truth.",
        "An attack ending can also cause a normal row to alert because its previous reading was manipulated.",
        "Stage 4 values, model, native prediction threshold and all earlier artifacts stay unchanged.",
        "Exact synthetic water balance makes small inconsistent changes unusually easy to detect.",
        "Real measurement noise, uncertain hydraulic parameters, lag and missing samples require new",
        "normal-data calibration; the numerical floor is not a real-world sensor noise model.",
        "Coordinated plausible manipulations or constant level offsets during balanced/idle periods",
        "can satisfy these checks. Truthful abnormal operation can itself violate normal state ranges.",
        "This rule layer validates consistency, not sensor authenticity, and is not a complete defence.",
        "One synthetic week and four held-out events limit generalisation. No further model or defence is added.",
        "", "Concrete rescued examples (RF fooled, hybrid catches):",
        examples[["experiment", "configuration", "source_row", "timestamp", "event_phase",
                  "adversarial_prediction", "consistency_alert", "defended_prediction", "triggered_rules"]].to_string(index=False),
        "", "All source/model/Stage 3/Stage 4 hashes checked unchanged.",
    ]
    return "\n".join(lines)


def run_defence(data_path: Path, stage3: Path, stage4: Path, output: Path) -> pd.DataFrame:
    if any(output.resolve().is_relative_to(path.resolve()) for path in (stage3, stage4)):
        raise ValueError("Stage 5 output must be outside the frozen Stage 3 and Stage 4 directories.")
    assets = load_frozen_assets(data_path, stage3)
    attack_hashes = stage4_hashes(stage4)
    normal_inputs = normal_training_inputs(assets.train)
    # Freeze this table before inspecting any held-out consistency outcomes.
    thresholds = calibrate_thresholds(normal_inputs)
    frozen_thresholds = thresholds.copy(deep=True)
    test = assets.test
    diagnostics = apply_consistency(test.loc[:, list(FEATURES)], thresholds)
    truthful_defended = hybrid_prediction(assets.baseline_prediction, diagnostics.consistency_alert)
    truth = truthful_metrics(test, assets.baseline_prediction, truthful_defended, diagnostics)
    print("Truthful test evaluated with frozen normal-training thresholds.", flush=True)
    archive = load_attack_archive(assets, stage4)
    print("Stage 4 archive verified; evaluating its unchanged 65 runs...", flush=True)
    scores, defended_rows, rule_scores, examples = [], [], [], []
    truth_details = {"experiment": "truthful", "configuration": "untouched", "alpha": 0.0,
                     "target_anomaly_type": "all", "selected_sensors": "", "sensor_count": 0}
    rule_scores.extend(rule_records(test, assets.baseline_prediction, assets.baseline_prediction,
                                   test.label.eq(1).to_numpy(), diagnostics, truth_details, diagnostics))
    saved_examples = pd.read_csv(stage4 / "row_level_examples.csv").set_index("configuration").source_row
    for (experiment, configuration, alpha), rows in archive.groupby(
        ["experiment", "configuration", "alpha"], sort=False,
    ):
        inputs = reported_inputs(rows)
        checks = apply_consistency(inputs, thresholds)
        forest = rows.adversarial_prediction.to_numpy()
        defended = hybrid_prediction(forest, checks.consistency_alert)
        eligible = rows.attack_eligible.to_numpy(dtype=bool)
        details = {"experiment": experiment, "configuration": configuration, "alpha": alpha,
                   "target_anomaly_type": rows.target_anomaly_type.iloc[0],
                   "selected_sensors": rows.selected_sensors.iloc[0],
                   "sensor_count": len(rows.selected_sensors.iloc[0].split("|"))}
        scores.extend(comparison_records(test, assets.baseline_prediction, forest, defended,
                                         eligible, checks, details))
        rule_scores.extend(rule_records(test, assets.baseline_prediction, forest, eligible, checks, details, diagnostics))
        if alpha == 0:
            if (not np.array_equal(checks[list(RULE_COLUMNS)], diagnostics[list(RULE_COLUMNS)])
                    or not np.array_equal(defended, truthful_defended)):
                raise ValueError("An alpha-0 control differs from truthful defence evaluation.")
        record = pd.concat([rows.reset_index(drop=True), checks.reset_index(drop=True)], axis=1)
        record["defended_prediction"] = defended
        defended_rows.append(record)
        if alpha == 1 and (experiment == "C" or experiment == "B" and configuration == "top_6"):
            rescued = record.loc[
                record.attack_eligible.eq(1) & record.baseline_prediction.eq(1)
                & record.adversarial_prediction.eq(0) & record.defended_prediction.eq(1)
                & record.event_phase.eq("active")
            ]
            if len(rescued):
                preferred = rescued.loc[rescued.source_row.eq(saved_examples.get(configuration, -1))]
                examples.append((preferred if len(preferred) else rescued).iloc[0])
    results = pd.DataFrame(scores)
    summary = primary_rows(results)
    rules = pd.DataFrame(rule_scores)
    leaders = leading_rules(summary, rules)
    example_table = pd.DataFrame(examples)
    if example_table.empty:
        raise ValueError("No concrete rescued examples found; inspect the evaluation before exporting.")
    # A direct comparison to Stage 4 prevents accidental changes to attacks,
    # test ordering, native RF predictions or ASR denominators.
    keys = ["experiment", "configuration", "alpha", "scope"]
    previous = pd.read_csv(stage4 / "adversarial_results.csv").set_index(keys).sort_index()
    current = results.set_index(keys).sort_index()
    if not current.index.equals(previous.index):
        raise ValueError("Stage 5 evaluation scopes differ from Stage 4.")
    for metric in ("accuracy", "precision", "recall", "f1", "false_positive_rate", "false_negative_rate",
                   "tn", "fp", "fn", "tp", "attack_success_rate"):
        if not np.allclose(current[f"undefended_{metric}"], previous[metric], atol=1e-12, rtol=0, equal_nan=True):
            raise ValueError(f"Undefended {metric} differs from Stage 4.")
    pd.testing.assert_frame_equal(thresholds, frozen_thresholds, check_exact=True)
    if frozen_hashes(data_path, stage3) != assets.hashes or stage4_hashes(stage4) != attack_hashes:
        raise RuntimeError("A frozen model, source data or earlier-stage artifact changed.")

    output.mkdir(parents=True, exist_ok=True)
    thresholds.to_csv(output / "defence_thresholds.csv", index=False)
    truth.to_csv(output / "truthful_defence_metrics.csv", index=False)
    results.to_csv(output / "defended_attack_results.csv", index=False)
    summary.to_csv(output / "defended_attack_summary.csv", index=False)
    rules.to_csv(output / "rule_trigger_summary.csv", index=False)
    leaders.to_csv(output / "rule_effectiveness.csv", index=False)
    truthful_rows = pd.concat([test, diagnostics], axis=1)
    truthful_rows.insert(0, "source_row", test.index.to_numpy())
    truthful_rows["baseline_prediction"] = assets.baseline_prediction
    truthful_rows["defended_prediction"] = truthful_defended
    truthful_rows.to_csv(output / "truthful_defended_predictions.csv", index=False)
    pd.concat(defended_rows, ignore_index=True).to_csv(
        output / "defended_rows.csv.gz", index=False, compression={"method": "gzip", "mtime": 0},
    )
    example_table.to_csv(output / "defence_examples.csv", index=False)
    example_rule_values(example_table, thresholds).to_csv(output / "defence_example_rules.csv", index=False)
    save_defence_plots(truth, results, summary, output)
    report = make_report(truth, summary, leaders, thresholds, example_table)
    (output / "summary.txt").write_text(report + "\n")
    metadata = {
        "model_retrained": False, "model_threshold": "unchanged native predict",
        "frozen_model_sha256": sha256(stage3 / "random_forest.joblib"),
        "frozen_input_hashes_before": assets.hashes,
        "frozen_input_hashes_after": frozen_hashes(data_path, stage3),
        "stage4_hashes_before": attack_hashes, "stage4_hashes_after": stage4_hashes(stage4),
        "defence_thresholds_sha256": sha256(output / "defence_thresholds.csv"),
        "calibration": "normal Stage 3 training rows only; no test-driven threshold tuning",
        "normal_training_rows": len(normal_inputs),
        "normal_training_pairs": int(normal_inputs.index.to_series().diff().eq(1).sum()),
        "quantile": QUANTILE, "numerical_tolerance": "8 float64 ULPs at the feature scale; 100% for tank levels",
        "quantile_interpolation": "linear (pandas default)",
        "stage4_float_parsing": "round_trip; preserve exported float64 attack values",
        "features_available_to_defence": list(FEATURES),
        "forbidden_defence_inputs": ["label", "anomaly_type", "event_id", "event_phase", "timestamp"],
        "first_test_sample": "state checks only; balance needs a preceding reported sample",
        "history": "previous reported level within the same run; reset at each run and sample gap",
        "normal_test_rows_checked": True, "rule_names": list(RULES), "alphas": ALPHAS,
        "attack_runs": len(summary), "scope_result_rows": len(results), "row_archive_rows": len(archive),
        "asr_denominator": "original Stage 3 RF true positives targeted by the unchanged Stage 4 attack",
        "asr_numerator_defended": "those same rows still predicted normal by RF OR consistency alert",
        "randomness": "none; frozen thresholds and deterministic arithmetic",
        "software_versions": assets.metadata["versions"],
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if frozen_hashes(data_path, stage3) != assets.hashes or stage4_hashes(stage4) != attack_hashes:
        raise RuntimeError("An earlier-stage artifact changed while exporting Stage 5.")
    print("\n" + report)
    print(f"\nSaved Stage 5 results to {output}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--stage3-dir", type=Path, default=DEFAULT_STAGE3)
    parser.add_argument("--stage4-dir", type=Path, default=DEFAULT_STAGE4)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        run_defence(args.data, args.stage3_dir, args.stage4_dir, args.output_dir)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
