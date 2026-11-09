"""Standalone comparison plots for the single Stage 5 consistency layer."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


LABELS = {
    "transfer_pump_failure": "Transfer pump failure",
    "inlet_valve_stuck_open": "Inlet valve stuck open",
    "outlet_pump_failure": "Outlet pump failure",
}
UNDEFENDED_COLOR = "#377eb8"
DEFENDED_COLOR = "#d95f02"


def rates_axis(ax, ylabel: str, title: str) -> None:
    ax.set(xlabel="Attack strength (alpha)", ylabel=ylabel, title=title,
           xticks=[0, 0.25, 0.5, 0.75, 1], ylim=(-0.025, 1.06))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(alpha=0.2)


def comparison_grid(summary: pd.DataFrame, metric: str, title: str, path: Path) -> None:
    fig, axes = plt.subplots(4, 4, figsize=(17, 14), sharex=True, sharey=True)
    for ax, ((experiment, name), group) in zip(
        axes.flat, summary.groupby(["experiment", "configuration"], sort=False),
    ):
        group = group.sort_values("alpha")
        for prefix, label, color, style in (
            ("undefended", "RF under attack", UNDEFENDED_COLOR, "--"),
            ("defended", "RF + consistency", DEFENDED_COLOR, "-"),
        ):
            ax.plot(group.alpha, group[f"{prefix}_{metric}"], marker="o", linestyle=style,
                    color=color, label=label, markersize=4)
        name = LABELS.get(name, name.replace("_", " "))
        rates_axis(ax, "Recall" if metric == "recall" else "Attack success rate", f"{experiment}: {name}")
        ax.tick_params(labelbottom=True, labelleft=True)
    for ax in list(axes.flat)[13:]:
        ax.set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.67, 0.13), frameon=False, fontsize=12)
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_defence_plots(truth: pd.DataFrame, results: pd.DataFrame, summary: pd.DataFrame, output: Path) -> None:
    overall = truth.loc[truth.scope.eq("all_abnormal")].set_index("model")
    baseline = overall.loc["random_forest"]
    defended = overall.loc["random_forest_plus_consistency"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metrics, names, title in (
        (axes[0], ("accuracy", "precision", "recall", "f1"), ("Accuracy", "Precision", "Recall", "F1"), "Truthful test performance"),
        (axes[1], ("false_positive_rate", "false_negative_rate"), ("False-positive rate", "False-negative rate"), "Truthful test error rates"),
    ):
        x = np.arange(len(metrics))
        for offset, values, color, label in (
            (-0.18, baseline, UNDEFENDED_COLOR, "Original RF"),
            (0.18, defended, DEFENDED_COLOR, "RF + consistency"),
        ):
            bars = ax.bar(x + offset, [values[m] for m in metrics], width=0.36, color=color, label=label)
            ax.bar_label(bars, labels=[f"{values[m]:.1%}" for m in metrics], fontsize=8, padding=3)
        ax.set(xticks=x, xticklabels=names, title=title)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_ylim(0, 1.12 if len(metrics) == 4 else max(baseline.false_negative_rate, defended.false_negative_rate,
                                                      baseline.false_positive_rate, defended.false_positive_rate) * 1.35)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle(f"New normal rows flagged: {int(defended.newly_flagged_normal_rows)} / {int(defended.normal_rows)}")
    fig.tight_layout()
    fig.savefig(output / "truthful_defence_performance.png", dpi=160)
    plt.close(fig)

    comparison_grid(summary, "recall", "Unchanged Stage 4 attacks: recall before and after consistency validation",
                    output / "recall_vs_attack_strength.png")
    comparison_grid(summary, "attack_success_rate", "Evasion of original RF true positives before and after validation",
                    output / "attack_success_vs_strength.png")

    six = summary.loc[summary.experiment.eq("B") & summary.configuration.eq("top_6")].sort_values("alpha")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(six.alpha, six.undefended_recall, "o--", color=UNDEFENDED_COLOR, label="RF under attack")
    ax.plot(six.alpha, six.defended_recall, "o-", color=DEFENDED_COLOR, label="RF + consistency under attack")
    ax.axhline(baseline.recall, color="gray", linestyle=":", label=f"Truthful RF recall ({baseline.recall:.2%})")
    rates_axis(ax, "Anomaly recall", "All six continuous sensors: unchanged interpolation attack")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(output / "all_six_sensor_recall.png", dpi=160)
    plt.close(fig)

    kinds = tuple(LABELS)
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), sharey=True)
    x = np.arange(3)
    for ax, phase, title in zip(axes, (None, "active", "recovery"),
                               ("Active + recovery", "Active faults", "Recovery")):
        before, attacked, protected = [], [], []
        for kind in kinds:
            scope = kind if phase is None else f"{kind}/{phase}"
            row = results.loc[results.experiment.eq("C") & results.configuration.eq(kind)
                              & results.alpha.eq(1) & results.scope.eq(scope)].iloc[0]
            before.append(row.baseline_recall)
            attacked.append(row.undefended_recall)
            protected.append(row.defended_recall)
        ax.bar(x - 0.25, before, width=0.25, color="#999999", label="Truthful RF")
        ax.bar(x, attacked, width=0.25, color=UNDEFENDED_COLOR, label="Attacked RF")
        ax.bar(x + 0.25, protected, width=0.25, color=DEFENDED_COLOR, label="Attacked RF + consistency")
        ax.set(xticks=x, xticklabels=["Transfer pump", "Stuck inlet valve", "Outlet pump"],
               ylabel="Anomaly recall", title=title, ylim=(0, 1.06))
        ax.tick_params(axis="x", rotation=12)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="y", alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=3, frameon=False)
    fig.suptitle("Fault-specific attacks at alpha = 1: effect of the same consistency layer")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(output / "fault_specific_defence.png", dpi=160)
    plt.close(fig)
