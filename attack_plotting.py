"""Plots of the fixed sensor-manipulation experiment grid."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


def line_panel(ax, data: pd.DataFrame, metric: str, ylabel: str, title: str) -> None:
    for name, group in data.groupby("configuration", sort=False):
        ordered = group.sort_values("alpha")
        label = name.replace("_", " ")
        ax.plot(ordered.alpha, ordered[metric], marker="o", label=label)
    ax.set(xlabel="Attack strength (alpha)", ylabel=ylabel, title=title,
           xticks=[0, 0.25, 0.5, 0.75, 1], ylim=(-0.03, 1.03))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)


def save_attack_plots(results: pd.DataFrame, summary: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    line_panel(ax, summary.loc[summary.experiment.eq("A")], "recall", "Anomaly recall",
               "Individual sensors: all abnormal test rows")
    fig.tight_layout()
    fig.savefig(output / "individual_sensor_recall.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    line_panel(ax, summary.loc[summary.experiment.eq("B")], "recall", "Anomaly recall",
               "Top 1, 2, 3 and all 6 sensors by Stage 3 importance")
    fig.tight_layout()
    fig.savefig(output / "multi_sensor_recall.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    for ax, experiment, title in zip(axes, "ABC", ("Individual sensors", "Ranked groups", "Fault-specific groups")):
        line_panel(ax, summary.loc[summary.experiment.eq(experiment)], "attack_success_rate",
                   "Baseline true positives evaded", title)
    fig.tight_layout()
    fig.savefig(output / "attack_success_rate.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    kinds = ("transfer_pump_failure", "inlet_valve_stuck_open", "outlet_pump_failure")
    labels = ("Transfer pump", "Stuck inlet valve", "Outlet pump")
    x = np.arange(3)
    for ax, scope, title in zip(axes, (None, "active", "recovery"), ("All fault + recovery rows", "Active fault rows", "Recovery rows")):
        before, after = [], []
        for kind in kinds:
            wanted_scope = kind if scope is None else f"{kind}/{scope}"
            group = results.loc[results.experiment.eq("C") & results.configuration.eq(kind)
                                & results.scope.eq(wanted_scope)]
            before.append(group.loc[group.alpha.eq(0), "recall"].iloc[0])
            after.append(group.loc[group.alpha.eq(1), "recall"].iloc[0])
        ax.bar(x - 0.18, before, width=0.36, label="Truthful baseline")
        ax.bar(x + 0.18, after, width=0.36, label="Targeted attack, alpha=1")
        ax.set(xticks=x, xticklabels=labels, ylabel="Anomaly recall", title=title,
               ylim=(0, 1.08))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "per_anomaly_targeted_recall.png", dpi=160)
    plt.close(fig)
