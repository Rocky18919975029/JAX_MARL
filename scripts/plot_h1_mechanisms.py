#!/usr/bin/env python3
"""Plot seed trajectories, seed summaries, paired effects, and decision validity."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {"none": "#e63946", "a_to_c": "#80b918", "c_to_a": "#377bd1"}
CONDITIONS = ("none", "a_to_c", "c_to_a")
PANELS = (
    ("heldout_return", "Held-out stochastic return", "Episode return"),
    ("epsilon_lat", r"Latent update distortion $\epsilon_{Lat}$", "Lower is better"),
    ("epsilon_dec", r"Actor decision error $\epsilon_{Dec}$", "Lower is better"),
    ("epsilon_bell", r"Critic Bellman error $\epsilon_{Bell}$", "Lower is better"),
)
DECISION_VALIDITY_PANELS = (
    ("decision_kendall_tau", "Decision ranking Kendall tau", 0.0),
    ("decision_pairwise_accuracy", "Decision pairwise accuracy", 0.5),
    ("decision_top1_agreement", "Decision top-1 agreement", None),
)
SEEDS = (1, 2, 3, 4)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def draw_raw_panel(axis, summaries, seed_rows, metric, chance=None):
    if chance is not None:
        axis.axhline(chance, color="#555555", linestyle="--", linewidth=1)
    for condition in CONDITIONS:
        selected = sorted(
            (
                row
                for row in summaries
                if row["metric"] == metric and row["condition"] == condition
            ),
            key=lambda row: int(row["nominal_step"]),
        )
        for seed in SEEDS:
            seed_selected = sorted(
                (row for row in seed_rows if row["condition"] == condition
                 and int(row["seed"]) == seed),
                key=lambda row: int(row["nominal_step"]),
            )
            axis.plot(
                [int(row["nominal_step"]) for row in seed_selected],
                [float(row[metric]) for row in seed_selected],
                color=COLORS[condition], linewidth=0.9, alpha=0.27,
            )
        x = np.asarray([int(row["nominal_step"]) for row in selected])
        mean = np.asarray([float(row["mean"]) for row in selected])
        low = np.asarray([float(row["ci95_low"]) for row in selected])
        high = np.asarray([float(row["ci95_high"]) for row in selected])
        axis.plot(x, mean, color=COLORS[condition], linewidth=2, label=condition)
        axis.fill_between(x, low, high, color=COLORS[condition], alpha=0.16)
    axis.grid(alpha=0.25)
    axis.set_xlabel("Environment steps")
    axis.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))


def draw_paired_panel(axis, effects, seed_rows, metric):
    axis.axhline(0, color="#555555", linestyle="--", linewidth=1)
    lookup = {
        (row["condition"], int(row["seed"]), int(row["nominal_step"])): row
        for row in seed_rows
    }
    for condition in ("a_to_c", "c_to_a"):
        selected = sorted(
            (row for row in effects if row["condition"] == condition
             and row["metric"] == metric),
            key=lambda row: int(row["nominal_step"]),
        )
        x = np.asarray([int(row["nominal_step"]) for row in selected])
        for seed in SEEDS:
            differences = [
                float(lookup[(condition, seed, step)][metric])
                - float(lookup[("none", seed, step)][metric])
                for step in x
            ]
            axis.plot(x, differences, color=COLORS[condition],
                      linewidth=0.9, alpha=0.27)
        mean = np.asarray([float(row["paired_mean_difference"]) for row in selected])
        low = np.asarray([float(row["paired_ci95_low"]) for row in selected])
        high = np.asarray([float(row["paired_ci95_high"]) for row in selected])
        axis.plot(x, mean, color=COLORS[condition], linewidth=2, label=condition)
        axis.fill_between(x, low, high, color=COLORS[condition], alpha=0.16)
    axis.grid(alpha=0.25)
    axis.set_xlabel("Environment steps")
    axis.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))


def save_figure(figure, stem):
    figure.savefig(stem.with_suffix(".png"), dpi=250, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    print(stem)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    args = parser.parse_args()
    analysis = args.analysis_root.expanduser().resolve()
    rows = read_csv(analysis / "curve_summary.csv")
    checkpoints = read_csv(analysis / "checkpoint_metrics.csv")
    effects = read_csv(analysis / "paired_effects.csv")
    figures = analysis / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    strata = sorted({(row["task"], row["align_distance"]) for row in rows})
    for task, distance in strata:
        selected = [
            row
            for row in rows
            if row["task"] == task and row["align_distance"] == distance
        ]
        selected_checkpoints = [
            row for row in checkpoints
            if row["task"] == task and row["align_distance"] == distance
        ]
        selected_effects = [
            row for row in effects
            if row["task"] == task and row["align_distance"] == distance
        ]
        figure, axes = plt.subplots(2, 2, figsize=(12, 8))
        for axis, (metric, title, ylabel) in zip(axes.flat, PANELS):
            draw_raw_panel(axis, selected, selected_checkpoints, metric,
                           chance=1.0 if metric == "epsilon_dec" else None)
            axis.set_title(title)
            axis.set_ylabel(ylabel)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
        figure.suptitle(f"H1 NPS — {task} — {distance} — each seed and mean", y=0.995)
        figure.tight_layout(rect=(0, 0, 1, 0.95))
        stem = figures / f"h1-nps-{task}-{distance}"
        save_figure(figure, stem)

        figure, axes = plt.subplots(2, 2, figsize=(12, 8))
        for axis, (metric, title, ylabel) in zip(axes.flat, PANELS):
            draw_paired_panel(axis, selected_effects, selected_checkpoints, metric)
            axis.set_title(f"{title}: alignment − none")
            axis.set_ylabel(ylabel)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
        figure.suptitle(f"H1 NPS — {task} — {distance} — seed-paired differences", y=0.995)
        figure.tight_layout(rect=(0, 0, 1, 0.95))
        save_figure(figure, figures / f"h1-nps-{task}-{distance}-paired")

        figure, axes = plt.subplots(1, 3, figsize=(15, 4))
        for axis, (metric, title, chance) in zip(axes, DECISION_VALIDITY_PANELS):
            draw_raw_panel(axis, selected, selected_checkpoints, metric, chance)
            axis.set_title(title)
            axis.set_ylabel(metric.removeprefix("decision_").replace("_", " "))
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
        figure.suptitle(
            f"H1 NPS — {task} — {distance} — decision-probe validity",
            y=1.01,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.91))
        save_figure(figure, figures / f"h1-nps-{task}-{distance}-decision-validity")


if __name__ == "__main__":
    main()
