#!/usr/bin/env python3
"""Plot the four canonical NPS H1 measurements for each distance and task."""

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


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def draw_panel(axis, rows, metric):
    for condition in CONDITIONS:
        selected = sorted(
            (
                row
                for row in rows
                if row["metric"] == metric and row["condition"] == condition
            ),
            key=lambda row: int(row["nominal_step"]),
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    args = parser.parse_args()
    analysis = args.analysis_root.expanduser().resolve()
    rows = read_csv(analysis / "curve_summary.csv")
    figures = analysis / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    strata = sorted({(row["task"], row["align_distance"]) for row in rows})
    for task, distance in strata:
        selected = [
            row
            for row in rows
            if row["task"] == task and row["align_distance"] == distance
        ]
        figure, axes = plt.subplots(2, 2, figsize=(12, 8))
        for axis, (metric, title, ylabel) in zip(axes.flat, PANELS):
            draw_panel(axis, selected, metric)
            axis.set_title(title)
            axis.set_ylabel(ylabel)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
        figure.suptitle(f"H1 NPS — {task} — {distance}", y=0.995)
        figure.tight_layout(rect=(0, 0, 1, 0.95))
        stem = figures / f"h1-nps-{task}-{distance}"
        figure.savefig(stem.with_suffix(".png"), dpi=250, bbox_inches="tight")
        figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(figure)
        print(stem)


if __name__ == "__main__":
    main()
