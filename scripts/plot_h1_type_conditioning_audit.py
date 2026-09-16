#!/usr/bin/env python3
"""Plot pooled versus slot-by-type H1 offline audit curves."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


COLORS = {
    "none": "#e63946",
    "a_to_c": "#80b918",
    "c_to_a": "#377bd1",
    "joint": "#b86f3c",
}
LABELS = {
    "none": "none",
    "a_to_c": "A → C",
    "c_to_a": "C → A",
    "joint": "joint",
}


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def selected(rows, task, distance, condition, metric):
    return sorted(
        (
            row
            for row in rows
            if row["task"] == task
            and row["align_distance"] == distance
            and row["condition"] == condition
            and row["metric"] == metric
        ),
        key=lambda row: int(row["nominal_step"]),
    )


def draw(axis, rows, task, distance, metric, conditions, linestyle="-"):
    for condition in conditions:
        values = selected(rows, task, distance, condition, metric)
        if not values:
            continue
        x = np.asarray([int(row["nominal_step"]) for row in values])
        mean = np.asarray([float(row["mean"]) for row in values])
        low = np.asarray([float(row["ci95_low"]) for row in values])
        high = np.asarray([float(row["ci95_high"]) for row in values])
        axis.plot(
            x,
            mean,
            color=COLORS[condition],
            linestyle=linestyle,
            linewidth=2,
        )
        axis.fill_between(x, low, high, color=COLORS[condition], alpha=0.10)


def figure(curves, paired, task, distance, output):
    conditions = tuple(
        condition
        for condition in ("none", "a_to_c", "c_to_a", "joint")
        if selected(curves, task, distance, condition, "heldout_return_mean")
    )
    aligned = tuple(condition for condition in conditions if condition != "none")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex="col")
    fig.suptitle(
        f"NPS type-conditioning audit — {task} — {distance}", fontsize=16, y=0.985
    )

    draw(axes[0, 0], curves, task, distance, "heldout_return_mean", conditions)
    draw(axes[0, 1], curves, task, distance, "epsilon_lat_slot", conditions)
    draw(
        axes[0, 1],
        curves,
        task,
        distance,
        "epsilon_lat_slot_type",
        conditions,
        "--",
    )
    draw(axes[0, 2], curves, task, distance, "linear_cka_distance_slot", conditions)
    draw(
        axes[0, 2],
        curves,
        task,
        distance,
        "linear_cka_distance_slot_type",
        conditions,
        "--",
    )
    axes[0, 0].set_title("Held-out return")
    axes[0, 1].set_title(r"Latent distortion $\epsilon_{Lat}$")
    axes[0, 2].set_title("Held-out Linear CKA distance")

    for axis in axes[1]:
        axis.axhline(0.0, color="#555555", linestyle=":", linewidth=1)
    draw(axes[1, 0], paired, task, distance, "heldout_return_mean", aligned)
    draw(axes[1, 1], paired, task, distance, "epsilon_lat_slot", aligned)
    draw(
        axes[1, 1],
        paired,
        task,
        distance,
        "epsilon_lat_slot_type",
        aligned,
        "--",
    )
    draw(axes[1, 2], paired, task, distance, "linear_cka_distance_slot", aligned)
    draw(
        axes[1, 2],
        paired,
        task,
        distance,
        "linear_cka_distance_slot_type",
        aligned,
        "--",
    )
    axes[1, 0].set_title("Paired return: alignment − none")
    axes[1, 1].set_title(r"Paired $\epsilon_{Lat}$: alignment − none")
    axes[1, 2].set_title("Paired CKA distance: alignment − none")

    for axis in axes.flat:
        axis.grid(alpha=0.22)
        axis.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
        axis.set_xlabel("Environment steps")
    axes[0, 0].set_ylabel("Absolute metric")
    axes[1, 0].set_ylabel("Seed-paired difference")

    condition_handles = [
        Line2D([0], [0], color=COLORS[item], lw=2, label=LABELS[item])
        for item in conditions
    ]
    scope_handles = [
        Line2D([0], [0], color="#222222", lw=2, linestyle="-", label="slot-pooled"),
        Line2D(
            [0],
            [0],
            color="#222222",
            lw=2,
            linestyle="--",
            label="slot × type",
        ),
    ]
    fig.legend(
        handles=condition_handles,
        loc="upper center",
        bbox_to_anchor=(0.42, 0.948),
        ncol=len(condition_handles),
        frameon=False,
        title="Condition",
    )
    fig.legend(
        handles=scope_handles,
        loc="upper center",
        bbox_to_anchor=(0.81, 0.948),
        ncol=2,
        frameon=False,
        title="Measurement grouping",
    )
    fig.subplots_adjust(top=0.84, left=0.07, right=0.98, bottom=0.08, wspace=0.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.audit_root.expanduser().resolve()
    curves = read_csv(root / "tables" / "curve_summary.csv")
    paired = read_csv(root / "tables" / "paired_curve_summary.csv")
    combinations = sorted({(row["task"], row["align_distance"]) for row in curves})
    for task, distance in combinations:
        output = root / "figures" / f"type-audit-{task}-{distance}.png"
        figure(curves, paired, task, distance, output)
        print(output)


if __name__ == "__main__":
    main()
