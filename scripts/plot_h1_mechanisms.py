#!/usr/bin/env python3
"""Create the preregistered three-panel H1 mechanism figures."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "none": "#e63946",
    "a_to_c": "#80b918",
    "c_to_a": "#377bd1",
    "joint": "#b46f40",
    "reciprocal": "#7f8c8d",
    "a_to_c_cka": "#80b918",
    "c_to_a_cka": "#377bd1",
    "a_to_c_shuffled": "#b5d56a",
    "c_to_a_shuffled": "#75a7e6",
}
CONDITION_ORDER = tuple(COLORS)


def read(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def conditions_for(rows, task, actor):
    available = {
        row["condition"]
        for row in rows
        if row["task"] == task and row["actor_parameterization"] == actor
    }
    return tuple(condition for condition in CONDITION_ORDER if condition in available)


def draw_curve(axis, rows, metric, conditions):
    for condition in conditions:
        selected = sorted(
            (
                row
                for row in rows
                if row["metric"] == metric and row["condition"] == condition
            ),
            key=lambda row: int(row["nominal_step"]),
        )
        if not selected:
            continue
        x = np.asarray([int(row["nominal_step"]) for row in selected])
        mean = np.asarray([float(row["mean"]) for row in selected])
        low = np.asarray([float(row["ci95_low"]) for row in selected])
        high = np.asarray([float(row["ci95_high"]) for row in selected])
        if condition.endswith("_cka"):
            linestyle = ":"
        elif condition.endswith("_shuffled"):
            linestyle = "--"
        else:
            linestyle = "-"
        axis.plot(
            x,
            mean,
            label=condition,
            color=COLORS[condition],
            linestyle=linestyle,
            linewidth=2,
        )
        axis.fill_between(x, low, high, color=COLORS[condition], alpha=0.16)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.expanduser().resolve()
    analysis = root / "analysis"
    performance = read(analysis / "curve_summary.csv")
    mechanisms = read(analysis / "mechanism_curve_summary.csv")
    prospective = read(analysis / "prospective_prediction_points.csv")
    output = analysis / "figures" / "mechanisms"
    output.mkdir(parents=True, exist_ok=True)

    strata = sorted(
        set((row["task"], row["actor_parameterization"]) for row in mechanisms)
    )
    for task, actor in strata:
        conditions = conditions_for(mechanisms, task, actor)
        performance_rows = [
            row
            for row in performance
            if row["task"] == task and row["actor_parameterization"] == actor
        ]
        mechanism_rows = [
            row
            for row in mechanisms
            if row["task"] == task and row["actor_parameterization"] == actor
        ]
        prospective_rows = [
            row
            for row in prospective
            if row["task"] == task
            and row["actor_parameterization"] == actor
            and row["condition"] in conditions
        ]
        figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        draw_curve(axes[0], performance_rows, "return_mean", conditions)
        axes[0].set_title("Deterministic evaluation return")
        axes[0].set_xlabel("Environment steps")
        axes[0].set_ylabel("Return")
        draw_curve(axes[1], mechanism_rows, "r_lat", conditions)
        axes[1].set_title("Relative latent distortion")
        axes[1].set_xlabel("Environment steps")
        axes[1].set_ylabel(r"$r_{lat}$ (lower is better)")
        for condition in conditions:
            selected = [
                row for row in prospective_rows if row["condition"] == condition
            ]
            if selected:
                axes[2].scatter(
                    [float(row["early_r_lat"]) for row in selected],
                    [float(row["future_return_gain"]) for row in selected],
                    color=COLORS[condition],
                    label=condition,
                    alpha=0.8,
                )
        axes[2].set_title("Early distortion vs future gain")
        axes[2].set_xlabel(r"Early $r_{lat}$ (5--20%)")
        axes[2].set_ylabel("Return gain (60% - 20%)")
        for axis in axes:
            axis.grid(alpha=0.25)
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper center", ncol=len(conditions))
        figure.suptitle(f"H1: {task} / {actor.upper()}", y=1.04)
        figure.tight_layout()
        stem = output / f"h1-{task}-{actor}-mechanisms"
        figure.savefig(stem.with_suffix(".png"), dpi=250, bbox_inches="tight")
        figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(figure)
        print(stem)


if __name__ == "__main__":
    main()
