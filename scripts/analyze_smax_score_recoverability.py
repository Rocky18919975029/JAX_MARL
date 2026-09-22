#!/usr/bin/env python3
"""Aggregate and plot task-separated SMAX score-recoverability results."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DISPLAY = {
    "none": "Isolated",
    "c_to_a_mse": "C → A\n(LN-MSE)",
    "c_to_a_cka": "C → A\n(Linear CKA)",
}
COLORS = {
    "none": "#333333",
    "c_to_a_mse": "#D55E00",
    "c_to_a_cka": "#0072B2",
}
TASK_DISPLAY = {
    "10m_vs_11m": "SMAX — 10m_vs_11m",
    "3s5z_vs_3s6z": "SMAX — 3s5z_vs_3s6z",
    "smacv2_10_units": "SMACv2 — 10 units",
}


def write_csv(path: Path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def percentile_bootstrap(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return float(values.mean()), math.nan, math.nan, 0
    if len(values) <= 6:
        indices = np.asarray(
            list(itertools.product(range(len(values)), repeat=len(values))),
            dtype=np.int16,
        )
    else:
        indices = np.random.default_rng(20260923).integers(
            0, len(values), size=(100_000, len(values))
        )
    samples = values[indices].mean(axis=1)
    return (
        float(values.mean()),
        float(np.percentile(samples, 2.5)),
        float(np.percentile(samples, 97.5)),
        len(samples),
    )


def collect(run_root: Path):
    protocol = json.loads((run_root / "protocol.json").read_text(encoding="utf-8"))
    agent_rows = []
    seed_rows = []
    missing = []
    for task in protocol["tasks"]:
        for condition in protocol["conditions"]:
            for seed in protocol["seeds"]:
                directory = run_root / "runs" / task / condition / f"seed_{seed}"
                summary_path = directory / "summary.json"
                table_path = directory / "agent_metrics.csv"
                if not summary_path.is_file() or not table_path.is_file():
                    missing.append(f"{task}:{condition}:seed{seed}")
                    continue
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                seed_rows.append(
                    {
                        "task": task,
                        "condition": condition,
                        "seed": seed,
                        "training_budget_env_steps": protocol["selected_budgets"][task],
                        "num_agents": summary["num_agents"],
                        "epsilon_rec": summary["epsilon_rec"],
                        "epsilon_rec_normalized": summary["epsilon_rec_normalized"],
                        "episodes": summary["episodes"],
                        "fit_samples_per_agent": summary["fit_samples_per_agent"],
                        "test_samples_per_agent": summary["test_samples_per_agent"],
                        "fisher_ridge_absolute": summary["fisher_ridge_absolute"],
                    }
                )
                with table_path.open(newline="", encoding="utf-8") as file:
                    for row in csv.DictReader(file):
                        agent_rows.append(row)
    if missing:
        raise RuntimeError(f"Recoverability matrix is incomplete: {missing}")
    return protocol, agent_rows, seed_rows


def summarize(protocol, seed_rows):
    rows = []
    for task in protocol["tasks"]:
        for condition in protocol["conditions"]:
            selected = [
                row
                for row in seed_rows
                if row["task"] == task and row["condition"] == condition
            ]
            values = [float(row["epsilon_rec_normalized"]) for row in selected]
            raw = [float(row["epsilon_rec"]) for row in selected]
            mean, low, high, repetitions = percentile_bootstrap(values)
            raw_mean, raw_low, raw_high, _ = percentile_bootstrap(raw)
            rows.append(
                {
                    "task": task,
                    "condition": condition,
                    "n_training_seeds": len(values),
                    "epsilon_rec_normalized_mean": mean,
                    "epsilon_rec_normalized_ci95_low": low,
                    "epsilon_rec_normalized_ci95_high": high,
                    "epsilon_rec_mean": raw_mean,
                    "epsilon_rec_ci95_low": raw_low,
                    "epsilon_rec_ci95_high": raw_high,
                    "bootstrap_unit": "training_seed",
                    "bootstrap_method": "ordinary percentile bootstrap",
                    "bootstrap_repetitions": repetitions,
                }
            )
    return rows


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.0,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 180,
            "savefig.dpi": 300,
        }
    )


def draw_task(ax, task, protocol, seed_rows, summary_rows):
    conditions = list(protocol["conditions"])
    positions = np.arange(len(conditions))
    for position, condition in zip(positions, conditions):
        summary = next(
            row
            for row in summary_rows
            if row["task"] == task and row["condition"] == condition
        )
        mean = float(summary["epsilon_rec_normalized_mean"])
        low = float(summary["epsilon_rec_normalized_ci95_low"])
        high = float(summary["epsilon_rec_normalized_ci95_high"])
        ax.bar(
            position,
            mean,
            width=0.64,
            color=COLORS[condition],
            alpha=0.87,
            edgecolor="white",
            linewidth=0.9,
            zorder=2,
        )
        ax.errorbar(
            position,
            mean,
            yerr=np.asarray([[mean - low], [high - mean]]),
            color="#111111",
            capsize=4,
            linewidth=1.2,
            zorder=4,
        )
        points = [
            float(row["epsilon_rec_normalized"])
            for row in seed_rows
            if row["task"] == task and row["condition"] == condition
        ]
        offsets = np.linspace(-0.12, 0.12, len(points))
        ax.scatter(
            position + offsets,
            points,
            s=22,
            facecolor="white",
            edgecolor=COLORS[condition],
            linewidth=1.1,
            zorder=5,
        )
    ax.set_xticks(positions, [DISPLAY[item] for item in conditions])
    ax.set_title(TASK_DISPLAY.get(task, task), fontweight="semibold", pad=10)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)


def plot(protocol, seed_rows, summary_rows, output_root):
    style()
    tasks = list(protocol["tasks"])
    figure, axes = plt.subplots(1, len(tasks), figsize=(4.15 * len(tasks), 3.7))
    axes = np.atleast_1d(axes)
    for ax, task in zip(axes, tasks):
        draw_task(ax, task, protocol, seed_rows, summary_rows)
    axes[0].set_ylabel(r"Normalized recoverability error $\epsilon_{\mathrm{Rec}}$")
    figure.suptitle("Policy-score recoverability from critic representations", y=1.02)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(
            output_root / f"smax-score-recoverability.{suffix}",
            bbox_inches="tight",
        )
    plt.close(figure)

    for task in tasks:
        figure, ax = plt.subplots(figsize=(4.7, 3.8))
        draw_task(ax, task, protocol, seed_rows, summary_rows)
        ax.set_ylabel(r"Normalized recoverability error $\epsilon_{\mathrm{Rec}}$")
        figure.tight_layout()
        task_root = output_root / task
        task_root.mkdir(parents=True, exist_ok=True)
        for suffix in ("png", "pdf"):
            figure.savefig(
                task_root / f"score-recoverability.{suffix}",
                bbox_inches="tight",
            )
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.expanduser().resolve()
    protocol, agent_rows, seed_rows = collect(root)
    summary_rows = summarize(protocol, seed_rows)
    output = root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "agent_level.csv", agent_rows)
    write_csv(output / "seed_level.csv", seed_rows)
    write_csv(output / "task_condition_summary.csv", summary_rows)
    for task in protocol["tasks"]:
        task_root = output / task
        task_root.mkdir(parents=True, exist_ok=True)
        write_csv(
            task_root / "agent_level.csv",
            [row for row in agent_rows if row["task"] == task],
        )
        write_csv(
            task_root / "seed_level.csv",
            [row for row in seed_rows if row["task"] == task],
        )
        write_csv(
            task_root / "task_condition_summary.csv",
            [row for row in summary_rows if row["task"] == task],
        )
    plot(protocol, seed_rows, summary_rows, output)
    manifest = {
        "schema_version": 1,
        "protocol": protocol["protocol"],
        "tasks_are_never_pooled": True,
        "metric": "held-out normalized vector squared error",
        "agent_aggregation": "unweighted mean within training seed",
        "seed_aggregation": "mean with 95% ordinary bootstrap CI",
        "tables": {
            "agent_level": str(output / "agent_level.csv"),
            "seed_level": str(output / "seed_level.csv"),
            "summary": str(output / "task_condition_summary.csv"),
        },
        "figure": str(output / "smax-score-recoverability.png"),
    }
    (output / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output / "task_condition_summary.csv")
    print(output / "smax-score-recoverability.png")


if __name__ == "__main__":
    main()
