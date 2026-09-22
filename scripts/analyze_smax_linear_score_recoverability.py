#!/usr/bin/env python3
"""Task-separated tables and plots for linear SMAX score recoverability."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CONDITION_LABEL = {
    "none": "Isolated",
    "c_to_a_mse": "C → A\n(LN-MSE)",
    "c_to_a_cka": "C → A\n(Linear CKA)",
}
COLORS = {
    "none": "#383838",
    "c_to_a_mse": "#D55E00",
    "c_to_a_cka": "#0072B2",
}
TASK_LABEL = {
    "10m_vs_11m": "SMAX — 10m vs 11m",
    "3s5z_vs_3s6z": "SMAX — 3s5z vs 3s6z",
    "smacv2_10_units": "SMACv2 — 10 units",
}


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        raise RuntimeError(f"Cannot write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def seed_bootstrap(values: list[float]) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(values)) or len(values) == 0:
        raise ValueError("Seed values must be nonempty and finite")
    mean = float(values.mean())
    if len(values) == 1:
        return mean, math.nan, math.nan
    rng = np.random.default_rng(20260923)
    indices = rng.integers(0, len(values), size=(100_000, len(values)))
    replicates = values[indices].mean(axis=1)
    return (
        mean,
        float(np.percentile(replicates, 2.5)),
        float(np.percentile(replicates, 97.5)),
    )


def read_results(root: Path):
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    if protocol["protocol"] != "smax-linear-score-recoverability-v1.0":
        raise RuntimeError("Not a linear score-recoverability run root")
    seed_rows = []
    agent_rows = []
    support_rows = []
    for task in protocol["tasks"]:
        for condition in protocol["conditions"]:
            for seed in protocol["seeds"]:
                cell = root / "runs" / task / condition / f"seed_{seed}"
                summary_path = cell / "summary.json"
                if not summary_path.is_file():
                    raise RuntimeError(f"Incomplete linear-probe matrix: {cell}")
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                seed_rows.append(
                    {
                        "task": task,
                        "condition": condition,
                        "seed": seed,
                        "num_agents": summary["num_agents"],
                        "epsilon_rec_lin_normalized": summary[
                            "epsilon_rec_lin_normalized"
                        ],
                        "epsilon_rec_lin": summary["epsilon_rec_lin"],
                        "zero_predictor_normalized": 1.0,
                        "fallback_test_fraction": summary["fallback_test_fraction"],
                        "max_agent_fallback_test_fraction": summary[
                            "max_agent_fallback_test_fraction"
                        ],
                        "fit_samples_per_agent": summary["fit_samples_per_agent"],
                        "test_samples_per_agent": summary["test_samples_per_agent"],
                        "fisher_ridge_absolute": summary["fisher_ridge_absolute"],
                        "ridge": summary["ridge"],
                        "min_fit_per_action": summary["min_fit_per_action"],
                    }
                )
                with (cell / "agent_metrics.csv").open(
                    newline="", encoding="utf-8"
                ) as file:
                    agent_rows.extend(csv.DictReader(file))
                with (cell / "action_support.csv").open(
                    newline="", encoding="utf-8"
                ) as file:
                    support_rows.extend(csv.DictReader(file))
    return protocol, seed_rows, agent_rows, support_rows


def summarize(protocol: dict, seed_rows: list[dict]) -> list[dict]:
    output = []
    for task in protocol["tasks"]:
        for condition in protocol["conditions"]:
            selected = [
                row
                for row in seed_rows
                if row["task"] == task and row["condition"] == condition
            ]
            error = [float(row["epsilon_rec_lin_normalized"]) for row in selected]
            raw = [float(row["epsilon_rec_lin"]) for row in selected]
            support = [float(row["fallback_test_fraction"]) for row in selected]
            mean, low, high = seed_bootstrap(error)
            raw_mean, raw_low, raw_high = seed_bootstrap(raw)
            output.append(
                {
                    "task": task,
                    "condition": condition,
                    "n_training_seeds": len(selected),
                    "epsilon_rec_lin_normalized_mean": mean,
                    "epsilon_rec_lin_normalized_ci95_low": low,
                    "epsilon_rec_lin_normalized_ci95_high": high,
                    "epsilon_rec_lin_mean": raw_mean,
                    "epsilon_rec_lin_ci95_low": raw_low,
                    "epsilon_rec_lin_ci95_high": raw_high,
                    "zero_predictor_normalized": 1.0,
                    "fraction_seeds_worse_than_zero": sum(x > 1 for x in error)
                    / len(error),
                    "fallback_test_fraction_mean": float(np.mean(support)),
                    "fallback_test_fraction_max_seed": float(np.max(support)),
                    "bootstrap_unit": "training_seed",
                    "bootstrap_method": "ordinary percentile bootstrap",
                    "bootstrap_repetitions": 100_000 if len(selected) > 1 else 0,
                }
            )
    return output


def plot_task(
    task: str, protocol: dict, seed_rows: list[dict], summary: list[dict], path: Path
):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 300,
        }
    )
    figure, ax = plt.subplots(figsize=(5.5, 3.8))
    for x, condition in enumerate(protocol["conditions"]):
        row = next(
            item
            for item in summary
            if item["task"] == task and item["condition"] == condition
        )
        mean = float(row["epsilon_rec_lin_normalized_mean"])
        low = float(row["epsilon_rec_lin_normalized_ci95_low"])
        high = float(row["epsilon_rec_lin_normalized_ci95_high"])
        color = COLORS.get(condition, "#555555")
        ax.bar(x, mean, width=0.58, color=color, alpha=0.84, zorder=2)
        if math.isfinite(low) and math.isfinite(high):
            ax.errorbar(
                x,
                mean,
                yerr=[[max(0, mean - low)], [max(0, high - mean)]],
                fmt="none",
                color="#1B1B1B",
                capsize=4,
                zorder=4,
            )
        points = [
            float(item["epsilon_rec_lin_normalized"])
            for item in seed_rows
            if item["task"] == task and item["condition"] == condition
        ]
        ax.scatter(
            x + np.linspace(-0.12, 0.12, len(points)),
            points,
            s=29,
            facecolor="white",
            edgecolor=color,
            linewidth=1.2,
            zorder=5,
        )
    ax.axhline(1.0, color="#6A6A6A", linestyle="--", linewidth=1, zorder=1)
    ax.set_xticks(
        list(range(len(protocol["conditions"]))),
        [CONDITION_LABEL.get(item, item) for item in protocol["conditions"]],
    )
    ax.set_ylabel(
        r"Held-out normalized error $\widehat\epsilon_{\mathrm{Rec}}^{\mathrm{lin}}$"
    )
    ax.set_title(TASK_LABEL.get(task, task), fontweight="semibold")
    ax.grid(axis="y", color="#E1E1E1", alpha=0.75, zorder=0)
    ax.set_axisbelow(True)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(path.with_suffix(f".{suffix}"), bbox_inches="tight")
    plt.close(figure)


def analyze(root: Path):
    root = root.expanduser().resolve()
    protocol, seed_rows, agent_rows, support_rows = read_results(root)
    summary = summarize(protocol, seed_rows)
    output = root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "seed_level.csv", seed_rows)
    write_csv(output / "agent_level.csv", agent_rows)
    write_csv(output / "action_support.csv", support_rows)
    write_csv(output / "task_condition_summary.csv", summary)
    for task in protocol["tasks"]:
        task_root = output / task
        task_root.mkdir(parents=True, exist_ok=True)
        for filename, rows in (
            ("seed_level.csv", seed_rows),
            ("agent_level.csv", agent_rows),
            ("action_support.csv", support_rows),
            ("task_condition_summary.csv", summary),
        ):
            write_csv(
                task_root / filename, [row for row in rows if row["task"] == task]
            )
        plot_task(
            task,
            protocol,
            seed_rows,
            summary,
            task_root / "linear-score-recoverability",
        )
        print(task_root / "task_condition_summary.csv", flush=True)
        print(task_root / "linear-score-recoverability.png", flush=True)
    (output / "analysis_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol": protocol["protocol"],
                "metric": "held-out normalized action-conditioned linear score error",
                "tasks_are_never_pooled": True,
                "zero_predictor_baseline": 1.0,
                "test_error_is_not_clipped": True,
                "agent_aggregation": "unweighted mean within seed",
                "seed_aggregation": "mean and training-seed bootstrap 95% CI",
                "rare_action_policy": protocol["rare_action_policy"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.run_root)


if __name__ == "__main__":
    main()
