#!/usr/bin/env python3
"""Task-separated return/recoverability trajectories at matched checkpoints."""

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
    "c_to_a_mse": "C → A (LN-MSE)",
    "c_to_a_cka": "C → A (Linear CKA)",
}
PALETTE_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "smax_score_recoverability_palette.yaml"
)
PALETTE = json.loads(PALETTE_PATH.read_text(encoding="utf-8"))
ROLES = PALETTE["chart_roles"]
STYLE = {
    "none": (ROLES["isolated"], "--", "o"),
    "c_to_a_mse": (ROLES["c_to_a_ln_mse"], "-.", "^"),
    "c_to_a_cka": (ROLES["c_to_a_linear_cka"], "-", "s"),
}
TASK_DISPLAY = {
    "10m_vs_11m": "SMAX — 10m_vs_11m",
    "3s5z_vs_3s6z": "SMAX — 3s5z_vs_3s6z",
    "smacv2_10_units": "SMACv2 — 10 units",
}


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percentile_bootstrap(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("Bootstrap values must be nonempty and finite")
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
    if protocol.get("protocol") != "smax-score-recoverability-resmlp-v2.0":
        raise RuntimeError("Analysis requires the v2 residual-MLP protocol")
    agent_rows, seed_rows, missing = [], [], []
    for task in protocol["tasks"]:
        budget = int(protocol["selected_budgets"][task])
        for checkpoint in protocol["checkpoint_plan"][task]:
            step = int(checkpoint["env_step"])
            for condition in protocol["conditions"]:
                for seed in protocol["seeds"]:
                    directory = (
                        run_root
                        / "runs"
                        / task
                        / condition
                        / f"seed_{seed}"
                        / f"step_{step:012d}"
                    )
                    summary_path = directory / "summary.json"
                    table_path = directory / "agent_metrics.csv"
                    if not summary_path.is_file() or not table_path.is_file():
                        missing.append(f"{task}:{condition}:seed{seed}:step{step}")
                        continue
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    expected_provenance = {
                        "protocol": protocol["protocol"],
                        "checkpoint_env_step": step,
                        "task": task,
                        "condition": condition,
                        "training_seed": seed,
                    }
                    actual_provenance = {
                        key: summary.get(key) for key in expected_provenance
                    }
                    mismatches = {
                        key: {"expected": value, "actual": actual_provenance[key]}
                        for key, value in expected_provenance.items()
                        if actual_provenance[key] != value
                    }
                    if mismatches:
                        raise RuntimeError(
                            f"Result provenance mismatch: {summary_path}: {mismatches}"
                        )
                    seed_rows.append(
                        {
                            "task": task,
                            "condition": condition,
                            "seed": seed,
                            "checkpoint_env_step": step,
                            "checkpoint_fraction_actual": step / budget,
                            "checkpoint_fraction_requested": checkpoint[
                                "requested_fraction"
                            ],
                            "training_budget_env_steps": budget,
                            "num_agents": summary["num_agents"],
                            "on_policy_episode_return_mean": summary[
                                "on_policy_episode_return_mean"
                            ],
                            "on_policy_episode_return_se": summary[
                                "on_policy_episode_return_se"
                            ],
                            "epsilon_rec": summary["epsilon_rec"],
                            "epsilon_rec_normalized": summary["epsilon_rec_normalized"],
                            "fit_epsilon_rec_normalized": summary[
                                "fit_epsilon_rec_normalized"
                            ],
                            "validation_epsilon_rec_normalized": summary[
                                "validation_epsilon_rec_normalized"
                            ],
                            "estimator_failure_agents_gt_one": summary[
                                "estimator_failure_agents_gt_one"
                            ],
                            "episodes": summary["episodes"],
                            "fit_samples_per_agent": summary["fit_samples_per_agent"],
                            "validation_samples_per_agent": summary[
                                "validation_samples_per_agent"
                            ],
                            "test_samples_per_agent": summary["test_samples_per_agent"],
                            "fisher_ridge_absolute": summary["fisher_ridge_absolute"],
                        }
                    )
                    with table_path.open(newline="", encoding="utf-8") as file:
                        agent_rows.extend(csv.DictReader(file))
    if missing:
        raise RuntimeError(f"Recoverability matrix is incomplete: {missing}")
    return protocol, agent_rows, seed_rows


def summarize(protocol, seed_rows):
    rows = []
    for task in protocol["tasks"]:
        for checkpoint in protocol["checkpoint_plan"][task]:
            step = int(checkpoint["env_step"])
            for condition in protocol["conditions"]:
                selected = [
                    row
                    for row in seed_rows
                    if row["task"] == task
                    and row["condition"] == condition
                    and int(row["checkpoint_env_step"]) == step
                ]
                if len(selected) != len(protocol["seeds"]):
                    raise RuntimeError(
                        f"Incomplete seed group: {task}/{condition}/{step}"
                    )
                metrics = {}
                repetitions = 0
                for name in (
                    "on_policy_episode_return_mean",
                    "epsilon_rec_normalized",
                    "fit_epsilon_rec_normalized",
                    "validation_epsilon_rec_normalized",
                ):
                    mean, low, high, repetitions = percentile_bootstrap(
                        [float(row[name]) for row in selected]
                    )
                    metrics[f"{name}_mean"] = mean
                    metrics[f"{name}_ci95_low"] = low
                    metrics[f"{name}_ci95_high"] = high
                rows.append(
                    {
                        "task": task,
                        "condition": condition,
                        "checkpoint_env_step": step,
                        "checkpoint_fraction_actual": checkpoint["actual_fraction"],
                        "checkpoint_fraction_requested": checkpoint[
                            "requested_fraction"
                        ],
                        "n_training_seeds": len(selected),
                        **metrics,
                        "estimator_failure_seeds_gt_one": sum(
                            float(row["epsilon_rec_normalized"]) > 1 for row in selected
                        ),
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
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def draw_task(axes, task, protocol, rows):
    budget = float(protocol["selected_budgets"][task])
    for condition in protocol["conditions"]:
        selected = sorted(
            (
                row
                for row in rows
                if row["task"] == task and row["condition"] == condition
            ),
            key=lambda row: int(row["checkpoint_env_step"]),
        )
        x = np.asarray([float(row["checkpoint_env_step"]) / 1e6 for row in selected])
        color, line, marker = STYLE[condition]
        for ax, prefix in (
            (axes[0], "on_policy_episode_return_mean"),
            (axes[1], "epsilon_rec_normalized"),
        ):
            y = np.asarray([float(row[f"{prefix}_mean"]) for row in selected])
            low = np.asarray([float(row[f"{prefix}_ci95_low"]) for row in selected])
            high = np.asarray([float(row[f"{prefix}_ci95_high"]) for row in selected])
            ax.plot(
                x,
                y,
                color=color,
                linestyle=line,
                marker=marker,
                markersize=5,
                linewidth=1.3,
                label=DISPLAY[condition],
                zorder=3,
            )
            if np.all(np.isfinite(low)) and np.all(np.isfinite(high)):
                ax.fill_between(x, low, high, color=color, alpha=0.13, linewidth=0)
    axes[0].set_ylabel("Stochastic return ↑")
    axes[1].set_ylabel(r"Score error $\widehat{\epsilon}_{\rm Rec}$ ↓")
    axes[1].axhline(1.0, color=ROLES["zero_predictor"], linestyle=":", linewidth=0.9)
    axes[1].text(
        0.98,
        1.0,
        "zero predictor = 1",
        ha="right",
        va="bottom",
        fontsize=7,
        color=ROLES["zero_predictor"],
        transform=axes[1].get_yaxis_transform(),
    )
    for ax in axes:
        ax.set_xlabel("Environment steps (millions)")
        ax.set_xlim(0, budget / 1e6 * 1.03)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.7)
        ax.set_axisbelow(True)


def plot(protocol, summary_rows, output):
    style()
    for task in protocol["tasks"]:
        figure, axes = plt.subplots(1, 2, figsize=(178 / 25.4, 80 / 25.4))
        draw_task(axes, task, protocol, summary_rows)
        figure.suptitle(TASK_DISPLAY.get(task, task), fontweight="semibold")
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="lower center",
            ncol=3,
            bbox_to_anchor=(0.5, -0.04),
            frameon=False,
        )
        figure.tight_layout(rect=(0, 0.06, 1, 0.96))
        task_root = output / task
        task_root.mkdir(parents=True, exist_ok=True)
        for suffix in ("png", "pdf", "svg"):
            figure.savefig(
                task_root / f"return-vs-score-recoverability.{suffix}",
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
    for name, rows in (
        ("agent_level", agent_rows),
        ("seed_checkpoint_level", seed_rows),
        ("task_condition_checkpoint_summary", summary_rows),
    ):
        write_csv(output / f"{name}.csv", rows)
        for task in protocol["tasks"]:
            write_csv(
                output / task / f"{name}.csv",
                [row for row in rows if row["task"] == task],
            )
    plot(protocol, summary_rows, output)
    manifest = {
        "schema_version": 2,
        "protocol": protocol["protocol"],
        "tasks_are_never_pooled": True,
        "metric": "held-out normalized vector squared error; never clipped",
        "agent_aggregation": "unweighted mean within training seed/checkpoint",
        "seed_aggregation": "mean with 95% ordinary bootstrap CI",
        "return_source": "the same stochastic on-policy episodes as the probe data",
        "checkpoint_plan": protocol["checkpoint_plan"],
    }
    (output / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output / "task_condition_checkpoint_summary.csv")
    for task in protocol["tasks"]:
        print(output / task / "return-vs-score-recoverability.png")


if __name__ == "__main__":
    main()
