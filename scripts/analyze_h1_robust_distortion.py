#!/usr/bin/env python3
"""Aggregate and plot seed-paired robust H1 distortion diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CONDITION_ORDER = ("c_to_a", "a_to_c", "joint", "reciprocal")
COLORS = {
    "c_to_a": "#377bd1",
    "a_to_c": "#80b918",
    "joint": "#bc6c35",
    "reciprocal": "#8d6eaa",
}
LABELS = {
    "c_to_a": "C → A",
    "a_to_c": "A → C",
    "joint": "Joint",
    "reciprocal": "Reciprocal",
}
METRICS = (
    ("heldout_return", "Held-out stochastic return", "Higher is better"),
    ("epsilon_lat_raw", r"Raw distortion $\epsilon_{Lat}^{raw}$", "Lower is better"),
    (
        "epsilon_lat_phase_matched",
        r"Phase-marginalized distortion $\epsilon_{Lat}^{phase}$",
        "Lower is better",
    ),
    (
        "fisher_natural_gradient_cosine",
        "Fisher-natural gradient cosine",
        "Higher is better",
    ),
    (
        "epsilon_lat_optimal_scale",
        r"Optimal-scale distortion $\epsilon_{Lat}^{scale}$",
        "Lower is better",
    ),
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path, rows):
    if not rows:
        raise RuntimeError(f"Refusing to write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def canonical_condition(value):
    return str(value).removesuffix("_cka")


def load_checkpoint_rows(metrics_root):
    rows = []
    summary_paths = sorted(
        (metrics_root / "checkpoints").glob("*/*/robust_distortion_summary.json")
    )
    run_manifest_path = metrics_root / "run_manifest.json"
    if run_manifest_path.is_file():
        expected = int(read_json(run_manifest_path)["discovered"])
        if len(summary_paths) != expected:
            raise RuntimeError(
                f"Robust recomputation is incomplete: {len(summary_paths)}/{expected} "
                "checkpoint summaries"
            )
    for summary_path in summary_paths:
        summary = read_json(summary_path)
        expected_cells = {
            (aggregation, label, float(ridge))
            for aggregation in ("slot", "slot_x_type")
            for label in ("M", "2M", "4M")
            for ridge in summary["fisher_ridges"]
        }
        actual_cells = {
            (
                metric["aggregation"],
                metric["episode_budget_label"],
                float(metric["fisher_ridge_absolute"]),
            )
            for metric in summary["aggregate_metrics"]
        }
        if actual_cells != expected_cells or len(summary["aggregate_metrics"]) != len(
            expected_cells
        ):
            raise RuntimeError(f"Incomplete metric grid in {summary_path}")
        condition = canonical_condition(summary["condition"])
        distance = str(summary.get("align_distance", "ln_mse"))
        if condition == "none":
            distance = "distance_free"
        for metric in summary["aggregate_metrics"]:
            rows.append(
                {
                    "run_name": summary["run_name"],
                    "task": summary["task"],
                    "actor_parameterization": "nps",
                    "native_align_distance": distance,
                    "condition": condition,
                    "seed": int(summary["seed"]),
                    "checkpoint_step": int(summary["checkpoint_step"]),
                    "checkpoint_nominal_step": int(summary["checkpoint_nominal_step"]),
                    "aggregation": metric["aggregation"],
                    "episode_budget": int(metric["episode_budget"]),
                    "episode_budget_label": metric["episode_budget_label"],
                    "fisher_ridge_absolute": float(metric["fisher_ridge_absolute"]),
                    "heldout_return": float(summary["heldout_episode_return_mean"]),
                    **{
                        name: float(metric[name])
                        for name, _, _ in METRICS
                        if name != "heldout_return"
                    },
                    "robust_distortion_protocol": summary["robust_distortion_protocol"],
                    "phase_protocol": summary["phase_protocol"],
                    "source_summary": str(summary_path),
                }
            )
    if not rows:
        raise RuntimeError(f"No robust distortion summaries under {metrics_root}")
    return rows


def paired_rows(checkpoint_rows):
    distances_by_task = defaultdict(set)
    for row in checkpoint_rows:
        if row["condition"] != "none":
            distances_by_task[row["task"]].add(row["native_align_distance"])
    baseline = {}
    aligned = {}
    for row in checkpoint_rows:
        key_tail = (
            row["seed"],
            row["checkpoint_nominal_step"],
            row["aggregation"],
            row["episode_budget_label"],
            row["fisher_ridge_absolute"],
        )
        if row["condition"] == "none":
            for distance in distances_by_task[row["task"]]:
                key = (row["task"], distance, "none", *key_tail)
                if key in baseline:
                    raise RuntimeError(f"Duplicate none baseline: {key}")
                baseline[key] = row
        else:
            key = (
                row["task"],
                row["native_align_distance"],
                row["condition"],
                *key_tail,
            )
            if key in aligned:
                raise RuntimeError(f"Duplicate aligned checkpoint metric: {key}")
            aligned[key] = row
    baseline_tails = defaultdict(set)
    aligned_tails = defaultdict(set)
    for key in baseline:
        task, distance, _, *tail = key
        baseline_tails[(task, distance)].add(tuple(tail))
    for key in aligned:
        task, distance, condition, *tail = key
        aligned_tails[(task, distance, condition)].add(tuple(tail))
    for (task, distance, condition), tails in aligned_tails.items():
        expected = baseline_tails[(task, distance)]
        if tails != expected:
            raise RuntimeError(
                f"Incomplete seed-paired matrix for {task}/{distance}/{condition}: "
                f"missing={len(expected - tails)}, unexpected={len(tails - expected)}"
            )
    output = []
    for key, target in sorted(aligned.items()):
        task, distance, condition, seed, step, aggregation, budget, ridge = key
        control_key = (
            task,
            distance,
            "none",
            seed,
            step,
            aggregation,
            budget,
            ridge,
        )
        if control_key not in baseline:
            raise RuntimeError(f"Missing seed-paired none baseline for {key}")
        control = baseline[control_key]
        output.append(
            {
                "task": task,
                "actor_parameterization": "nps",
                "align_distance": distance,
                "condition": condition,
                "baseline": "none",
                "seed": seed,
                "checkpoint_nominal_step": step,
                "aggregation": aggregation,
                "episode_budget_label": budget,
                "episode_budget": target["episode_budget"],
                "fisher_ridge_absolute": ridge,
                **{
                    f"delta_{metric}": target[metric] - control[metric]
                    for metric, _, _ in METRICS
                },
                **{f"aligned_{metric}": target[metric] for metric, _, _ in METRICS},
                **{f"baseline_{metric}": control[metric] for metric, _, _ in METRICS},
            }
        )
    return output


def summarize_paired(rows):
    grouped = defaultdict(list)
    for row in rows:
        prefix = (
            row["task"],
            row["align_distance"],
            row["condition"],
            row["checkpoint_nominal_step"],
            row["aggregation"],
            row["episode_budget_label"],
            row["episode_budget"],
            row["fisher_ridge_absolute"],
        )
        for metric, _, _ in METRICS:
            grouped[(*prefix, metric)].append(float(row[f"delta_{metric}"]))
    output = []
    for key, values in sorted(grouped.items()):
        array = np.asarray(values, dtype=np.float64)
        std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
        output.append(
            {
                "task": key[0],
                "actor_parameterization": "nps",
                "align_distance": key[1],
                "condition": key[2],
                "baseline": "none",
                "checkpoint_nominal_step": key[3],
                "aggregation": key[4],
                "episode_budget_label": key[5],
                "episode_budget": key[6],
                "fisher_ridge_absolute": key[7],
                "metric": key[8],
                "paired_mean_difference": float(array.mean()),
                "paired_std": std,
                "paired_stderr": std / math.sqrt(len(array)),
                "n_paired_seeds": len(array),
            }
        )
    return output


def matching_ridge(value, target):
    return math.isclose(float(value), float(target), rel_tol=1e-12, abs_tol=1e-15)


def conditions_in_order(rows):
    observed = {row["condition"] for row in rows}
    ordered = [item for item in CONDITION_ORDER if item in observed]
    ordered.extend(sorted(observed - set(ordered)))
    return ordered


def plot_temporal(
    figures,
    task,
    distance,
    aggregation,
    ridge,
    budget_label,
    seed_rows,
    summary_rows,
):
    selected_seed = [
        row
        for row in seed_rows
        if row["task"] == task
        and row["align_distance"] == distance
        and row["aggregation"] == aggregation
        and row["episode_budget_label"] == budget_label
        and matching_ridge(row["fisher_ridge_absolute"], ridge)
    ]
    selected_summary = [
        row
        for row in summary_rows
        if row["task"] == task
        and row["align_distance"] == distance
        and row["aggregation"] == aggregation
        and row["episode_budget_label"] == budget_label
        and matching_ridge(row["fisher_ridge_absolute"], ridge)
    ]
    conditions = conditions_in_order(selected_seed)
    figure, axes = plt.subplots(3, 2, figsize=(13.2, 12.0), sharex=True)
    for axis, (metric, title, direction) in zip(axes.flat, METRICS):
        axis.axhline(0.0, color="#555555", linestyle="--", linewidth=1)
        for condition in conditions:
            rows = [row for row in selected_seed if row["condition"] == condition]
            seeds = sorted({int(row["seed"]) for row in rows})
            for seed in seeds:
                trajectory = sorted(
                    (row for row in rows if int(row["seed"]) == seed),
                    key=lambda row: int(row["checkpoint_nominal_step"]),
                )
                axis.plot(
                    [int(row["checkpoint_nominal_step"]) for row in trajectory],
                    [float(row[f"delta_{metric}"]) for row in trajectory],
                    color=COLORS.get(condition, "#555555"),
                    linewidth=0.85,
                    alpha=0.24,
                )
            means = sorted(
                (
                    row
                    for row in selected_summary
                    if row["condition"] == condition and row["metric"] == metric
                ),
                key=lambda row: int(row["checkpoint_nominal_step"]),
            )
            x = np.asarray([int(row["checkpoint_nominal_step"]) for row in means])
            mean = np.asarray([float(row["paired_mean_difference"]) for row in means])
            stderr = np.asarray([float(row["paired_stderr"]) for row in means])
            axis.plot(
                x,
                mean,
                color=COLORS.get(condition, "#555555"),
                linewidth=2.3,
                label=LABELS.get(condition, condition),
            )
            axis.fill_between(
                x,
                mean - stderr,
                mean + stderr,
                color=COLORS.get(condition, "#555555"),
                alpha=0.16,
            )
        axis.set_title(f"{title}: alignment − none")
        axis.set_ylabel(f"Seed-paired difference ({direction})")
        axis.grid(alpha=0.25)
        axis.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
        axis.set_xlabel("Environment steps")
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.suptitle(
        f"{task} — {distance} — {aggregation} — ξ={ridge:g} — {budget_label}",
        fontsize=14,
        y=0.985,
    )
    figure.legend(
        handles,
        labels,
        title="Alignment condition",
        loc="lower center",
        bbox_to_anchor=(0.74, 0.13),
        ncol=2,
        frameon=True,
    )
    figure.tight_layout(rect=(0.02, 0.02, 0.98, 0.95), h_pad=2.0, w_pad=2.0)
    stem = figures / (
        f"robust-{task}-{distance}-{aggregation}-ridge{ridge:g}-{budget_label}-paired"
    )
    figure.savefig(stem.with_suffix(".png"), dpi=250, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return stem


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--plot-ridge", type=float, default=0.001)
    parser.add_argument("--plot-budget-label", default="4M", choices=("M", "2M", "4M"))
    return parser.parse_args()


def main():
    args = parse_args()
    metrics_root = args.metrics_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_rows = load_checkpoint_rows(metrics_root)
    seed_rows = paired_rows(checkpoint_rows)
    summary_rows = summarize_paired(seed_rows)
    write_csv(output_root / "checkpoint_metrics.csv", checkpoint_rows)
    write_csv(output_root / "paired_seed_differences.csv", seed_rows)
    write_csv(output_root / "paired_curve_summary.csv", summary_rows)
    write_csv(
        output_root / "m_2m_4m_convergence.csv",
        [
            row
            for row in checkpoint_rows
            if row["episode_budget_label"] in {"M", "2M", "4M"}
        ],
    )
    write_csv(
        output_root / "fisher_ridge_sweep.csv",
        [row for row in checkpoint_rows if row["episode_budget_label"] == "4M"],
    )
    figures = output_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    strata = sorted(
        {(row["task"], row["align_distance"], row["aggregation"]) for row in seed_rows}
    )
    stems = [
        plot_temporal(
            figures,
            task,
            distance,
            aggregation,
            args.plot_ridge,
            args.plot_budget_label,
            seed_rows,
            summary_rows,
        )
        for task, distance, aggregation in strata
    ]
    manifest = {
        "schema_version": 1,
        "metrics_root": str(metrics_root),
        "plot_ridge": args.plot_ridge,
        "plot_budget_label": args.plot_budget_label,
        "seed_pairing": "alignment minus none within task, seed, checkpoint, aggregation, budget, and ridge",
        "uncertainty": "mean plus/minus standard error across training seeds",
        "figures": [str(stem.with_suffix(".png")) for stem in stems],
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
