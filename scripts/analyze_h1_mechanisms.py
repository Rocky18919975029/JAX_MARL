#!/usr/bin/env python3
"""Aggregate checkpoint mechanisms and perform seed-paired H1 analyses."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_ci(values, rng, repetitions=10_000):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 1:
        return float(values[0]), float(values[0])
    index = rng.integers(0, len(values), size=(repetitions, len(values)))
    return tuple(np.quantile(values[index].mean(axis=1), (0.025, 0.975)))


def ranks(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    output = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        output[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return output


def spearman(first, second):
    first_rank = ranks(first)
    second_rank = ranks(second)
    if np.std(first_rank) == 0 or np.std(second_rank) == 0:
        return math.nan
    return float(np.corrcoef(first_rank, second_rank)[0, 1])


def load_mechanism_rows(root):
    rows = []
    for metadata_path in sorted(
        (root / "diagnostics_raw").glob("H1-*/*/metadata.json")
    ):
        directory = metadata_path.parent
        required = (
            directory / "latent_distortion_summary.json",
            directory / "decision_summary.json",
            directory / "bellman_summary.json",
        )
        if not all(path.is_file() for path in required):
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        latent = json.loads(required[0].read_text(encoding="utf-8"))
        decision = json.loads(required[1].read_text(encoding="utf-8"))
        bellman = json.loads(required[2].read_text(encoding="utf-8"))
        rows.append(
            {
                "run_id": metadata.get("run_id"),
                "run_name": metadata["run_name"],
                "task": metadata["map_name"],
                "actor_parameterization": (
                    "ps" if metadata["actor_parameter_sharing"] else "nps"
                ),
                "condition": metadata["condition"],
                "matrix_profile": metadata.get("matrix_profile", ""),
                "seed": int(metadata["training_seed"]),
                "checkpoint_step": int(metadata.get("checkpoint_env_step") or 0),
                "nominal_step": int(metadata.get("checkpoint_nominal_env_step") or 0),
                "epsilon_lat": float(latent["epsilon_lat"]),
                "r_lat": float(latent["r_lat"]),
                "epsilon_dec": float(decision["epsilon_dec"]),
                "kendall_tau": float(decision["kendall_tau"]),
                "pairwise_accuracy": float(decision["pairwise_accuracy"]),
                "top1_agreement": float(decision["top1_agreement"]),
                "epsilon_bell": float(bellman["epsilon_bell"]),
                "epsilon_bell_excess": float(bellman["epsilon_bell_excess"]),
                "protocol_version": metadata["protocol_version"],
                "git_commit": metadata["git_commit"],
            }
        )
    if not rows:
        raise RuntimeError("No complete mechanism summaries found")
    return rows


def curve_summary(rows, rng):
    grouped = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["task"],
                row["actor_parameterization"],
                row["condition"],
                row["nominal_step"],
            )
        ].append(row)
    output = []
    metrics = ("r_lat", "epsilon_lat", "epsilon_dec", "epsilon_bell_excess")
    for key, group in sorted(grouped.items()):
        for metric in metrics:
            values = np.asarray([row[metric] for row in group])
            low, high = bootstrap_ci(values, rng)
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            output.append(
                {
                    "task": key[0],
                    "actor_parameterization": key[1],
                    "condition": key[2],
                    "nominal_step": key[3],
                    "metric": metric,
                    "mean": float(values.mean()),
                    "std": std,
                    "stderr": std / math.sqrt(len(values)),
                    "ci95_low": low,
                    "ci95_high": high,
                    "n_seeds": len(values),
                }
            )
    return output


def paired_checkpoint_effects(rows, rng):
    lookup = {
        (
            row["task"],
            row["actor_parameterization"],
            row["condition"],
            row["seed"],
            row["nominal_step"],
        ): row
        for row in rows
    }
    output = []
    strata = sorted(
        set(
            (
                row["task"],
                row["actor_parameterization"],
                row["condition"],
                row["nominal_step"],
            )
            for row in rows
            if row["condition"] != "none"
        )
    )
    for task, actor, condition, step in strata:
        seeds = sorted(
            row[3]
            for row in lookup
            if row[:3] == (task, actor, condition)
            and row[4] == step
            and (task, actor, "none", row[3], step) in lookup
        )
        for metric in ("r_lat", "epsilon_dec", "epsilon_bell_excess"):
            if not seeds:
                continue
            differences = np.asarray(
                [
                    lookup[(task, actor, condition, seed, step)][metric]
                    - lookup[(task, actor, "none", seed, step)][metric]
                    for seed in seeds
                ]
            )
            low, high = bootstrap_ci(differences, rng)
            row = {
                "task": task,
                "actor_parameterization": actor,
                "condition": condition,
                "baseline": "none",
                "nominal_step": step,
                "metric": metric,
                "paired_mean_difference": float(differences.mean()),
                "paired_ci95_low": low,
                "paired_ci95_high": high,
                "n_paired_seeds": len(seeds),
                "seeds": ";".join(map(str, seeds)),
                "noninferiority_margin": "",
                "noninferiority_pass": "",
            }
            if metric == "epsilon_dec":
                row["noninferiority_margin"] = 0.02
                row["noninferiority_pass"] = bool(high <= 0.02)
            elif metric == "epsilon_bell_excess":
                ratios = np.asarray(
                    [
                        lookup[(task, actor, condition, seed, step)][metric]
                        / max(lookup[(task, actor, "none", seed, step)][metric], 1e-12)
                        for seed in seeds
                    ]
                )
                ratio_low, ratio_high = bootstrap_ci(ratios, rng)
                row["noninferiority_margin"] = 1.05
                row["noninferiority_ratio_mean"] = float(ratios.mean())
                row["noninferiority_ratio_ci95_low"] = ratio_low
                row["noninferiority_ratio_ci95_high"] = ratio_high
                row["noninferiority_pass"] = bool(ratio_high <= 1.05)
            output.append(row)
    # DictWriter requires a common schema even for metric-specific fields.
    all_fields = set().union(*(row.keys() for row in output))
    return [{field: row.get(field, "") for field in all_fields} for row in output]


def prospective(rows, evaluation_rows, rng):
    rlat_group = defaultdict(list)
    for row in rows:
        if 500_000 <= row["nominal_step"] <= 2_000_000:
            rlat_group[
                (
                    row["task"],
                    row["actor_parameterization"],
                    row["condition"],
                    row["seed"],
                )
            ].append(row["r_lat"])
    eval_lookup = {
        (
            row["task"],
            row["actor_parameterization"],
            row["condition"],
            int(row["seed"]),
            int(row["nominal_step"]),
        ): float(row["return_mean"])
        for row in evaluation_rows
    }
    points = []
    for key, values in rlat_group.items():
        early = eval_lookup.get((*key, 2_000_000))
        later = eval_lookup.get((*key, 6_000_000))
        if early is None or later is None:
            continue
        points.append(
            {
                "task": key[0],
                "actor_parameterization": key[1],
                "condition": key[2],
                "seed": key[3],
                "early_r_lat": float(np.mean(values)),
                "future_return_gain": later - early,
            }
        )
    correlations = []
    grouped = defaultdict(list)
    for point in points:
        grouped[(point["task"], point["actor_parameterization"])].append(point)
    for key, group in sorted(grouped.items()):
        first = -np.asarray([point["early_r_lat"] for point in group])
        second = np.asarray([point["future_return_gain"] for point in group])
        observed = spearman(first, second)
        bootstrap = []
        for _ in range(10_000):
            index = rng.integers(0, len(group), size=len(group))
            bootstrap.append(spearman(first[index], second[index]))
        finite = np.asarray([value for value in bootstrap if np.isfinite(value)])
        correlations.append(
            {
                "task": key[0],
                "actor_parameterization": key[1],
                "spearman_negative_early_r_lat_vs_future_gain": observed,
                "ci95_low": (
                    float(np.quantile(finite, 0.025)) if len(finite) else math.nan
                ),
                "ci95_high": (
                    float(np.quantile(finite, 0.975)) if len(finite) else math.nan
                ),
                "n_runs": len(group),
            }
        )
    return points, correlations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20260914)
    args = parser.parse_args()
    root = args.run_root.expanduser().resolve()
    analysis = root / "analysis"
    evaluation_path = analysis / "evaluation_records.csv"
    if not evaluation_path.is_file():
        raise FileNotFoundError(
            f"Run analyze_h1_performance.py first: {evaluation_path}"
        )
    evaluation_rows = read_csv(evaluation_path)
    rows = load_mechanism_rows(root)
    rng = np.random.default_rng(args.bootstrap_seed)
    curves = curve_summary(rows, rng)
    effects = paired_checkpoint_effects(rows, rng)
    prospective_points, correlations = prospective(rows, evaluation_rows, rng)
    write_csv(analysis / "checkpoint_mechanism_metrics.csv", rows)
    write_csv(analysis / "mechanism_curve_summary.csv", curves)
    write_csv(analysis / "mechanism_paired_effects.csv", effects)
    write_csv(analysis / "prospective_prediction_points.csv", prospective_points)
    write_csv(analysis / "prospective_prediction_summary.csv", correlations)
    print(f"Complete checkpoint summaries: {len(rows)}")
    print(analysis)


if __name__ == "__main__":
    main()
