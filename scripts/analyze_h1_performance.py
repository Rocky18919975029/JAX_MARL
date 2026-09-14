#!/usr/bin/env python3
"""Aggregate H1 deterministic evaluation, AUCs, and paired seed effects."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


PRIMARY = {
    ("10m_vs_11m", "ps", "a_to_c"),
    ("10m_vs_11m", "nps", "a_to_c"),
    ("smacv2_10_units", "nps", "c_to_a"),
}
EFFECT_FIELDS = (
    "task",
    "actor_parameterization",
    "condition",
    "baseline",
    "metric",
    "preregistered_primary",
    "paired_mean_difference",
    "paired_ci95_low",
    "paired_ci95_high",
    "standardized_effect_dz",
    "p_value",
    "n_paired_seeds",
    "seeds",
    "bh_fdr_q",
)
COLORS = {
    "none": "#e63946",
    "a_to_c": "#80b918",
    "c_to_a": "#377bd1",
    "reciprocal": "#7f8c8d",
    "joint": "#b46f40",
    "a_to_c_shuffled": "#b5d56a",
    "c_to_a_shuffled": "#75a7e6",
}


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_mean(values, rng, repetitions=10_000):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 1:
        return float(values[0]), float(values[0])
    sample_indices = rng.integers(0, values.size, size=(repetitions, values.size))
    means = values[sample_indices].mean(axis=1)
    return tuple(np.quantile(means, (0.025, 0.975)).tolist())


def trapezoidal_integral(values, x):
    """Integrate with the API available in both NumPy 1.x and current NumPy."""

    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is not None:
        return trapezoid(values, x)
    return np.trapz(values, x)


def summarize(values, rng):
    values = np.asarray(values, dtype=np.float64)
    std = float(values.std(ddof=1)) if values.size > 1 else 0.0
    low, high = bootstrap_mean(values, rng)
    return {
        "mean": float(values.mean()),
        "std": std,
        "stderr": std / math.sqrt(values.size),
        "ci95_low": low,
        "ci95_high": high,
        "n_seeds": int(values.size),
    }


def bh_adjust(p_values):
    p_values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 1.0
    count = len(p_values)
    for reverse_rank in range(count - 1, -1, -1):
        index = order[reverse_rank]
        rank = reverse_rank + 1
        running = min(running, p_values[index] * count / rank)
        adjusted[index] = running
    return adjusted


def load_evaluations(root):
    records = []
    for path in sorted((root / "evaluation").glob("H1-*/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("policy") != "deterministic":
            continue
        condition = payload.get("condition", payload["align_mode"])
        actor = "ps" if payload["actor_parameter_sharing"] else "nps"
        actual_step = int(payload.get("checkpoint_env_step") or 0)
        nominal_step = payload.get("checkpoint_nominal_env_step")
        if path.stem == "final":
            nominal_step = 10_000_000
        elif nominal_step is None:
            nominal_step = actual_step
        records.append(
            {
                "path": str(path),
                "run_id": payload.get("run_id"),
                "run_name": path.parent.name,
                "task": payload["map_name"],
                "actor_parameterization": actor,
                "condition": condition,
                "matrix_profile": payload.get("matrix_profile", ""),
                "seed": int(payload["training_seed"]),
                "checkpoint": path.stem,
                "checkpoint_step": actual_step,
                "nominal_step": int(nominal_step),
                "return_mean": float(payload["return_mean"]),
                "win_rate": float(payload["win_rate"]),
                "episodes": int(payload["episodes"]),
                "eval_seed": int(payload["eval_seed"]),
                "protocol_version": payload.get("protocol_version", ""),
                "git_commit": payload.get("git_commit", ""),
            }
        )
    if not records:
        raise RuntimeError(f"No deterministic H1 evaluation JSON found under {root}")
    return records


def seed_endpoints(records):
    grouped = defaultdict(list)
    for record in records:
        key = (
            record["task"],
            record["actor_parameterization"],
            record["condition"],
            record["seed"],
        )
        grouped[key].append(record)
    rows = []
    for key, points in sorted(grouped.items()):
        points.sort(key=lambda row: (row["nominal_step"], row["checkpoint"] == "final"))
        # If a scheduled checkpoint and final share an x value, final is the
        # authoritative endpoint and the duplicate scheduled point is removed.
        by_step = {point["nominal_step"]: point for point in points}
        points = [by_step[step] for step in sorted(by_step)]
        x = np.asarray([point["nominal_step"] for point in points], dtype=np.float64)
        returns = np.asarray([point["return_mean"] for point in points])
        wins = np.asarray([point["win_rate"] for point in points])
        budget = 10_000_000.0
        rows.append(
            {
                "task": key[0],
                "actor_parameterization": key[1],
                "condition": key[2],
                "seed": key[3],
                "return_auc": float(trapezoidal_integral(returns, x) / budget),
                "win_rate_auc": float(trapezoidal_integral(wins, x) / budget),
                "final_return": float(returns[-1]),
                "final_win_rate": float(wins[-1]),
                "last_nominal_step": int(x[-1]),
                "num_checkpoints": len(points),
            }
        )
    return rows


def curve_rows(records, rng):
    grouped = defaultdict(list)
    for record in records:
        key = (
            record["task"],
            record["actor_parameterization"],
            record["condition"],
            record["nominal_step"],
        )
        grouped[key].append(record)
    rows = []
    for key, values in sorted(grouped.items()):
        for metric in ("return_mean", "win_rate"):
            summary = summarize([value[metric] for value in values], rng)
            rows.append(
                {
                    "task": key[0],
                    "actor_parameterization": key[1],
                    "condition": key[2],
                    "nominal_step": key[3],
                    "metric": metric,
                    **summary,
                }
            )
    return rows


def endpoint_summaries(endpoints, rng):
    grouped = defaultdict(list)
    for row in endpoints:
        grouped[(row["task"], row["actor_parameterization"], row["condition"])].append(
            row
        )
    output = []
    for key, rows in sorted(grouped.items()):
        for metric in (
            "return_auc",
            "win_rate_auc",
            "final_return",
            "final_win_rate",
        ):
            output.append(
                {
                    "task": key[0],
                    "actor_parameterization": key[1],
                    "condition": key[2],
                    "metric": metric,
                    **summarize([row[metric] for row in rows], rng),
                }
            )
    return output


def paired_effects(endpoints, rng):
    lookup = {
        (
            row["task"],
            row["actor_parameterization"],
            row["condition"],
            row["seed"],
        ): row
        for row in endpoints
    }
    strata = sorted(
        set((row["task"], row["actor_parameterization"]) for row in endpoints)
    )
    output = []
    for task, actor in strata:
        conditions = sorted(
            set(
                row["condition"]
                for row in endpoints
                if row["task"] == task and row["actor_parameterization"] == actor
            )
            - {"none"}
        )
        for condition in conditions:
            seeds = sorted(
                row[3]
                for row in lookup
                if row[:3] == (task, actor, condition)
                and (task, actor, "none", row[3]) in lookup
            )
            if not seeds:
                continue
            for metric in (
                "return_auc",
                "win_rate_auc",
                "final_return",
                "final_win_rate",
            ):
                differences = np.asarray(
                    [
                        lookup[(task, actor, condition, seed)][metric]
                        - lookup[(task, actor, "none", seed)][metric]
                        for seed in seeds
                    ]
                )
                low, high = bootstrap_mean(differences, rng)
                std = float(differences.std(ddof=1)) if len(seeds) > 1 else 0.0
                # Two-sided paired bootstrap p-value around a zero null.
                indices = rng.integers(0, len(seeds), size=(10_000, len(seeds)))
                centered = differences - differences.mean()
                null_means = centered[indices].mean(axis=1)
                p_value = float(
                    min(
                        1.0,
                        (np.sum(np.abs(null_means) >= abs(differences.mean())) + 1)
                        / 10_001,
                    )
                )
                output.append(
                    {
                        "task": task,
                        "actor_parameterization": actor,
                        "condition": condition,
                        "baseline": "none",
                        "metric": metric,
                        "preregistered_primary": ((task, actor, condition) in PRIMARY),
                        "paired_mean_difference": float(differences.mean()),
                        "paired_ci95_low": low,
                        "paired_ci95_high": high,
                        "standardized_effect_dz": (
                            float(differences.mean() / std) if std > 0 else math.nan
                        ),
                        "p_value": p_value,
                        "n_paired_seeds": len(seeds),
                        "seeds": ";".join(map(str, seeds)),
                    }
                )
    secondary_indices = [
        index for index, row in enumerate(output) if not row["preregistered_primary"]
    ]
    adjusted = bh_adjust([output[index]["p_value"] for index in secondary_indices])
    for row in output:
        row["bh_fdr_q"] = ""
    for index, q_value in zip(secondary_indices, adjusted):
        output[index]["bh_fdr_q"] = float(q_value)
    return output


def figures(curves, output_dir):
    import matplotlib.pyplot as plt

    grouped = defaultdict(list)
    for row in curves:
        grouped[(row["task"], row["actor_parameterization"], row["metric"])].append(row)
    for (task, actor, metric), rows in grouped.items():
        figure, axis = plt.subplots(figsize=(8, 5))
        for condition in COLORS:
            selected = sorted(
                (row for row in rows if row["condition"] == condition),
                key=lambda row: row["nominal_step"],
            )
            if not selected:
                continue
            x = np.asarray([row["nominal_step"] for row in selected])
            mean = np.asarray([row["mean"] for row in selected])
            low = np.asarray([row["ci95_low"] for row in selected])
            high = np.asarray([row["ci95_high"] for row in selected])
            linestyle = "--" if condition.endswith("_shuffled") else "-"
            axis.plot(
                x,
                mean,
                label=condition,
                color=COLORS[condition],
                linestyle=linestyle,
                linewidth=2,
            )
            axis.fill_between(x, low, high, color=COLORS[condition], alpha=0.16)
        axis.set_title(f"H1 {task} / {actor.upper()} / {metric}")
        axis.set_xlabel("Environment steps")
        axis.set_ylabel(
            "Evaluation return" if metric == "return_mean" else "Evaluation win rate"
        )
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, ncol=2)
        figure.tight_layout()
        stem = f"{task}-{actor}-{metric}"
        figure.savefig(output_dir / f"{stem}.png", dpi=200)
        figure.savefig(output_dir / f"{stem}.pdf")
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20260913)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    root = args.run_root.expanduser().resolve()
    output = root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.bootstrap_seed)

    records = load_evaluations(root)
    endpoints = seed_endpoints(records)
    curves = curve_rows(records, rng)
    summaries = endpoint_summaries(endpoints, rng)
    effects = paired_effects(endpoints, rng)

    write_csv(output / "evaluation_records.csv", records, list(records[0]))
    write_csv(output / "seed_endpoints.csv", endpoints, list(endpoints[0]))
    write_csv(output / "curve_summary.csv", curves, list(curves[0]))
    write_csv(output / "endpoint_summary.csv", summaries, list(summaries[0]))
    write_csv(
        output / "confirmatory_effects.csv",
        effects,
        list(effects[0]) if effects else EFFECT_FIELDS,
    )
    if not args.no_figures:
        figure_dir = output / "figures"
        figure_dir.mkdir(exist_ok=True)
        figures(curves, figure_dir)
    print(f"Evaluations: {len(records)}")
    print(f"Seed-level curves: {len(endpoints)}")
    print(output)


if __name__ == "__main__":
    main()
