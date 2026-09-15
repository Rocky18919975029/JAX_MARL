#!/usr/bin/env python3
"""Analyze the canonical NPS H1 experiment across LN-MSE and Linear CKA."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


REFERENCE_PROTOCOL = "baseline_free_mc_return_train_matched_gae"
TASKS = ("10m_vs_11m", "smacv2_10_units")
DISTANCES = ("ln_mse", "linear_cka")
CONDITIONS = ("none", "a_to_c", "c_to_a")
SEEDS = (1, 2, 3, 4)
CHECKPOINT_STEPS = (
    0,
    500_000,
    1_000_000,
    2_000_000,
    4_000_000,
    6_000_000,
    8_000_000,
    10_000_000,
)
METRICS = ("heldout_return", "epsilon_lat", "epsilon_dec", "epsilon_bell")
DECISION_AUDIT_METRICS = (
    "decision_kendall_tau",
    "decision_pairwise_accuracy",
    "decision_top1_agreement",
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path, rows):
    if not rows:
        raise RuntimeError(f"Refusing to write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_ci(values, rng, repetitions):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        raise ValueError("Cannot bootstrap an empty sample")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    indices = rng.integers(0, len(values), size=(repetitions, len(values)))
    means = values[indices].mean(axis=1)
    return tuple(map(float, np.quantile(means, (0.025, 0.975))))


def summarize(values, rng, repetitions):
    values = np.asarray(values, dtype=np.float64)
    standard_deviation = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    low, high = bootstrap_ci(values, rng, repetitions)
    return {
        "mean": float(values.mean()),
        "std": standard_deviation,
        "stderr": standard_deviation / math.sqrt(len(values)),
        "ci95_low": low,
        "ci95_high": high,
        "n_seeds": len(values),
    }


def canonical_condition(raw_condition):
    condition = raw_condition.removesuffix("_cka")
    return condition if condition in CONDITIONS else None


def nominal_step(metadata, directory):
    value = metadata.get("checkpoint_nominal_env_step")
    if value is not None:
        return int(value)
    if directory.name == "initial":
        return 0
    if directory.name == "final":
        config = read_json(Path(metadata["checkpoint"]) / "config.json")
        return int(config["TOTAL_TIMESTEPS"])
    if directory.name.startswith("step_"):
        return int(directory.name.removeprefix("step_"))
    raise RuntimeError(f"Cannot infer nominal checkpoint step for {directory}")


def load_root(root, expected_distance, expected_conditions, ridge_absolute):
    metadata_paths = sorted((root / "diagnostics_raw").glob("H1-*/*/metadata.json"))
    selected = []
    incomplete = []
    for metadata_path in metadata_paths:
        directory = metadata_path.parent
        metadata = read_json(metadata_path)
        if metadata.get("actor_parameter_sharing"):
            continue
        distance = metadata.get("align_distance", "ln_mse")
        condition = canonical_condition(metadata["condition"])
        if distance != expected_distance or condition not in expected_conditions:
            continue
        paths = {
            "latent": directory / "latent_summary.json",
            "decision": directory / "decision_summary.json",
            "bellman": directory / "bellman_summary.json",
        }
        missing = [name for name, path in paths.items() if not path.is_file()]
        if missing:
            incomplete.append(f"{directory}: {','.join(missing)}")
            continue
        latent = read_json(paths["latent"])
        decision = read_json(paths["decision"])
        bellman = read_json(paths["bellman"])
        if latent.get("reference_protocol") != REFERENCE_PROTOCOL:
            raise RuntimeError(f"Obsolete latent protocol in {paths['latent']}")
        if float(latent["fisher_ridge_absolute"]) != ridge_absolute:
            raise RuntimeError(f"Fisher ridge mismatch in {paths['latent']}")
        if decision.get("probe_scope") != "one_independent_probe_per_agent":
            raise RuntimeError(
                f"Obsolete decision probe protocol in {paths['decision']}"
            )
        for label, value in (
            ("heldout return", latent["heldout_episode_return_mean"]),
            ("epsilon_lat", latent["epsilon_lat"]),
            ("epsilon_dec", decision["epsilon_dec"]),
            ("epsilon_bell", bellman["epsilon_bell"]),
            ("decision Kendall tau", decision["kendall_tau"]),
            ("decision pairwise accuracy", decision["pairwise_accuracy"]),
            ("decision top-1 agreement", decision["top1_agreement"]),
        ):
            if not math.isfinite(float(value)):
                raise RuntimeError(f"Non-finite {label} in {directory}")
        convergence = {
            item["episode_budget_label"]: item for item in latent["mc_convergence"]
        }
        if set(convergence) != {"M", "2M", "4M"}:
            raise RuntimeError(
                f"Incomplete MC gradient convergence audit in {directory}"
            )
        step = nominal_step(metadata, directory)
        selected.append(
            {
                "run_id": metadata.get("run_id"),
                "run_name": metadata["run_name"],
                "task": metadata["map_name"],
                "actor_parameterization": "nps",
                "align_distance": expected_distance,
                "condition": condition,
                "seed": int(metadata["training_seed"]),
                "checkpoint_step": int(metadata.get("checkpoint_env_step") or step),
                "nominal_step": step,
                "heldout_return": float(latent["heldout_episode_return_mean"]),
                "heldout_return_std": float(latent["heldout_episode_return_std"]),
                "heldout_return_stderr": float(latent["heldout_episode_return_stderr"]),
                "heldout_episodes": int(latent["heldout_episodes"]),
                "alignment_coef": float(metadata["alignment_coef"]),
                "mc_ref_cosine_m_to_4m": float(
                    convergence["M"]["reference_gradient_cosine_to_4m"]
                ),
                "mc_ref_cosine_2m_to_4m": float(
                    convergence["2M"]["reference_gradient_cosine_to_4m"]
                ),
                "epsilon_lat_relative_change_m_to_4m": float(
                    convergence["M"]["epsilon_lat_relative_error_to_4m"]
                ),
                "epsilon_lat_relative_change_2m_to_4m": float(
                    convergence["2M"]["epsilon_lat_relative_error_to_4m"]
                ),
                "epsilon_lat": float(latent["epsilon_lat"]),
                "epsilon_dec": float(decision["epsilon_dec"]),
                "epsilon_bell": float(bellman["epsilon_bell"]),
                "decision_kendall_tau": float(decision["kendall_tau"]),
                "decision_pairwise_accuracy": float(decision["pairwise_accuracy"]),
                "decision_top1_agreement": float(decision["top1_agreement"]),
                "decision_num_test_anchor_agents": int(
                    decision["num_test_anchor_agents"]
                ),
                "fisher_ridge_absolute": ridge_absolute,
                "reference_protocol": REFERENCE_PROTOCOL,
                "protocol_version": metadata["protocol_version"],
                "git_commit": metadata["git_commit"],
                "baseline_source": "native",
            }
        )
    if incomplete:
        preview = "\n".join(incomplete[:10])
        raise RuntimeError(
            f"{len(incomplete)} selected checkpoints lack canonical diagnostics:\n{preview}"
        )
    return selected


def validate_and_reuse_baseline(mse_rows, cka_rows):
    mse_coefficients = {row["alignment_coef"] for row in mse_rows}
    cka_coefficients = {row["alignment_coef"] for row in cka_rows}
    if mse_coefficients != {0.1} or len(cka_coefficients) != 1:
        raise RuntimeError(
            f"Unmatched alignment coefficients: LN-MSE={mse_coefficients}, "
            f"Linear CKA={cka_coefficients}"
        )
    expected_mse = {
        (task, condition, seed, step)
        for task in TASKS
        for condition in CONDITIONS
        for seed in SEEDS
        for step in CHECKPOINT_STEPS
    }
    expected_cka = {
        (task, condition, seed, step)
        for task in TASKS
        for condition in ("a_to_c", "c_to_a")
        for seed in SEEDS
        for step in CHECKPOINT_STEPS
    }

    def keys(rows):
        return {
            (row["task"], row["condition"], row["seed"], row["nominal_step"])
            for row in rows
        }

    for label, actual, expected in (
        ("LN-MSE", keys(mse_rows), expected_mse),
        ("Linear CKA", keys(cka_rows), expected_cka),
    ):
        if actual != expected or len(actual) != len(expected):
            raise RuntimeError(
                f"{label} matrix mismatch: missing={sorted(expected - actual)[:8]}, "
                f"unexpected={sorted(actual - expected)[:8]}"
            )

    reused = []
    for row in mse_rows:
        if row["condition"] == "none":
            copy = dict(row)
            copy["align_distance"] = "linear_cka"
            copy["baseline_source"] = "reused_ln_mse_none"
            reused.append(copy)
    rows = mse_rows + cka_rows + reused
    expected_count = (
        len(TASKS)
        * len(DISTANCES)
        * len(CONDITIONS)
        * len(SEEDS)
        * len(CHECKPOINT_STEPS)
    )
    if len(rows) != expected_count:
        raise RuntimeError(f"Expected {expected_count} records, found {len(rows)}")
    return sorted(
        rows,
        key=lambda row: (
            row["task"],
            row["align_distance"],
            CONDITIONS.index(row["condition"]),
            row["seed"],
            row["nominal_step"],
        ),
    )


def curve_summary(rows, rng, repetitions):
    grouped = defaultdict(list)
    for row in rows:
        for metric in (*METRICS, *DECISION_AUDIT_METRICS):
            grouped[
                (
                    row["task"],
                    row["align_distance"],
                    row["condition"],
                    row["nominal_step"],
                    metric,
                )
            ].append(row[metric])
    return [
        {
            "task": key[0],
            "actor_parameterization": "nps",
            "align_distance": key[1],
            "condition": key[2],
            "nominal_step": key[3],
            "metric": key[4],
            **summarize(values, rng, repetitions),
        }
        for key, values in sorted(grouped.items())
    ]


def paired_effects(rows, rng, repetitions):
    lookup = {
        (
            row["task"],
            row["align_distance"],
            row["condition"],
            row["seed"],
            row["nominal_step"],
        ): row
        for row in rows
    }
    output = []
    for task in TASKS:
        for distance in DISTANCES:
            for condition in ("a_to_c", "c_to_a"):
                for step in CHECKPOINT_STEPS:
                    for metric in METRICS:
                        differences = [
                            lookup[(task, distance, condition, seed, step)][metric]
                            - lookup[(task, distance, "none", seed, step)][metric]
                            for seed in SEEDS
                        ]
                        summary = summarize(differences, rng, repetitions)
                        output.append(
                            {
                                "task": task,
                                "actor_parameterization": "nps",
                                "align_distance": distance,
                                "condition": condition,
                                "baseline": "none",
                                "nominal_step": step,
                                "metric": metric,
                                "paired_mean_difference": summary["mean"],
                                "paired_std": summary["std"],
                                "paired_stderr": summary["stderr"],
                                "paired_ci95_low": summary["ci95_low"],
                                "paired_ci95_high": summary["ci95_high"],
                                "n_paired_seeds": summary["n_seeds"],
                                "seeds": ";".join(map(str, SEEDS)),
                            }
                        )
    return output


def auc_analysis(rows, rng, repetitions):
    grouped = defaultdict(dict)
    for row in rows:
        grouped[(row["task"], row["align_distance"], row["condition"], row["seed"])][
            row["nominal_step"]
        ] = row["heldout_return"]
    seed_rows = []
    for key, values in sorted(grouped.items()):
        x = np.asarray(CHECKPOINT_STEPS, dtype=np.float64)
        y = np.asarray([values[step] for step in CHECKPOINT_STEPS], dtype=np.float64)
        seed_rows.append(
            {
                "task": key[0],
                "actor_parameterization": "nps",
                "align_distance": key[1],
                "condition": key[2],
                "seed": key[3],
                "return_auc_normalized": float(np.trapezoid(y, x) / x[-1]),
            }
        )
    lookup = {
        (row["task"], row["align_distance"], row["condition"], row["seed"]): row
        for row in seed_rows
    }
    summary_rows = []
    for task in TASKS:
        for distance in DISTANCES:
            for condition in ("a_to_c", "c_to_a"):
                differences = [
                    lookup[(task, distance, condition, seed)]["return_auc_normalized"]
                    - lookup[(task, distance, "none", seed)]["return_auc_normalized"]
                    for seed in SEEDS
                ]
                summary_rows.append(
                    {
                        "task": task,
                        "actor_parameterization": "nps",
                        "align_distance": distance,
                        "condition": condition,
                        "baseline": "none",
                        **{
                            f"paired_{key}": value
                            for key, value in summarize(
                                differences, rng, repetitions
                            ).items()
                        },
                    }
                )
    return seed_rows, summary_rows


def temporal_precedence(rows, rng, repetitions):
    lookup = {
        (
            row["task"],
            row["align_distance"],
            row["condition"],
            row["seed"],
            row["nominal_step"],
        ): row
        for row in rows
    }
    points = []
    for task in TASKS:
        for distance in DISTANCES:
            for condition in ("a_to_c", "c_to_a"):
                for seed in SEEDS:
                    early_delta_lat = np.mean(
                        [
                            lookup[(task, distance, condition, seed, step)][
                                "epsilon_lat"
                            ]
                            - lookup[(task, distance, "none", seed, step)][
                                "epsilon_lat"
                            ]
                            for step in (500_000, 1_000_000, 2_000_000)
                        ]
                    )
                    delta_return_2m = (
                        lookup[(task, distance, condition, seed, 2_000_000)][
                            "heldout_return"
                        ]
                        - lookup[(task, distance, "none", seed, 2_000_000)][
                            "heldout_return"
                        ]
                    )
                    delta_return_6m = (
                        lookup[(task, distance, condition, seed, 6_000_000)][
                            "heldout_return"
                        ]
                        - lookup[(task, distance, "none", seed, 6_000_000)][
                            "heldout_return"
                        ]
                    )
                    points.append(
                        {
                            "task": task,
                            "actor_parameterization": "nps",
                            "align_distance": distance,
                            "condition": condition,
                            "seed": seed,
                            "early_delta_epsilon_lat_mean_5_to_20pct": float(
                                early_delta_lat
                            ),
                            "delta_return_at_20pct": float(delta_return_2m),
                            "delta_return_at_60pct": float(delta_return_6m),
                            "subsequent_return_separation_20_to_60pct": float(
                                delta_return_6m - delta_return_2m
                            ),
                        }
                    )
    grouped = defaultdict(list)
    for row in points:
        grouped[(row["task"], row["align_distance"], row["condition"])].append(row)
    summaries = []
    for key, group in sorted(grouped.items()):
        latent = [row["early_delta_epsilon_lat_mean_5_to_20pct"] for row in group]
        separation = [row["subsequent_return_separation_20_to_60pct"] for row in group]
        summaries.append(
            {
                "task": key[0],
                "actor_parameterization": "nps",
                "align_distance": key[1],
                "condition": key[2],
                **{
                    f"early_delta_epsilon_lat_{name}": value
                    for name, value in summarize(latent, rng, repetitions).items()
                },
                **{
                    f"subsequent_return_separation_{name}": value
                    for name, value in summarize(separation, rng, repetitions).items()
                },
            }
        )
    return points, summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mse-root", type=Path, required=True)
    parser.add_argument("--cka-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fisher-ridge-absolute", type=float, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20260915)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    args = parser.parse_args()
    if args.fisher_ridge_absolute <= 0:
        parser.error("--fisher-ridge-absolute must be positive")
    if args.bootstrap_repetitions < 1:
        parser.error("--bootstrap-repetitions must be positive")

    mse_root = args.mse_root.expanduser().resolve()
    cka_root = args.cka_root.expanduser().resolve()
    output = args.output_root.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "schema_version": 1,
        "analysis_protocol_version": "h1-nps-two-distance-v2.1",
        "training_protocol_version": "h1-v1.0",
        "scope": "NPS only; LN-MSE and Linear CKA are co-primary distance strata",
        "mse_root": str(mse_root),
        "cka_root": str(cka_root),
        "tasks": TASKS,
        "conditions": CONDITIONS,
        "seeds": SEEDS,
        "checkpoint_steps": CHECKPOINT_STEPS,
        "reference_protocol": REFERENCE_PROTOCOL,
        "fisher_ridge_absolute": args.fisher_ridge_absolute,
        "interpretation": (
            "descriptive seed-paired trends; no tolerance-based or CI-based "
            "condition pass/fail"
        ),
        "decision_validity_audit": (
            "Kendall tau and pairwise accuracy are shown against random "
            "ordering expectations of 0 and 0.5"
        ),
        "performance_source": "same 512 held-out stochastic diagnostic rollouts",
        "statistical_unit": "training seed",
        "baseline_reuse": "the same LN-MSE none run is reused for Linear CKA",
    }
    (output / "analysis_protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    mse_rows = load_root(
        mse_root, "ln_mse", set(CONDITIONS), args.fisher_ridge_absolute
    )
    cka_rows = load_root(
        cka_root,
        "linear_cka",
        {"a_to_c", "c_to_a"},
        args.fisher_ridge_absolute,
    )
    rows = validate_and_reuse_baseline(mse_rows, cka_rows)
    rng = np.random.default_rng(args.bootstrap_seed)
    curves = curve_summary(rows, rng, args.bootstrap_repetitions)
    effects = paired_effects(rows, rng, args.bootstrap_repetitions)
    seed_auc, auc_summary = auc_analysis(rows, rng, args.bootstrap_repetitions)
    temporal_points, temporal_summary = temporal_precedence(
        rows, rng, args.bootstrap_repetitions
    )

    write_csv(output / "checkpoint_metrics.csv", rows)
    write_csv(output / "curve_summary.csv", curves)
    write_csv(output / "paired_effects.csv", effects)
    write_csv(
        output / "decision_probe_audit.csv",
        [
            {
                key: row[key]
                for key in (
                    "task",
                    "align_distance",
                    "condition",
                    "seed",
                    "nominal_step",
                    "decision_kendall_tau",
                    "decision_pairwise_accuracy",
                    "decision_top1_agreement",
                    "decision_num_test_anchor_agents",
                )
            }
            for row in rows
        ],
    )
    write_csv(output / "seed_return_auc.csv", seed_auc)
    write_csv(output / "return_auc_paired_summary.csv", auc_summary)
    write_csv(output / "temporal_precedence_points.csv", temporal_points)
    write_csv(output / "temporal_precedence_summary.csv", temporal_summary)
    print(f"Canonical checkpoint records: {len(rows)}")
    print(output)


if __name__ == "__main__":
    main()
