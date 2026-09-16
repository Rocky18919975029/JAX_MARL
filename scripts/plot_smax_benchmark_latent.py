#!/usr/bin/env python3
"""Plot seed-paired held-out return and legacy latent distortion curves.

This script is intentionally limited to already collected NPS SMAX benchmark
diagnostics.  For each training seed and checkpoint it subtracts the single
distance-free ``none`` run from an aligned run before aggregating over seeds.
The legacy latent metric is read verbatim from ``latent_summary.json``; no
diagnostic is recomputed here.
"""

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


DISTANCES = ("ln_mse", "linear_cka")
CONDITIONS = ("none", "c_to_a", "a_to_c", "joint")
ALIGNED_CONDITIONS = CONDITIONS[1:]
COLORS = {
    "c_to_a": "#377bd1",
    "a_to_c": "#80b918",
    "joint": "#bc6c35",
}
LABELS = {"c_to_a": "C → A", "a_to_c": "A → C", "joint": "Joint"}
DISTANCE_LABELS = {"ln_mse": "LN-MSE", "linear_cka": "Linear CKA"}
METRICS = (
    ("heldout_return", "Held-out stochastic return", "Higher is better"),
    (
        "epsilon_lat",
        r"Legacy latent distortion $\epsilon_{Lat}$",
        "Lower is better",
    ),
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows):
    if not rows:
        raise RuntimeError(f"Refusing to write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def canonical_condition(raw_condition: str):
    condition = raw_condition.removesuffix("_cka")
    return condition if condition in CONDITIONS else None


def nominal_step(metadata, directory: Path):
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
    raise RuntimeError(f"Cannot infer checkpoint step for {directory}")


def discover_rows(run_root: Path, map_name: str, actor_variant: str):
    pattern = f"SMAXB4-{map_name}-{actor_variant}-*"
    rows = []
    incomplete = []
    for metadata_path in sorted(
        (run_root / "diagnostics_raw").glob(f"{pattern}/*/metadata.json")
    ):
        directory = metadata_path.parent
        metadata = read_json(metadata_path)
        condition = canonical_condition(str(metadata.get("condition", "")))
        if condition is None:
            continue
        if metadata.get("map_name") != map_name:
            continue
        sharing = bool(metadata.get("actor_parameter_sharing"))
        actual_variant = "ps" if sharing else "nps"
        if actual_variant != actor_variant:
            continue
        latent_path = directory / "latent_summary.json"
        if not latent_path.is_file():
            incomplete.append(str(directory))
            continue
        latent = read_json(latent_path)
        distance = str(metadata.get("align_distance", "ln_mse"))
        if condition == "none":
            distance = "distance_free"
        if distance not in (*DISTANCES, "distance_free"):
            raise RuntimeError(f"Unknown alignment distance {distance!r} in {directory}")
        step = nominal_step(metadata, directory)
        row = {
            "run_name": metadata["run_name"],
            "task": map_name,
            "actor_parameterization": actor_variant,
            "native_align_distance": distance,
            "condition": condition,
            "seed": int(metadata["training_seed"]),
            "nominal_step": step,
            "heldout_return": float(latent["heldout_episode_return_mean"]),
            "epsilon_lat": float(latent["epsilon_lat"]),
            "reference_protocol": str(latent.get("reference_protocol", "")),
            "fisher_ridge_absolute": float(latent["fisher_ridge_absolute"]),
            "diagnostics_dir": str(directory),
        }
        for metric in ("heldout_return", "epsilon_lat"):
            if not math.isfinite(row[metric]):
                raise RuntimeError(f"Non-finite {metric} in {directory}")
        rows.append(row)
    if incomplete:
        raise RuntimeError(
            f"{len(incomplete)} selected checkpoints have no latent_summary.json; "
            f"first missing: {incomplete[0]}"
        )
    if not rows:
        raise RuntimeError(
            f"No completed diagnostics matched {run_root}/diagnostics_raw/{pattern}"
        )
    return rows


def validate_matrix(rows, requested_seeds):
    baselines = [row for row in rows if row["condition"] == "none"]
    if not baselines:
        raise RuntimeError("The distance-free none baseline is missing")
    steps = tuple(sorted({row["nominal_step"] for row in baselines}))
    seeds = tuple(sorted({row["seed"] for row in baselines}))
    if requested_seeds is not None and seeds != requested_seeds:
        raise RuntimeError(f"Expected baseline seeds {requested_seeds}, found {seeds}")
    if len(steps) < 2:
        raise RuntimeError(f"Need at least two checkpoints, found {steps}")
    baseline_keys = {(row["seed"], row["nominal_step"]) for row in baselines}
    expected = {(seed, step) for seed in seeds for step in steps}
    if baseline_keys != expected or len(baselines) != len(expected):
        raise RuntimeError("The none baseline seed/checkpoint matrix is incomplete")
    for distance in DISTANCES:
        for condition in ALIGNED_CONDITIONS:
            selected = [
                row
                for row in rows
                if row["condition"] == condition
                and row["native_align_distance"] == distance
            ]
            keys = {(row["seed"], row["nominal_step"]) for row in selected}
            if keys != expected or len(selected) != len(expected):
                missing = sorted(expected - keys)[:8]
                raise RuntimeError(
                    f"Incomplete {distance}/{condition} matrix; missing={missing}, "
                    f"records={len(selected)}, expected={len(expected)}"
                )
    ridges = {row["fisher_ridge_absolute"] for row in rows}
    protocols = {row["reference_protocol"] for row in rows}
    if len(ridges) != 1 or len(protocols) != 1:
        raise RuntimeError(
            f"Legacy latent protocol is not uniform: ridges={ridges}, "
            f"reference_protocols={protocols}"
        )
    return seeds, steps


def paired_tables(rows, seeds, steps):
    baseline = {
        (row["seed"], row["nominal_step"]): row
        for row in rows
        if row["condition"] == "none"
    }
    aligned = {
        (
            row["native_align_distance"],
            row["condition"],
            row["seed"],
            row["nominal_step"],
        ): row
        for row in rows
        if row["condition"] != "none"
    }
    seed_rows = []
    for distance in DISTANCES:
        for condition in ALIGNED_CONDITIONS:
            for seed in seeds:
                for step in steps:
                    target = aligned[(distance, condition, seed, step)]
                    control = baseline[(seed, step)]
                    seed_rows.append(
                        {
                            "task": target["task"],
                            "actor_parameterization": target[
                                "actor_parameterization"
                            ],
                            "align_distance": distance,
                            "condition": condition,
                            "baseline": "none",
                            "seed": seed,
                            "nominal_step": step,
                            "delta_heldout_return": target["heldout_return"]
                            - control["heldout_return"],
                            "delta_epsilon_lat": target["epsilon_lat"]
                            - control["epsilon_lat"],
                            "aligned_heldout_return": target["heldout_return"],
                            "baseline_heldout_return": control["heldout_return"],
                            "aligned_epsilon_lat": target["epsilon_lat"],
                            "baseline_epsilon_lat": control["epsilon_lat"],
                            "fisher_ridge_absolute": target[
                                "fisher_ridge_absolute"
                            ],
                            "reference_protocol": target["reference_protocol"],
                        }
                    )
    grouped = defaultdict(list)
    for row in seed_rows:
        for metric in ("heldout_return", "epsilon_lat"):
            grouped[
                (
                    row["task"],
                    row["actor_parameterization"],
                    row["align_distance"],
                    row["condition"],
                    row["nominal_step"],
                    metric,
                )
            ].append(float(row[f"delta_{metric}"]))
    summary_rows = []
    for key, values in sorted(grouped.items()):
        array = np.asarray(values, dtype=np.float64)
        std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
        summary_rows.append(
            {
                "task": key[0],
                "actor_parameterization": key[1],
                "align_distance": key[2],
                "condition": key[3],
                "baseline": "none",
                "nominal_step": key[4],
                "metric": key[5],
                "paired_mean_difference": float(array.mean()),
                "paired_std": std,
                "paired_stderr": std / math.sqrt(len(array)),
                "n_paired_seeds": len(array),
                "seeds": ";".join(map(str, seeds)),
            }
        )
    return seed_rows, summary_rows


def plot_distance(output_dir, map_name, actor_variant, distance, seed_rows, summaries):
    selected_seed_rows = [
        row for row in seed_rows if row["align_distance"] == distance
    ]
    selected_summaries = [
        row for row in summaries if row["align_distance"] == distance
    ]
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), sharex=True)
    for axis, (metric, title, direction) in zip(axes, METRICS):
        axis.axhline(0.0, color="#555555", linestyle="--", linewidth=1)
        for condition in ALIGNED_CONDITIONS:
            condition_rows = sorted(
                (
                    row
                    for row in selected_seed_rows
                    if row["condition"] == condition
                ),
                key=lambda row: (int(row["seed"]), int(row["nominal_step"])),
            )
            for seed in sorted({int(row["seed"]) for row in condition_rows}):
                trajectory = [
                    row for row in condition_rows if int(row["seed"]) == seed
                ]
                axis.plot(
                    [int(row["nominal_step"]) for row in trajectory],
                    [float(row[f"delta_{metric}"]) for row in trajectory],
                    color=COLORS[condition],
                    linewidth=0.9,
                    alpha=0.25,
                )
            mean_rows = sorted(
                (
                    row
                    for row in selected_summaries
                    if row["condition"] == condition and row["metric"] == metric
                ),
                key=lambda row: int(row["nominal_step"]),
            )
            x = np.asarray([int(row["nominal_step"]) for row in mean_rows])
            mean = np.asarray(
                [float(row["paired_mean_difference"]) for row in mean_rows]
            )
            stderr = np.asarray([float(row["paired_stderr"]) for row in mean_rows])
            axis.plot(
                x,
                mean,
                color=COLORS[condition],
                linewidth=2.4,
                label=LABELS[condition],
            )
            axis.fill_between(
                x,
                mean - stderr,
                mean + stderr,
                color=COLORS[condition],
                alpha=0.16,
            )
        axis.set_title(f"{title}: alignment − none")
        axis.set_xlabel("Environment steps")
        axis.set_ylabel(f"Seed-paired difference ({direction})")
        axis.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
        axis.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.suptitle(
        f"SMAX {map_name} — {actor_variant.upper()} — "
        f"{DISTANCE_LABELS[distance]} — seed-paired legacy diagnostics",
        fontsize=14,
        y=0.98,
    )
    figure.legend(
        handles,
        labels,
        title="Alignment condition",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=len(ALIGNED_CONDITIONS),
        frameon=True,
    )
    figure.tight_layout(rect=(0.02, 0.02, 0.98, 0.82), w_pad=2.2)
    stem = output_dir / f"smaxb4-{map_name}-{actor_variant}-{distance}-seed-paired"
    figure.savefig(stem.with_suffix(".png"), dpi=250, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return stem


def parse_seeds(value):
    values = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    if not values or any(seed < 0 for seed in values):
        raise argparse.ArgumentTypeError("Seeds must be non-negative integers")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--map", dest="map_name", required=True)
    parser.add_argument("--actor-variant", choices=("nps",), default="nps")
    parser.add_argument("--seeds", type=parse_seeds, default=(1, 2, 3, 4))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    run_root = args.run_root.expanduser().resolve()
    output = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else run_root / "analysis" / args.map_name / "legacy_latent_seed_paired"
    )
    output.mkdir(parents=True, exist_ok=True)
    rows = discover_rows(run_root, args.map_name, args.actor_variant)
    seeds, steps = validate_matrix(rows, args.seeds)
    seed_rows, summaries = paired_tables(rows, seeds, steps)
    write_csv(output / "checkpoint_metrics.csv", rows)
    write_csv(output / "paired_seed_differences.csv", seed_rows)
    write_csv(output / "paired_curve_summary.csv", summaries)
    stems = [
        plot_distance(
            output,
            args.map_name,
            args.actor_variant,
            distance,
            seed_rows,
            summaries,
        )
        for distance in DISTANCES
    ]
    manifest = {
        "schema_version": 1,
        "map": args.map_name,
        "actor_parameterization": args.actor_variant,
        "seeds": seeds,
        "checkpoint_steps": steps,
        "aggregation": "within-seed alignment-minus-none, then mean-and-stderr",
        "latent_metric": "legacy epsilon_lat read from latent_summary.json",
        "none_baseline": "single distance-free run reused in both distance panels",
        "figures": [str(stem.with_suffix(".png")) for stem in stems],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
