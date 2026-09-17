#!/usr/bin/env python3
"""Build task-separated, NPS-only RQ1 tables and curves from W&B histories.

The canonical matrix contains two maps, but this script never pools them.  It
writes one independent result directory per map.  A table row is populated
only when all requested training seeds are complete and their W&B histories
are available; otherwise numeric cells are deliberately left empty.  PS runs
are intentionally outside the RQ1 estimand and are neither loaded nor shown.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TASKS = ("10m_vs_11m", "3s5z_vs_3s6z")
ACTOR_VARIANTS = ("nps",)
DISTANCES = ("ln_mse", "linear_cka")
MODES = ("none", "c_to_a", "a_to_c", "joint")
ALIGNED_MODES = MODES[1:]
COLORS = {
    "none": "#222222",
    "c_to_a": "#377bd1",
    "a_to_c": "#80b918",
    "joint": "#bc6c35",
}
LABELS = {
    "none": "Isolated",
    "c_to_a": "C → A",
    "a_to_c": "A → C",
    "joint": "Joint",
}
DISTANCE_LABELS = {"ln_mse": "LN-MSE", "linear_cka": "Linear CKA"}
METRICS = ("returns", "win_rate")


@dataclass(frozen=True)
class Cell:
    task: str
    actor_variant: str
    distance: str
    mode: str
    seed: int

    @property
    def key(self):
        distance = "distance_free" if self.mode == "none" else self.distance
        return self.task, self.actor_variant, distance, self.mode, self.seed

    @property
    def method_key(self):
        return self.actor_variant, self.distance, self.mode


@dataclass(frozen=True)
class RunSource:
    checkpoint: Path
    project: str
    run_id: str
    run_name: str
    alignment_coef: float
    source: str


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        if not rows:
            raise RuntimeError(f"Need field names for empty CSV: {path}")
        fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def expected_cells(tasks, seeds):
    cells = []
    for task in tasks:
        for actor_variant in ACTOR_VARIANTS:
            for seed in seeds:
                cells.append(Cell(task, actor_variant, "distance_free", "none", seed))
                for distance in DISTANCES:
                    for mode in ALIGNED_MODES:
                        cells.append(Cell(task, actor_variant, distance, mode, seed))
    return cells


def metadata_key(metadata, config):
    task = metadata.get("map_name", config.get("MAP_NAME"))
    if task not in TASKS:
        return None
    sharing = metadata.get(
        "actor_parameter_sharing", config.get("ACTOR_PARAMETER_SHARING")
    )
    if isinstance(sharing, str):
        normalized = sharing.strip().lower()
        if normalized not in {"true", "false"}:
            return None
        sharing = normalized == "true"
    actor_variant = "ps" if bool(sharing) else "nps"
    mode = metadata.get("align_mode", config.get("ALIGN_MODE"))
    if mode not in MODES:
        return None
    distance = metadata.get("align_distance", config.get("ALIGN_DISTANCE", "ln_mse"))
    if mode == "none":
        distance = "distance_free"
    if distance not in (*DISTANCES, "distance_free"):
        return None
    seed = metadata.get("seed", config.get("SEED"))
    if seed is None:
        return None
    return task, actor_variant, distance, mode, int(seed)


def source_from_checkpoint(checkpoint: Path, source_label: str):
    metadata_path = checkpoint / "metadata.json"
    config_path = checkpoint / "config.json"
    model_path = checkpoint / "model.safetensors"
    if not (metadata_path.is_file() and config_path.is_file() and model_path.is_file()):
        return None, None
    metadata = read_json(metadata_path)
    config = read_json(config_path)
    key = metadata_key(metadata, config)
    project = metadata.get("wandb_project")
    run_id = metadata.get("wandb_run_id")
    run_name = metadata.get("wandb_run_name")
    coefficient = metadata.get("alignment_coef", config.get("ALIGNMENT_COEF"))
    if key is None or not project or not run_id or not run_name or coefficient is None:
        return None, None
    return key, RunSource(
        checkpoint=checkpoint.resolve(),
        project=str(project),
        run_id=str(run_id),
        run_name=str(run_name),
        alignment_coef=float(coefficient),
        source=source_label,
    )


def discover_sources(matrix_root: Path):
    """Resolve the exact reused or newly trained final checkpoint per cell."""
    sources = {}
    reuse_path = matrix_root / "reused_runs.json"
    if reuse_path.is_file():
        for serialized_key, raw_checkpoint in read_json(reuse_path).get("runs", {}).items():
            expected_key = tuple(serialized_key.split("|"))
            expected_key = (*expected_key[:4], int(expected_key[4]))
            key, source = source_from_checkpoint(Path(raw_checkpoint), "reused")
            if key != expected_key:
                raise RuntimeError(
                    f"Reuse key mismatch: manifest={expected_key}, checkpoint={key}"
                )
            sources[key] = source
    for metadata_path in sorted(
        (matrix_root / "checkpoints").rglob("final/metadata.json")
    ):
        key, source = source_from_checkpoint(metadata_path.parent, "matrix_root")
        if key is not None:
            sources[key] = source
    return sources


def read_history_cache(path: Path):
    rows = []
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            rows.append(
                {
                    "env_step": int(float(row["env_step"])),
                    "returns": float(row["returns"]),
                    "win_rate": float(row["win_rate"]),
                }
            )
    return rows


def write_history_cache(path: Path, rows):
    write_csv(path, rows, ("env_step", "returns", "win_rate"))


def fetch_wandb_history(source, cache_root, api, entity=None, refresh=False):
    cache = cache_root / source.project / f"{source.run_id}.csv"
    if cache.is_file() and not refresh:
        return read_history_cache(cache), "cache"
    paths = []
    if entity:
        paths.append(f"{entity}/{source.project}/{source.run_id}")
    paths.append(f"{source.project}/{source.run_id}")
    errors = []
    run = None
    for path in paths:
        try:
            run = api.run(path)
            break
        except Exception as error:  # W&B supplies several API exception types.
            errors.append(f"{path}: {error}")
    if run is None:
        raise RuntimeError("; ".join(errors))
    by_step = {}
    for row in run.scan_history(keys=["env_step", "returns", "win_rate"], page_size=1000):
        try:
            step = int(float(row["env_step"]))
            returns = float(row["returns"])
            win_rate = float(row["win_rate"])
        except (KeyError, TypeError, ValueError):
            continue
        if all(math.isfinite(value) for value in (returns, win_rate)):
            by_step[step] = {
                "env_step": step,
                "returns": returns,
                "win_rate": win_rate,
            }
    rows = [by_step[step] for step in sorted(by_step)]
    if not rows:
        raise RuntimeError(f"No finite return/win-rate history in {source.run_name}")
    write_history_cache(cache, rows)
    return rows, "wandb"


def interquartile_mean(values):
    """Return the 25%-trimmed mean used as the per-task IQM summary."""
    values = np.sort(np.asarray(values, dtype=np.float64))
    trim = int(math.floor(0.25 * len(values)))
    retained = values[trim : len(values) - trim] if trim else values
    return float(retained.mean())


def exact_bootstrap_ci(values, statistic=np.mean):
    """Exact ordinary-bootstrap percentile CI for the four-seed protocol."""
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return None, None, None
    if len(values) == 1:
        value = float(values[0])
        return value, value, value
    if len(values) <= 6:
        bootstrap_statistics = np.fromiter(
            (
                statistic(values[np.asarray(indices, dtype=np.int64)])
                for indices in itertools.product(range(len(values)), repeat=len(values))
            ),
            dtype=np.float64,
        )
    else:
        rng = np.random.default_rng(20260917)
        indices = rng.integers(0, len(values), size=(10_000, len(values)))
        bootstrap_statistics = np.asarray(
            [statistic(sample) for sample in values[indices]], dtype=np.float64
        )
    low, high = np.quantile(bootstrap_statistics, (0.025, 0.975))
    return float(statistic(values)), float(low), float(high)


def exact_bootstrap_mean_ci(values):
    return exact_bootstrap_ci(values, np.mean)


def exact_bootstrap_iqm_ci(values):
    return exact_bootstrap_ci(values, interquartile_mean)


def normalized_auc(history, metric):
    x = np.asarray([row["env_step"] for row in history], dtype=np.float64)
    y = np.asarray([row[metric] for row in history], dtype=np.float64)
    if len(x) < 2 or x[-1] <= x[0]:
        return None
    area = np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) * 0.5)
    return float(area / (x[-1] - x[0]))


def first_crossing(history, threshold):
    points = [(row["env_step"], row["returns"]) for row in history]
    for index, (step, value) in enumerate(points):
        if value < threshold:
            continue
        if index == 0:
            return float(step)
        previous_step, previous_value = points[index - 1]
        if value == previous_value:
            return float(step)
        fraction = (threshold - previous_value) / (value - previous_value)
        fraction = min(1.0, max(0.0, fraction))
        return float(previous_step + fraction * (step - previous_step))
    return None


def common_curve(histories, metric):
    mappings = [
        {int(row["env_step"]): float(row[metric]) for row in history}
        for history in histories
    ]
    common_steps = sorted(set.intersection(*(set(mapping) for mapping in mappings)))
    rows = []
    for step in common_steps:
        values = [mapping[step] for mapping in mappings]
        mean, low, high = exact_bootstrap_mean_ci(values)
        rows.append(
            {
                "env_step": step,
                "mean": mean,
                "ci95_low": low,
                "ci95_high": high,
                "n_seeds": len(values),
            }
        )
    return rows


def method_specs():
    yield "distance_free", "none"
    for distance in DISTANCES:
        for mode in ALIGNED_MODES:
            yield distance, mode


def method_label(distance, mode):
    if mode == "none":
        return "Isolated"
    return f"{DISTANCE_LABELS[distance]} {LABELS[mode]}"


def build_task_results(task, seeds, sources, histories):
    seed_rows = []
    history_rows = []
    for actor_variant in ACTOR_VARIANTS:
        for distance, mode in method_specs():
            for seed in seeds:
                cell = Cell(task, actor_variant, distance, mode, seed)
                source = sources.get(cell.key)
                history = histories.get(cell.key)
                status = "complete" if source is not None and history else "missing"
                seed_row = {
                    "task": task,
                    "actor_parameterization": actor_variant,
                    "method": method_label(distance, mode),
                    "align_distance": distance,
                    "align_mode": mode,
                    "seed": seed,
                    "status": status,
                    "wandb_project": source.project if source else "",
                    "wandb_run_id": source.run_id if source else "",
                    "wandb_run_name": source.run_name if source else "",
                    "checkpoint": str(source.checkpoint) if source else "",
                    "alignment_coef": source.alignment_coef if source else "",
                    "final_return": history[-1]["returns"] if history else "",
                    "final_win_rate": history[-1]["win_rate"] if history else "",
                    "return_auc": normalized_auc(history, "returns") if history else "",
                    "win_rate_auc": normalized_auc(history, "win_rate") if history else "",
                    "steps_to_threshold": "",
                    "relative_steps_to_threshold": "",
                }
                seed_rows.append(seed_row)
                if history:
                    for point in history:
                        history_rows.append(
                            {
                                "task": task,
                                "actor_parameterization": actor_variant,
                                "align_distance": distance,
                                "align_mode": mode,
                                "seed": seed,
                                **point,
                            }
                        )

    seed_lookup = {
        (
            row["actor_parameterization"],
            row["align_distance"],
            row["align_mode"],
            row["seed"],
        ): row
        for row in seed_rows
    }
    thresholds = {}
    for actor_variant in ACTOR_VARIANTS:
        baseline_histories = [
            histories.get(Cell(task, actor_variant, "distance_free", "none", seed).key)
            for seed in seeds
        ]
        if not all(baseline_histories):
            thresholds[actor_variant] = None
            continue
        curve = common_curve(baseline_histories, "returns")
        if not curve:
            thresholds[actor_variant] = None
            continue
        thresholds[actor_variant] = curve[0]["mean"] + 0.8 * (
            curve[-1]["mean"] - curve[0]["mean"]
        )

    for actor_variant in ACTOR_VARIANTS:
        threshold = thresholds[actor_variant]
        if threshold is None:
            continue
        baseline_steps = {}
        for seed in seeds:
            history = histories.get(
                Cell(task, actor_variant, "distance_free", "none", seed).key
            )
            baseline_steps[seed] = first_crossing(history, threshold) if history else None
        for distance, mode in method_specs():
            for seed in seeds:
                row = seed_lookup[(actor_variant, distance, mode, seed)]
                history = histories.get(Cell(task, actor_variant, distance, mode, seed).key)
                step = first_crossing(history, threshold) if history else None
                if step is not None:
                    row["steps_to_threshold"] = step
                baseline_step = baseline_steps[seed]
                if step is not None and baseline_step not in (None, 0.0):
                    row["relative_steps_to_threshold"] = step / baseline_step

    table_rows = []
    curve_rows = []
    for actor_variant in ACTOR_VARIANTS:
        baseline_by_seed = {
            seed: seed_lookup[(actor_variant, "distance_free", "none", seed)]
            for seed in seeds
        }
        baseline_complete = all(
            item["status"] == "complete" for item in baseline_by_seed.values()
        )
        for distance, mode in method_specs():
            selected = [
                seed_lookup[(actor_variant, distance, mode, seed)] for seed in seeds
            ]
            complete = [row for row in selected if row["status"] == "complete"]
            missing = [str(row["seed"]) for row in selected if row["status"] != "complete"]
            row = {
                "task": task,
                "actor_parameterization": actor_variant,
                "method": method_label(distance, mode),
                "align_distance": distance,
                "align_mode": mode,
                "expected_seeds": ";".join(map(str, seeds)),
                "n_complete_seeds": len(complete),
                "missing_seeds": ";".join(missing),
                "data_status": "complete" if not missing else "incomplete",
                "threshold_return": thresholds[actor_variant]
                if thresholds[actor_variant] is not None
                else "",
            }
            summary_metrics = (
                "final_return",
                "final_win_rate",
                "return_auc",
                "win_rate_auc",
                "steps_to_threshold",
                "relative_steps_to_threshold",
            )
            for metric in summary_metrics:
                row[f"{metric}_mean"] = ""
                row[f"{metric}_ci95_low"] = ""
                row[f"{metric}_ci95_high"] = ""
            for metric in ("final_return_iqm", "final_win_rate_iqm"):
                row[f"{metric}"] = ""
                row[f"{metric}_ci95_low"] = ""
                row[f"{metric}_ci95_high"] = ""
            for metric in ("delta_final_return", "delta_final_win_rate"):
                row[f"{metric}_mean"] = ""
                row[f"{metric}_ci95_low"] = ""
                row[f"{metric}_ci95_high"] = ""
            if not missing:
                for metric in summary_metrics:
                    values = [item[metric] for item in selected]
                    if all(value != "" for value in values):
                        mean, low, high = exact_bootstrap_mean_ci(values)
                        row[f"{metric}_mean"] = mean
                        row[f"{metric}_ci95_low"] = low
                        row[f"{metric}_ci95_high"] = high
                for output_metric, source_metric in (
                    ("final_return_iqm", "final_return"),
                    ("final_win_rate_iqm", "final_win_rate"),
                ):
                    value, low, high = exact_bootstrap_iqm_ci(
                        [item[source_metric] for item in selected]
                    )
                    row[output_metric] = value
                    row[f"{output_metric}_ci95_low"] = low
                    row[f"{output_metric}_ci95_high"] = high
                if baseline_complete:
                    paired_return = [
                        item["final_return"]
                        - baseline_by_seed[item["seed"]]["final_return"]
                        for item in selected
                    ]
                    paired_win = [
                        item["final_win_rate"]
                        - baseline_by_seed[item["seed"]]["final_win_rate"]
                        for item in selected
                    ]
                    for metric, values in (
                        ("delta_final_return", paired_return),
                        ("delta_final_win_rate", paired_win),
                    ):
                        mean, low, high = exact_bootstrap_mean_ci(values)
                        row[f"{metric}_mean"] = mean
                        row[f"{metric}_ci95_low"] = low
                        row[f"{metric}_ci95_high"] = high
                method_histories = [
                    histories[Cell(task, actor_variant, distance, mode, seed).key]
                    for seed in seeds
                ]
                for metric in METRICS:
                    for point in common_curve(method_histories, metric):
                        curve_rows.append(
                            {
                                "task": task,
                                "actor_parameterization": actor_variant,
                                "align_distance": distance,
                                "align_mode": mode,
                                "method": method_label(distance, mode),
                                "metric": metric,
                                **point,
                            }
                        )
            table_rows.append(row)
    return seed_rows, history_rows, table_rows, curve_rows, thresholds


def plot_metric(task, metric, table_rows, curve_rows, output):
    figure, axes = plt.subplots(
        len(ACTOR_VARIANTS),
        len(DISTANCES),
        figsize=(13.2, 5.0),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    table_lookup = {
        (row["actor_parameterization"], row["align_distance"], row["align_mode"]): row
        for row in table_rows
    }
    for row_index, actor_variant in enumerate(ACTOR_VARIANTS):
        for column_index, distance in enumerate(DISTANCES):
            axis = axes[row_index, column_index]
            missing_labels = []
            for mode in MODES:
                method_distance = "distance_free" if mode == "none" else distance
                status = table_lookup[(actor_variant, method_distance, mode)]["data_status"]
                selected = sorted(
                    (
                        row
                        for row in curve_rows
                        if row["actor_parameterization"] == actor_variant
                        and row["align_distance"] == method_distance
                        and row["align_mode"] == mode
                        and row["metric"] == metric
                    ),
                    key=lambda row: int(row["env_step"]),
                )
                if status != "complete" or not selected:
                    missing_labels.append(LABELS[mode])
                    continue
                x = np.asarray([row["env_step"] for row in selected])
                mean = np.asarray([row["mean"] for row in selected])
                low = np.asarray([row["ci95_low"] for row in selected])
                high = np.asarray([row["ci95_high"] for row in selected])
                axis.plot(x, mean, color=COLORS[mode], linewidth=2.2, label=LABELS[mode])
                axis.fill_between(x, low, high, color=COLORS[mode], alpha=0.15)
            axis.set_title(DISTANCE_LABELS[distance])
            axis.set_xlabel("Environment steps")
            axis.set_ylabel("Episode return" if metric == "returns" else "Win rate")
            axis.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
            axis.grid(alpha=0.25)
            if metric == "win_rate":
                axis.set_ylim(-0.02, 1.02)
            if missing_labels:
                axis.text(
                    0.02,
                    0.04,
                    "Missing: " + ", ".join(missing_labels),
                    transform=axis.transAxes,
                    fontsize=8.5,
                    color="#777777",
                    va="bottom",
                )
    handles = [
        plt.Line2D([0], [0], color=COLORS[mode], linewidth=2.2, label=LABELS[mode])
        for mode in MODES
    ]
    figure.suptitle(
        f"RQ1 — {task} — NPS — "
        f"{'episode return' if metric == 'returns' else 'win rate'}",
        fontsize=15,
        y=0.985,
    )
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=4,
        frameon=True,
    )
    figure.tight_layout(rect=(0.02, 0.03, 0.98, 0.80), w_pad=2.0)
    stem = output / "figures" / f"rq1-{task}-{metric}"
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(stem.with_suffix(".png"), dpi=250, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return stem


def parse_csv_ints(value):
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result or len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("Seeds must be a non-empty unique CSV")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--seeds", type=parse_csv_ints, default=(1, 2, 3, 4))
    parser.add_argument("--wandb-entity")
    parser.add_argument("--refresh-wandb", action="store_true")
    args = parser.parse_args()

    tasks = tuple(item.strip() for item in args.tasks.split(",") if item.strip())
    if not tasks or not set(tasks).issubset(TASKS):
        parser.error(f"--tasks must be selected from {TASKS}")
    matrix_root = args.matrix_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    sources = discover_sources(matrix_root)
    expected = expected_cells(tasks, args.seeds)

    histories = {}
    issues = []
    required_sources = {cell.key: sources.get(cell.key) for cell in expected}
    available_sources = {key: source for key, source in required_sources.items() if source}
    api = None
    wandb_entity = args.wandb_entity or os.environ.get("WANDB_ENTITY")
    if available_sources:
        try:
            import wandb

            api = wandb.Api(timeout=60)
            if not wandb_entity:
                wandb_entity = api.default_entity
        except Exception as error:
            raise RuntimeError(f"Cannot initialize W&B API: {error}") from error
    cache_root = output_root / "wandb_history_cache"
    for key, source in sorted(available_sources.items()):
        try:
            history, history_source = fetch_wandb_history(
                source,
                cache_root,
                api,
                entity=wandb_entity,
                refresh=args.refresh_wandb,
            )
            histories[key] = history
            print(
                f"HISTORY {history_source:5s} {source.project}/{source.run_id} "
                f"points={len(history)}",
                flush=True,
            )
        except Exception as error:
            issues.append(
                {
                    "task": key[0],
                    "actor_parameterization": key[1],
                    "align_distance": key[2],
                    "align_mode": key[3],
                    "seed": key[4],
                    "run_name": source.run_name,
                    "error": str(error),
                }
            )
    issue_fields = (
        "task",
        "actor_parameterization",
        "align_distance",
        "align_mode",
        "seed",
        "run_name",
        "error",
    )
    write_csv(output_root / "wandb_history_issues.csv", issues, issue_fields)

    top_manifest = {
        "schema_version": 1,
        "matrix_root": str(matrix_root),
        "tasks_are_never_pooled": True,
        "actor_parameterization": "nps",
        "ps_runs_excluded": True,
        "tasks": list(tasks),
        "seeds": list(args.seeds),
        "complete_checkpoint_sources": len(available_sources),
        "histories_loaded": len(histories),
        "history_errors": len(issues),
        "wandb_entity": wandb_entity,
        "task_outputs": {},
    }
    for task in tasks:
        task_output = output_root / task
        task_output.mkdir(parents=True, exist_ok=True)
        seed_rows, history_rows, table_rows, curve_rows, thresholds = build_task_results(
            task, args.seeds, sources, histories
        )
        write_csv(task_output / "rq1_seed_results.csv", seed_rows)
        write_csv(
            task_output / "rq1_run_history.csv",
            history_rows,
            (
                "task",
                "actor_parameterization",
                "align_distance",
                "align_mode",
                "seed",
                "env_step",
                "returns",
                "win_rate",
            ),
        )
        write_csv(task_output / "rq1_table.csv", table_rows)
        write_csv(
            task_output / "rq1_learning_curve.csv",
            curve_rows,
            (
                "task",
                "actor_parameterization",
                "align_distance",
                "align_mode",
                "method",
                "metric",
                "env_step",
                "mean",
                "ci95_low",
                "ci95_high",
                "n_seeds",
            ),
        )
        stems = [
            plot_metric(task, metric, table_rows, curve_rows, task_output)
            for metric in METRICS
        ]
        task_manifest = {
            "schema_version": 1,
            "task": task,
            "actor_parameterizations": list(ACTOR_VARIANTS),
            "seeds": list(args.seeds),
            "task_separated_statistics": True,
            "required_complete_seeds_per_method": len(args.seeds),
            "missing_policy": (
                "A method row and curve are numeric only when every requested seed "
                "has a completed checkpoint and W&B history; otherwise fields are blank."
            ),
            "curve_center": "mean across training seeds",
            "curve_interval": "exact ordinary-bootstrap percentile 95% CI",
            "threshold_definition": (
                "Within this task and actor parameterization only: first isolated "
                "mean return + 0.8 * (final isolated mean - first isolated mean)."
            ),
            "thresholds": thresholds,
            "tables": {
                "main": str(task_output / "rq1_table.csv"),
                "seed": str(task_output / "rq1_seed_results.csv"),
                "history": str(task_output / "rq1_run_history.csv"),
                "curve": str(task_output / "rq1_learning_curve.csv"),
            },
            "figures": [str(stem.with_suffix(".png")) for stem in stems],
        }
        (task_output / "manifest.json").write_text(
            json.dumps(task_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        top_manifest["task_outputs"][task] = str(task_output)
        complete_methods = sum(row["data_status"] == "complete" for row in table_rows)
        print(
            f"TASK {task}: complete_methods={complete_methods}/{len(table_rows)} "
            f"output={task_output}",
            flush=True,
        )
    (output_root / "manifest.json").write_text(
        json.dumps(top_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
