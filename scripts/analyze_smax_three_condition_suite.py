#!/usr/bin/env python3
"""Export task- and parameterization-separated SMAX three-condition results.

The script recursively discovers completed SMAX checkpoints, identifies every
map/parameterization pair with a complete 3-condition x 4-seed matrix, and
selects its largest complete training budget.  It never pools maps or PS/NPS.

Sample efficiency is the time-normalised return AUC over the selected training
budget.  Final performance is computed within each seed by averaging return at
the last five saved checkpoints, then aggregated across training seeds. All
95% confidence intervals use an ordinary percentile bootstrap over training
seeds. For the four-seed protocol, all 4^4 ordered resamples are enumerated.
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
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, ScalarFormatter


ALIGNMENT_DIRECTIONS = ("c_to_a", "a_to_c")
CONDITIONS_BY_DIRECTION = {
    direction: ("none", f"{direction}_cka", f"{direction}_mse")
    for direction in ALIGNMENT_DIRECTIONS
}
CONDITIONS = CONDITIONS_BY_DIRECTION["c_to_a"]
CONDITION_SPECS = {
    "none": ("none", None),
    "c_to_a_mse": ("c_to_a", "ln_mse"),
    "c_to_a_cka": ("c_to_a", "linear_cka"),
    "a_to_c_mse": ("a_to_c", "ln_mse"),
    "a_to_c_cka": ("a_to_c", "linear_cka"),
}
DISPLAY = {
    "none": "Isolated",
    "c_to_a_cka": "C → A (Linear CKA)",
    "c_to_a_mse": "C → A (LN-MSE)",
    "a_to_c_cka": "A → C (Linear CKA)",
    "a_to_c_mse": "A → C (LN-MSE)",
}
STYLE = {
    "none": ("#333333", (0, (5, 2)), "o"),
    "c_to_a_cka": ("#0072B2", "-", "s"),
    "c_to_a_mse": ("#D55E00", "-.", "^"),
    "a_to_c_cka": ("#0072B2", "-", "s"),
    "a_to_c_mse": ("#D55E00", "-.", "^"),
}
BOOTSTRAP_UNIT = "training_seed"
BOOTSTRAP_CI_METHOD = "exact ordinary-bootstrap percentile 95% CI"
CONFIDENCE_LEVEL = 0.95


@dataclass(frozen=True)
class Source:
    task: str
    actor_parameterization: str
    budget: int
    condition: str
    seed: int
    checkpoint: Path
    project: str
    run_id: str
    run_name: str
    alignment_coef: float
    protocol_version: str

    @property
    def key(self):
        return (
            self.task,
            self.actor_parameterization,
            self.budget,
            self.condition,
            self.seed,
        )


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        if not rows:
            raise RuntimeError(f"Field names are required for empty CSV: {path}")
        fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    return None


def classify_condition(metadata, config):
    mode = metadata.get("align_mode", config.get("ALIGN_MODE"))
    distance = metadata.get("align_distance", config.get("ALIGN_DISTANCE", "ln_mse"))
    for condition, (expected_mode, expected_distance) in CONDITION_SPECS.items():
        if mode == expected_mode and (
            expected_distance is None or distance == expected_distance
        ):
            return condition
    return None


def source_from_final(metadata_path: Path):
    checkpoint = metadata_path.parent
    config_path = checkpoint / "config.json"
    model_path = checkpoint / "model.safetensors"
    if not (config_path.is_file() and model_path.is_file()):
        return None
    metadata = read_json(metadata_path)
    config = read_json(config_path)
    sharing = parse_bool(
        metadata.get("actor_parameter_sharing", config.get("ACTOR_PARAMETER_SHARING"))
    )
    if sharing is None:
        return None
    actor_parameterization = "ps" if sharing else "nps"
    task = metadata.get("map_name", config.get("MAP_NAME"))
    seed = metadata.get("seed", config.get("SEED"))
    raw_budget = metadata.get("total_timesteps", config.get("TOTAL_TIMESTEPS"))
    condition = classify_condition(metadata, config)
    project = metadata.get("wandb_project")
    run_id = metadata.get("wandb_run_id")
    run_name = metadata.get("wandb_run_name")
    coefficient = metadata.get("alignment_coef", config.get("ALIGNMENT_COEF"))
    try:
        budget = int(float(raw_budget))
        seed = int(seed)
        coefficient = float(coefficient)
    except (TypeError, ValueError):
        return None
    if (
        not task
        or budget <= 0
        or condition is None
        or not project
        or not run_id
        or not run_name
        or not math.isfinite(coefficient)
    ):
        return None
    return Source(
        task=str(task),
        actor_parameterization=actor_parameterization,
        budget=budget,
        condition=condition,
        seed=seed,
        checkpoint=checkpoint.resolve(),
        project=str(project),
        run_id=str(run_id),
        run_name=str(run_name),
        alignment_coef=coefficient,
        protocol_version=str(
            metadata.get("protocol_version", config.get("PROTOCOL_VERSION", ""))
        ),
    )


def discover_sources(matrix_root: Path):
    """Return the newest source for each task/variant/budget/condition/seed."""
    sources = {}
    ranks = {}
    resolved_root = matrix_root.resolve()
    for metadata_path in sorted(resolved_root.rglob("final/metadata.json")):
        relative_parts = metadata_path.relative_to(resolved_root).parts
        if "archive" in relative_parts or "checkpoints" not in relative_parts:
            continue
        source = source_from_final(metadata_path)
        if source is None:
            continue
        run_dir = source.checkpoint.parent
        completed = int(
            (run_dir / "completed.json").is_file()
            or (source.checkpoint / "completed.json").is_file()
        )
        rank = (completed, metadata_path.stat().st_mtime_ns, str(metadata_path))
        if rank > ranks.get(source.key, (-1, -1, "")):
            sources[source.key] = source
            ranks[source.key] = rank
    return sources


def choose_complete_cohorts(sources, seeds, conditions=CONDITIONS):
    """Choose the largest complete budget for every task/parameterization."""
    cell_budgets = defaultdict(set)
    for task, actor_parameterization, budget, _, _ in sources:
        cell_budgets[(task, actor_parameterization)].add(budget)
    selected = {}
    audit = []
    for task, actor_parameterization in sorted(cell_budgets):
        complete_budgets = []
        for budget in sorted(cell_budgets[(task, actor_parameterization)]):
            missing = [
                f"{condition}:seed{seed}"
                for condition in conditions
                for seed in seeds
                if (
                    task,
                    actor_parameterization,
                    budget,
                    condition,
                    seed,
                )
                not in sources
            ]
            audit.append(
                {
                    "task": task,
                    "actor_parameterization": actor_parameterization,
                    "training_budget_env_steps": budget,
                    "n_expected_runs": len(conditions) * len(seeds),
                    "n_complete_runs": len(conditions) * len(seeds) - len(missing),
                    "complete": not missing,
                    "selected": False,
                    "missing_runs": ";".join(missing),
                }
            )
            if not missing:
                complete_budgets.append(budget)
        if complete_budgets:
            selected[(task, actor_parameterization)] = max(complete_budgets)
    for row in audit:
        row["selected"] = (
            selected.get((row["task"], row["actor_parameterization"]))
            == row["training_budget_env_steps"]
        )
    return selected, audit


def read_history_cache(path: Path):
    rows = []
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            rows.append(
                {
                    "env_step": int(float(row["env_step"])),
                    "returns": float(row["returns"]),
                    "win_rate": (
                        float(row["win_rate"])
                        if row.get("win_rate") not in (None, "")
                        else math.nan
                    ),
                }
            )
    return rows


def write_history_cache(path: Path, rows):
    write_csv(path, rows, ("env_step", "returns", "win_rate"))


def fetch_wandb_history(source, cache_root, api, entity=None, refresh=False):
    cache = cache_root / source.project / f"{source.run_id}.csv"
    if cache.is_file() and not refresh:
        return read_history_cache(cache), "cache"
    candidates = []
    if entity:
        candidates.append(f"{entity}/{source.project}/{source.run_id}")
    candidates.append(f"{source.project}/{source.run_id}")
    errors = []
    run = None
    for candidate in candidates:
        try:
            run = api.run(candidate)
            break
        except Exception as error:  # W&B exposes multiple API exception types.
            errors.append(f"{candidate}: {error}")
    if run is None:
        raise RuntimeError("; ".join(errors))
    by_step = {}
    for row in run.scan_history(
        keys=["env_step", "returns", "win_rate"], page_size=2000
    ):
        try:
            step = int(float(row["env_step"]))
            returns = float(row["returns"])
        except (KeyError, TypeError, ValueError):
            continue
        try:
            win_rate = float(row["win_rate"])
        except (KeyError, TypeError, ValueError):
            win_rate = math.nan
        if step >= 0 and math.isfinite(returns):
            by_step[step] = {
                "env_step": step,
                "returns": returns,
                "win_rate": win_rate,
            }
    history = [by_step[step] for step in sorted(by_step)]
    if len(history) < 2:
        raise RuntimeError(f"Insufficient return history: {source.run_name}")
    write_history_cache(cache, history)
    return history, "wandb"


def checkpoint_steps(source):
    steps = set()
    for metadata_path in source.checkpoint.parent.glob("*/metadata.json"):
        metadata = read_json(metadata_path)
        if metadata.get("is_initial"):
            continue
        raw_step = metadata.get("nominal_env_step", metadata.get("env_step"))
        try:
            step = int(float(raw_step))
        except (TypeError, ValueError):
            continue
        if step > 0:
            steps.add(min(step, source.budget))
    steps.add(source.budget)
    return sorted(steps)


def interpolate_history(history, metric, steps):
    finite = [row for row in history if math.isfinite(float(row[metric]))]
    if len(finite) < 2:
        return None
    x = np.asarray([row["env_step"] for row in finite], dtype=np.float64)
    y = np.asarray([row[metric] for row in finite], dtype=np.float64)
    return np.interp(np.asarray(steps, dtype=np.float64), x, y)


def trapezoidal_integral(values, grid):
    """Integrate with either the NumPy 2.x or legacy NumPy API."""
    if hasattr(np, "trapezoid"):
        return np.trapezoid(values, grid)
    return np.trapz(values, grid)


def normalized_auc(history, metric, budget):
    finite = [row for row in history if math.isfinite(float(row[metric]))]
    if len(finite) < 2:
        return math.nan
    inner = sorted(
        {int(row["env_step"]) for row in finite if 0 < int(row["env_step"]) < budget}
    )
    grid = [0, *inner, budget]
    values = interpolate_history(finite, metric, grid)
    return float(
        trapezoidal_integral(values, np.asarray(grid, dtype=np.float64)) / budget
    )


def seed_metrics(source, history, final_checkpoint_count):
    saved_steps = checkpoint_steps(source)
    if len(saved_steps) < final_checkpoint_count:
        raise RuntimeError(
            f"{source.run_name} has {len(saved_steps)} saved checkpoints; "
            f"need {final_checkpoint_count}"
        )
    final_steps = saved_steps[-final_checkpoint_count:]
    final_returns = interpolate_history(history, "returns", final_steps)
    final_win_rates = interpolate_history(history, "win_rate", final_steps)
    return {
        "task": source.task,
        "actor_parameterization": source.actor_parameterization,
        "training_budget_env_steps": source.budget,
        "condition": source.condition,
        "method": DISPLAY[source.condition],
        "seed": source.seed,
        "alignment_coef": source.alignment_coef,
        "return_auc": normalized_auc(history, "returns", source.budget),
        "final_return_last5_ckpt": float(np.mean(final_returns)),
        "win_rate_auc": normalized_auc(history, "win_rate", source.budget),
        "final_win_rate_last5_ckpt": (
            float(np.mean(final_win_rates)) if final_win_rates is not None else math.nan
        ),
        "final_checkpoint_count": final_checkpoint_count,
        "final_checkpoint_steps": ";".join(map(str, final_steps)),
        "wandb_project": source.project,
        "wandb_run_id": source.run_id,
        "wandb_run_name": source.run_name,
        "checkpoint": str(source.checkpoint),
        "protocol_version": source.protocol_version,
    }


def exact_bootstrap_mean_ci(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return math.nan, math.nan, math.nan
    if len(values) == 1:
        value = float(values[0])
        return value, value, value
    if len(values) <= 6:
        statistics = np.fromiter(
            (
                values[np.asarray(indices, dtype=np.int64)].mean()
                for indices in itertools.product(range(len(values)), repeat=len(values))
            ),
            dtype=np.float64,
        )
    else:
        rng = np.random.default_rng(20260922)
        indices = rng.integers(0, len(values), size=(10_000, len(values)))
        statistics = values[indices].mean(axis=1)
    low, high = np.quantile(statistics, (0.025, 0.975))
    return float(values.mean()), float(low), float(high)


def bootstrap_resample_count(sample_count):
    """Return the number of ordered resamples used by the exact bootstrap."""

    return int(sample_count) ** int(sample_count)


def aggregate_table(
    task, actor_parameterization, budget, rows, seeds, conditions=CONDITIONS
):
    baseline = {row["seed"]: row for row in rows if row["condition"] == "none"}
    table = []
    metrics = (
        "return_auc",
        "final_return_last5_ckpt",
        "win_rate_auc",
        "final_win_rate_last5_ckpt",
    )
    for condition in conditions:
        selected = sorted(
            (row for row in rows if row["condition"] == condition),
            key=lambda row: row["seed"],
        )
        if [row["seed"] for row in selected] != list(seeds):
            raise RuntimeError(f"Incomplete seed table for {task}/{condition}")
        result = {
            "task": task,
            "actor_parameterization": actor_parameterization,
            "training_budget_env_steps": budget,
            "condition": condition,
            "method": DISPLAY[condition],
            "n_seeds": len(selected),
            "seeds": ";".join(map(str, seeds)),
            "alignment_coef": ";".join(
                f"{value:.10g}"
                for value in sorted({float(row["alignment_coef"]) for row in selected})
            ),
            "final_checkpoint_definition": "per-seed mean over last 5 saved checkpoints",
            "auc_definition": "trapezoidal return integral divided by env-step budget",
            "uncertainty_unit": BOOTSTRAP_UNIT,
            "ci_method": BOOTSTRAP_CI_METHOD,
            "confidence_level": CONFIDENCE_LEVEL,
            "bootstrap_resamples": bootstrap_resample_count(len(selected)),
        }
        for metric in metrics:
            values = np.asarray([row[metric] for row in selected], dtype=np.float64)
            mean, low, high = exact_bootstrap_mean_ci(values)
            finite = values[np.isfinite(values)]
            result[f"{metric}_mean"] = mean
            result[f"{metric}_std"] = (
                float(finite.std(ddof=1)) if len(finite) > 1 else 0.0
            )
            result[f"{metric}_ci95_low"] = low
            result[f"{metric}_ci95_high"] = high
            paired = np.asarray(
                [row[metric] - baseline[row["seed"]][metric] for row in selected],
                dtype=np.float64,
            )
            delta_mean, delta_low, delta_high = exact_bootstrap_mean_ci(paired)
            result[f"delta_{metric}_vs_isolated_mean"] = delta_mean
            result[f"delta_{metric}_vs_isolated_ci95_low"] = delta_low
            result[f"delta_{metric}_vs_isolated_ci95_high"] = delta_high
        table.append(result)
    return table


def aggregate_curves(
    task,
    actor_parameterization,
    budget,
    histories,
    seeds,
    points=201,
    conditions=CONDITIONS,
):
    grid = np.linspace(0, budget, points, dtype=np.float64)
    rows = []
    for condition in conditions:
        values = []
        for seed in seeds:
            key = (task, actor_parameterization, budget, condition, seed)
            values.append(interpolate_history(histories[key], "returns", grid))
        values = np.asarray(values, dtype=np.float64)
        for index, step in enumerate(grid):
            mean, low, high = exact_bootstrap_mean_ci(values[:, index])
            rows.append(
                {
                    "task": task,
                    "actor_parameterization": actor_parameterization,
                    "training_budget_env_steps": budget,
                    "condition": condition,
                    "method": DISPLAY[condition],
                    "env_step": int(round(step)),
                    "mean_return": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "n_seeds": len(seeds),
                    "uncertainty_unit": BOOTSTRAP_UNIT,
                    "ci_method": BOOTSTRAP_CI_METHOD,
                    "confidence_level": CONFIDENCE_LEVEL,
                    "bootstrap_resamples": bootstrap_resample_count(len(seeds)),
                }
            )
    return rows


def plot_task(
    task,
    actor_parameterization,
    budget,
    curve_rows,
    output,
    conditions=CONDITIONS,
    alignment_direction="c_to_a",
):
    style = {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
        "font.size": 10.5,
        "axes.labelsize": 11.5,
        "axes.titlesize": 14,
        "axes.titleweight": "semibold",
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
        "axes.linewidth": 0.9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with plt.rc_context(style):
        figure, axis = plt.subplots(figsize=(7.2, 4.45))
        handles = []
        for condition in conditions:
            color, linestyle, marker = STYLE[condition]
            selected = sorted(
                (row for row in curve_rows if row["condition"] == condition),
                key=lambda row: row["env_step"],
            )
            x = np.asarray([row["env_step"] for row in selected])
            mean = np.asarray([row["mean_return"] for row in selected])
            low = np.asarray([row["ci95_low"] for row in selected])
            high = np.asarray([row["ci95_high"] for row in selected])
            axis.fill_between(x, low, high, color=color, alpha=0.13, linewidth=0)
            axis.plot(
                x,
                mean,
                color=color,
                linestyle=linestyle,
                linewidth=2.5,
                marker=marker,
                markersize=4.3,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.0,
                markevery=25,
            )
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=color,
                    linestyle=linestyle,
                    linewidth=2.5,
                    marker=marker,
                    markersize=5.2,
                    markerfacecolor="white",
                    label=DISPLAY[condition],
                )
            )
        axis.set_title(f"SMAX — {task} — {actor_parameterization.upper()}", pad=11)
        axis.set_xlabel("Environment steps", labelpad=6)
        axis.set_ylabel("Episode return", labelpad=7)
        axis.set_xlim(0, budget)
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((0, 0))
        axis.xaxis.set_major_formatter(formatter)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=4))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=4))
        axis.grid(axis="y", color="#D8D8D8", linewidth=0.7, alpha=0.65)
        axis.grid(axis="x", visible=False)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.legend(
            handles=handles,
            title="Mean and bootstrapped 95% CI",
            loc="best",
            frameon=True,
            fancybox=False,
            framealpha=0.96,
            edgecolor="#D0D0D0",
            handlelength=2.8,
        )
        figure.tight_layout(pad=0.8)
        direction_slug = alignment_direction.replace("_", "-")
        stem = output / (
            f"smax-{task}-{actor_parameterization}-{direction_slug}-"
            "three-condition-learning-curve"
        )
        figure.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight")
        figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(figure)
    return stem


def parse_seeds(value):
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds or len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("Seeds must be a non-empty unique CSV")
    return seeds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", type=parse_seeds, default=(1, 2, 3, 4))
    parser.add_argument("--final-checkpoints", type=int, default=5)
    parser.add_argument(
        "--alignment-direction",
        choices=ALIGNMENT_DIRECTIONS,
        default="c_to_a",
    )
    parser.add_argument("--wandb-entity")
    parser.add_argument("--refresh-wandb", action="store_true")
    args = parser.parse_args()
    if args.final_checkpoints != 5:
        parser.error("This report protocol requires exactly five final checkpoints")

    matrix_root = args.matrix_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    conditions = CONDITIONS_BY_DIRECTION[args.alignment_direction]
    sources = discover_sources(matrix_root)
    selected, cohort_audit = choose_complete_cohorts(
        sources, args.seeds, conditions=conditions
    )
    write_csv(
        output_root / "cohort_audit.csv",
        cohort_audit,
        (
            "task",
            "actor_parameterization",
            "training_budget_env_steps",
            "n_expected_runs",
            "n_complete_runs",
            "complete",
            "selected",
            "missing_runs",
        ),
    )
    if not selected:
        raise RuntimeError(
            "No task/parameterization has a complete 3-condition x 4-seed cohort"
        )

    try:
        import wandb

        api = wandb.Api(timeout=90)
    except Exception as error:
        raise RuntimeError(f"Cannot initialize W&B API: {error}") from error
    entity = args.wandb_entity or os.environ.get("WANDB_ENTITY") or api.default_entity
    cache_root = output_root / "wandb_history_cache"
    histories = {}
    issues = []
    selected_sources = {}
    for (task, actor_parameterization), budget in sorted(selected.items()):
        for condition in conditions:
            for seed in args.seeds:
                key = (task, actor_parameterization, budget, condition, seed)
                source = sources[key]
                selected_sources[key] = source
                try:
                    history, origin = fetch_wandb_history(
                        source,
                        cache_root,
                        api,
                        entity=entity,
                        refresh=args.refresh_wandb,
                    )
                    histories[key] = history
                    print(
                        f"HISTORY {origin:5s} {task} "
                        f"{actor_parameterization.upper()} {condition} seed={seed} "
                        f"points={len(history)}",
                        flush=True,
                    )
                except Exception as error:
                    issues.append(
                        {
                            "task": task,
                            "actor_parameterization": actor_parameterization,
                            "training_budget_env_steps": budget,
                            "condition": condition,
                            "seed": seed,
                            "run_name": source.run_name,
                            "error": str(error),
                        }
                    )
    issue_fields = (
        "task",
        "actor_parameterization",
        "training_budget_env_steps",
        "condition",
        "seed",
        "run_name",
        "error",
    )
    write_csv(output_root / "history_issues.csv", issues, issue_fields)
    if issues:
        raise RuntimeError(
            f"Failed to load {len(issues)} histories; inspect "
            f"{output_root / 'history_issues.csv'}"
        )

    manifest = {
        "schema_version": 2,
        "matrix_root": str(matrix_root),
        "alignment_direction": args.alignment_direction,
        "conditions": list(conditions),
        "seeds": list(args.seeds),
        "tasks_are_never_pooled": True,
        "parameterizations_are_never_pooled": True,
        "cohort_selection": "largest complete budget per task and parameterization",
        "auc_definition": "trapezoidal return integral divided by env-step budget",
        "final_performance_definition": (
            "mean of interpolated W&B return at the last five saved checkpoint steps "
            "within each seed, followed by aggregation across seeds"
        ),
        "curve_center": "mean across four training seeds",
        "uncertainty_unit": BOOTSTRAP_UNIT,
        "confidence_level": CONFIDENCE_LEVEL,
        "bootstrap_method": BOOTSTRAP_CI_METHOD,
        "bootstrap_resamples_per_estimate": bootstrap_resample_count(len(args.seeds)),
        "curve_interval": BOOTSTRAP_CI_METHOD,
        "table_interval": BOOTSTRAP_CI_METHOD,
        "selected_cohorts": {
            f"{task}|{actor_parameterization}": budget
            for (task, actor_parameterization), budget in selected.items()
        },
        "outputs": {},
    }
    for (task, actor_parameterization), budget in sorted(selected.items()):
        task_output = output_root / task / actor_parameterization
        task_output.mkdir(parents=True, exist_ok=True)
        rows = []
        for condition in conditions:
            for seed in args.seeds:
                key = (task, actor_parameterization, budget, condition, seed)
                rows.append(
                    seed_metrics(
                        selected_sources[key],
                        histories[key],
                        args.final_checkpoints,
                    )
                )
        summary = aggregate_table(
            task,
            actor_parameterization,
            budget,
            rows,
            args.seeds,
            conditions=conditions,
        )
        curves = aggregate_curves(
            task,
            actor_parameterization,
            budget,
            histories,
            args.seeds,
            conditions=conditions,
        )
        write_csv(task_output / "seed_metrics.csv", rows)
        write_csv(task_output / "summary.csv", summary)
        write_csv(task_output / "learning_curve.csv", curves)
        stem = plot_task(
            task,
            actor_parameterization,
            budget,
            curves,
            task_output,
            conditions=conditions,
            alignment_direction=args.alignment_direction,
        )
        task_manifest = {
            "schema_version": 2,
            "task": task,
            "actor_parameterization": actor_parameterization,
            "training_budget_env_steps": budget,
            "alignment_direction": args.alignment_direction,
            "conditions": list(conditions),
            "seeds": list(args.seeds),
            "uncertainty_unit": BOOTSTRAP_UNIT,
            "confidence_level": CONFIDENCE_LEVEL,
            "bootstrap_method": BOOTSTRAP_CI_METHOD,
            "bootstrap_resamples_per_estimate": bootstrap_resample_count(
                len(args.seeds)
            ),
            "table": str(task_output / "summary.csv"),
            "seed_metrics": str(task_output / "seed_metrics.csv"),
            "curve": str(task_output / "learning_curve.csv"),
            "figure_png": str(stem.with_suffix(".png")),
            "figure_pdf": str(stem.with_suffix(".pdf")),
        }
        (task_output / "manifest.json").write_text(
            json.dumps(task_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        output_key = f"{task}|{actor_parameterization}"
        manifest["outputs"][output_key] = task_manifest
        print(
            f"TASK {task} {actor_parameterization.upper()}: "
            f"budget={budget:,} output={task_output}",
            flush=True,
        )
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
