#!/usr/bin/env python3
"""Plot seed-aggregated SMAX actor-score-recovery sweep learning curves."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from scripts.analyze_smax_actor_score_recovery_sweep import (
        PROTOCOL,
        history,
        write_csv,
    )
except ModuleNotFoundError:  # Direct execution from scripts/.
    from analyze_smax_actor_score_recovery_sweep import PROTOCOL, history, write_csv


METRICS = ("win_rate", "returns")
BASELINE_COLOR = "#343A40"
COEF_COLORS = ("#2563EB", "#7C3AED", "#B45309", "#15803D", "#B91C1C")
COEF_MARKERS = ("s", "D", "^", "v", "P")
Q_STYLES = ("-", (0, (5, 2)), (0, (1, 2)))


def _run_series(root: Path, run: dict, metric: str) -> dict[int, float]:
    name = run["run_name"]
    status_path = root / "status" / f"{name}.json"
    status = (
        json.loads(status_path.read_text(encoding="utf-8")).get("status")
        if status_path.is_file()
        else "pending"
    )
    if status != "completed":
        raise RuntimeError(f"Sweep run is not completed: {name} ({status})")
    points = {}
    for row in history(root / "metrics" / f"{name}.jsonl"):
        step = int(row["env_step"])
        if step > int(run["steps"]):
            continue
        try:
            value = float(row[metric])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            points[step] = value
    if len(points) < 2:
        raise RuntimeError(f"Fewer than two finite {metric} points: {name}")
    return points


def seed_bootstrap_curves(
    root: Path,
    *,
    metric: str = "win_rate",
    bootstrap_samples: int = 20_000,
    bootstrap_seed: int = 20260923,
) -> tuple[dict, list[dict]]:
    """Compute pointwise percentile CIs by resampling whole training seeds.

    For each task/setting, only exact environment steps observed for every
    seed are retained. No cross-task pooling, smoothing, interpolation, or
    extrapolation is used. The same bootstrap seed indices are applied to the
    entire trajectory, preserving within-seed temporal dependence.
    """
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}")
    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")
    manifest = json.loads(
        (root / "experiment_manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("protocol") != PROTOCOL:
        raise RuntimeError("Expected an actor-score-recovery sweep root")
    expected_seeds = tuple(int(seed) for seed in manifest["seeds"])
    if len(expected_seeds) < 2 or len(set(expected_seeds)) != len(expected_seeds):
        raise RuntimeError("Bootstrap CI requires at least two distinct seeds")
    intervention_runs = [
        run for run in manifest["runs"] if run["condition"] == "actor_score_recovery"
    ]
    if len({float(run["q_learning_rate"]) for run in intervention_runs}) != 1:
        raise RuntimeError("Plot requires one fixed q learning rate")
    if len({float(run["fisher_ridge"]) for run in intervention_runs}) != 1:
        raise RuntimeError("Plot requires one fixed Fisher ridge")

    grouped: dict[tuple, dict[int, dict[int, float]]] = defaultdict(dict)
    for run in manifest["runs"]:
        task = run["map_name"]
        if task not in manifest["maps"]:
            raise RuntimeError(f"Run has unexpected task: {task}")
        seed = int(run["seed"])
        if seed not in expected_seeds:
            raise RuntimeError(f"Run has unexpected seed: {seed}")
        key = (
            task,
            run["condition"],
            float(run["coef"]),
            int(run["q_steps"]),
            float(run["q_learning_rate"]),
            float(run["fisher_ridge"]),
        )
        if seed in grouped[key]:
            raise RuntimeError(f"Duplicate seed in sweep group: {key} seed={seed}")
        grouped[key][seed] = _run_series(root, run, metric)

    indices = np.random.default_rng(bootstrap_seed).integers(
        0, len(expected_seeds), size=(bootstrap_samples, len(expected_seeds))
    )
    curve_rows = []
    for key, seed_series in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], item[0][1] != "none", item[0][2:]),
    ):
        task, condition, coef, q_steps, q_lr, ridge = key
        if set(seed_series) != set(expected_seeds):
            raise RuntimeError(
                f"Missing seeds for {key}: {set(expected_seeds) - set(seed_series)}"
            )
        common_steps = sorted(set.intersection(*(set(v) for v in seed_series.values())))
        if len(common_steps) < 2:
            raise RuntimeError(f"Fewer than two shared finite steps for {key}")
        values = np.asarray(
            [
                [seed_series[seed][step] for step in common_steps]
                for seed in expected_seeds
            ],
            dtype=np.float64,
        )
        boot_means = values[indices].mean(axis=1)
        low, high = np.quantile(boot_means, (0.025, 0.975), axis=0)
        mean = values.mean(axis=0)
        for index, step in enumerate(common_steps):
            curve_rows.append(
                {
                    "task": task,
                    "condition": condition,
                    "coef": coef,
                    "q_steps": q_steps,
                    "q_learning_rate": q_lr,
                    "fisher_ridge": ridge,
                    "env_step": step,
                    "n_seeds": len(expected_seeds),
                    f"mean_{metric}": float(mean[index]),
                    "ci95_low": float(low[index]),
                    "ci95_high": float(high[index]),
                }
            )
    if not curve_rows:
        raise RuntimeError("No sweep curves to plot")
    return manifest, curve_rows


def render_figure(manifest: dict, rows: list[dict], output: Path, metric: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    maps = manifest["maps"]
    coefs = sorted(
        {
            float(row["coef"])
            for row in rows
            if row["condition"] == "actor_score_recovery"
        }
    )
    q_steps = sorted(
        {
            int(row["q_steps"])
            for row in rows
            if row["condition"] == "actor_score_recovery"
        }
    )
    if len(coefs) > len(COEF_COLORS) or len(q_steps) > len(Q_STYLES):
        raise RuntimeError("Too many sweep settings for a legible single figure")
    colors = dict(zip(coefs, COEF_COLORS))
    markers = dict(zip(coefs, COEF_MARKERS))
    styles = dict(zip(q_steps, Q_STYLES))
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(
        1, len(maps), figsize=(178 / 25.4, 105 / 25.4), sharey=metric == "win_rate"
    )
    if len(maps) == 1:
        axes = [axes]
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.20, top=0.79, wspace=0.13)
    legend = [Line2D([0], [0], color=BASELINE_COLOR, linewidth=1.6, label="Isolated")]
    for coef in coefs:
        for q in q_steps:
            if not any(
                row["condition"] == "actor_score_recovery"
                and float(row["coef"]) == coef
                and int(row["q_steps"]) == q
                for row in rows
            ):
                continue
            legend.append(
                Line2D(
                    [0],
                    [0],
                    color=colors[coef],
                    linestyle=styles[q],
                    linewidth=1.35,
                    marker=markers[coef],
                    markerfacecolor="white",
                    markeredgewidth=0.8,
                    markersize=3.2,
                    label=f"λ={coef:.0e}, q={q}",
                )
            )
    for ax, task in zip(axes, maps):
        task_rows = [row for row in rows if row["task"] == task]
        groups = defaultdict(list)
        for row in task_rows:
            groups[(row["condition"], float(row["coef"]), int(row["q_steps"]))].append(
                row
            )
        for (condition, coef, q), points in sorted(
            groups.items(), key=lambda item: (item[0][0] != "none", item[0][1:])
        ):
            points.sort(key=lambda row: int(row["env_step"]))
            x = np.asarray([row["env_step"] / 1e6 for row in points])
            mean = np.asarray([row[f"mean_{metric}"] for row in points])
            low = np.asarray([row["ci95_low"] for row in points])
            high = np.asarray([row["ci95_high"] for row in points])
            baseline = condition == "none"
            color = BASELINE_COLOR if baseline else colors[coef]
            ax.fill_between(
                x, low, high, color=color, alpha=0.15 if baseline else 0.11, linewidth=0
            )
            ax.plot(
                x,
                mean,
                color=color,
                linestyle="-" if baseline else styles[q],
                linewidth=1.6 if baseline else 1.35,
                marker=None if baseline else markers[coef],
                markerfacecolor="white",
                markeredgewidth=0.8,
                markersize=3.0,
                markevery=None if baseline else max(1, len(x) // 9),
                zorder=3 if baseline else 2,
            )
        ax.set_title(
            f"SMAX · {task.replace('_vs_', ' vs ').replace('_', ' ')}",
            fontweight="bold",
        )
        ax.set_xlim(0, int(manifest["budgets"][task]) / 1e6)
        ax.set_xlabel("Environment steps (millions)")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.3g}"))
        ax.grid(axis="y", color="#D9DEE5", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(direction="out", width=0.8, length=3)
        if metric == "win_rate":
            ax.set_ylim(0, 1)
    axes[0].set_ylabel(
        "Training win rate" if metric == "win_rate" else "Training return"
    )
    if len(legend) == 7 and len(q_steps) == 2 and len(coefs) == 3:
        # Matplotlib fills legend columns first; this preserves row-wise
        # reading as baseline, all q=4, then all q=8.
        legend = [legend[index] for index in (0, 2, 1, 4, 3, 6, 5)]
    fig.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=4,
        frameon=False,
    )
    fig.text(
        0.5,
        0.06,
        f"Mean across {len(manifest['seeds'])} pilot seeds; shading: pointwise 95% seed-bootstrap CI",
        ha="center",
        va="center",
        fontsize=7,
        color="#5B6472",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=300)
    plt.close(fig)


def make_figure(
    root: Path,
    *,
    metric: str = "win_rate",
    bootstrap_samples: int = 20_000,
    bootstrap_seed: int = 20260923,
) -> Path:
    manifest, rows = seed_bootstrap_curves(
        root,
        metric=metric,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    output_dir = root / "analysis"
    stem = f"smax-actor-score-recovery-sweep-{metric.replace('_', '-')}"
    write_csv(output_dir / f"{stem}-curves.csv", rows)
    for task in manifest["maps"]:
        write_csv(
            output_dir / task / f"{stem}-curves.csv",
            [row for row in rows if row["task"] == task],
        )
    output = output_dir / stem
    render_figure(manifest, rows, output, metric)
    (output_dir / f"{stem}-manifest.json").write_text(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "metric": metric,
                "tasks_are_separate_panels_not_pooled": True,
                "aggregation_unit": "training seed",
                "bootstrap_method": "pointwise percentile CI; whole-seed resampling",
                "bootstrap_samples": bootstrap_samples,
                "bootstrap_seed": bootstrap_seed,
                "seeds": manifest["seeds"],
                "finite_step_alignment": "intersection of exact env_step values within each setting",
                "smoothing": "none",
                "interpolation": "none",
                "warning": "Two pilot seeds make bootstrap bounds exploratory, not confirmatory.",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output.with_suffix(".png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--metric", choices=METRICS, default="win_rate")
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260923)
    args = parser.parse_args()
    print(
        make_figure(
            args.run_root.expanduser().resolve(),
            metric=args.metric,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
    )


if __name__ == "__main__":
    main()
