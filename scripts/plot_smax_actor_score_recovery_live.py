#!/usr/bin/env python3
"""Render a one-shot, seed-aggregated snapshot of an active ARec sweep.

This is an exploratory live view, not a completed-run comparison. Every curve
uses only seeds with at least two logged points, and only environment steps
present in *all* of those seeds. An n=1 curve has no confidence band.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    from scripts.analyze_smax_actor_score_recovery_sweep import PROTOCOL
    from scripts.plot_smax_actor_score_recovery_sweep import (
        BASELINE_COLOR,
        COEF_COLORS,
        COEF_MARKERS,
        METRICS,
    )
except ModuleNotFoundError:  # Direct execution from scripts/.
    from analyze_smax_actor_score_recovery_sweep import PROTOCOL
    from plot_smax_actor_score_recovery_sweep import (
        BASELINE_COLOR,
        COEF_COLORS,
        COEF_MARKERS,
        METRICS,
    )


def read_available_series(path: Path, metric: str, budget: int) -> dict[int, float]:
    """Read complete JSONL lines only; tolerate a concurrent partial last line."""
    if not path.is_file():
        return {}
    data = path.read_bytes()
    if not data:
        return {}
    lines = data.split(b"\n")
    if lines[-1]:
        lines.pop()  # Writer may be in the middle of the final record.
    points: dict[int, float] = {}
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            step = int(row["env_step"])
            value = float(row[metric])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        if 0 < step <= budget and math.isfinite(value):
            points[step] = value
    return points


def snapshot_curves(
    root: Path,
    *,
    metric: str = "win_rate",
    bootstrap_samples: int = 20_000,
    bootstrap_seed: int = 20260923,
) -> tuple[dict, list[dict], list[dict]]:
    """Return current curve rows and an explicit run/seed coverage audit."""
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}")
    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")
    manifest_path = root / "experiment_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Sweep has not started: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != PROTOCOL:
        raise RuntimeError("Expected an actor-score-recovery sweep root")
    intervention_runs = [
        run for run in manifest["runs"] if run["condition"] == "actor_score_recovery"
    ]
    if len({float(run["q_learning_rate"]) for run in intervention_runs}) > 1:
        raise RuntimeError("Live plot requires one fixed q learning rate")
    if len({float(run["fisher_ridge"]) for run in intervention_runs}) > 1:
        raise RuntimeError("Live plot requires one fixed Fisher ridge")

    expected_seeds = {int(seed) for seed in manifest["seeds"]}
    grouped: dict[tuple, dict[int, dict[int, float]]] = defaultdict(dict)
    coverage: list[dict] = []
    for run in manifest["runs"]:
        task, seed = run["map_name"], int(run["seed"])
        if task not in manifest["maps"] or seed not in expected_seeds:
            raise RuntimeError(
                f"Unexpected task or seed in manifest: {run['run_name']}"
            )
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
        status_path = root / "status" / f"{run['run_name']}.json"
        try:
            status = json.loads(status_path.read_text(encoding="utf-8")).get(
                "status", "unknown"
            )
        except (FileNotFoundError, json.JSONDecodeError):
            status = "pending"
        points = (
            read_available_series(
                root / "metrics" / f"{run['run_name']}.jsonl",
                metric,
                int(run["steps"]),
            )
            if status in {"running", "completed"}
            else {}
        )
        coverage.append(
            {
                "task": task,
                "condition": run["condition"],
                "coef": float(run["coef"]),
                "q_steps": int(run["q_steps"]),
                "seed": seed,
                "run_name": run["run_name"],
                "status": status,
                "finite_logged_points": len(points),
                "latest_env_step": max(points, default=0),
                "included": len(points) >= 2,
            }
        )
        if len(points) >= 2:
            grouped[key][seed] = points

    rng = np.random.default_rng(bootstrap_seed)
    curves: list[dict] = []
    for key, seed_series in sorted(grouped.items()):
        if not seed_series:
            continue
        common_steps = sorted(
            set.intersection(*(set(series) for series in seed_series.values()))
        )
        if len(common_steps) < 2:
            continue
        seeds = sorted(seed_series)
        values = np.asarray(
            [[seed_series[seed][step] for step in common_steps] for seed in seeds],
            dtype=np.float64,
        )
        mean = values.mean(axis=0)
        if len(seeds) >= 2:
            indices = rng.integers(0, len(seeds), size=(bootstrap_samples, len(seeds)))
            low, high = np.quantile(
                values[indices].mean(axis=1), (0.025, 0.975), axis=0
            )
        else:
            low = high = [None] * len(common_steps)
        task, condition, coef, q_steps, q_lr, ridge = key
        for index, step in enumerate(common_steps):
            curves.append(
                {
                    "task": task,
                    "condition": condition,
                    "coef": coef,
                    "q_steps": q_steps,
                    "q_learning_rate": q_lr,
                    "fisher_ridge": ridge,
                    "env_step": step,
                    "n_seeds": len(seeds),
                    "seeds": ",".join(str(seed) for seed in seeds),
                    f"mean_{metric}": float(mean[index]),
                    "ci95_low": None if low[index] is None else float(low[index]),
                    "ci95_high": None if high[index] is None else float(high[index]),
                }
            )
    return manifest, curves, coverage


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_task(
    task: str, manifest: dict, rows: list[dict], output: Path, metric: str
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    task_runs = [run for run in manifest["runs"] if run["map_name"] == task]
    coefs = sorted(
        {float(run["coef"]) for run in task_runs if run["condition"] != "none"}
    )
    q_steps = sorted(
        {int(run["q_steps"]) for run in task_runs if run["condition"] != "none"}
    )
    if len(coefs) > len(COEF_COLORS):
        raise RuntimeError("Too many coefficients for the five-color live figure")
    if not q_steps:
        q_steps = [0]
    colors = dict(zip(coefs, COEF_COLORS))
    markers = dict(zip(coefs, COEF_MARKERS))
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
            "lines.linewidth": 1.2,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(
        1,
        len(q_steps),
        figsize=(178 / 25.4, 85 / 25.4),
        sharey=metric == "win_rate",
        squeeze=False,
    )
    axes = axes[0]
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.22, top=0.72, wspace=0.18)
    handles = [Line2D([0], [0], color=BASELINE_COLOR, linewidth=1.5, label="Isolated")]
    handles.extend(
        Line2D(
            [0],
            [0],
            color=colors[coef],
            marker=markers[coef],
            markersize=3,
            markerfacecolor="white",
            linewidth=1.2,
            label=f"λ={coef:.0e}",
        )
        for coef in coefs
    )
    task_rows = [row for row in rows if row["task"] == task]
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in task_rows:
        grouped[(row["condition"], float(row["coef"]), int(row["q_steps"]))].append(row)

    def draw(ax, points: list[dict], color: str, marker: str | None) -> None:
        points.sort(key=lambda row: int(row["env_step"]))
        x = np.asarray([row["env_step"] / 1e6 for row in points])
        y = np.asarray([row[f"mean_{metric}"] for row in points])
        if points[0]["n_seeds"] >= 2:
            low = np.asarray([row["ci95_low"] for row in points])
            high = np.asarray([row["ci95_high"] for row in points])
            ax.fill_between(x, low, high, color=color, alpha=0.13, linewidth=0)
        ax.plot(
            x,
            y,
            color=color,
            linewidth=1.5 if marker is None else 1.2,
            marker=marker,
            markersize=2.7,
            markerfacecolor="white",
            markevery=max(1, len(x) // 8) if marker else None,
            zorder=3 if marker is None else 2,
        )

    baseline_run = next((run for run in task_runs if run["condition"] == "none"), None)
    baseline = (
        grouped.get(("none", 0.0, int(baseline_run["q_steps"])), [])
        if baseline_run is not None
        else []
    )
    for ax, q in zip(axes, q_steps):
        panel_counts = []
        if baseline:
            draw(ax, baseline.copy(), BASELINE_COLOR, None)
            panel_counts.append(int(baseline[0]["n_seeds"]))
        for coef in coefs:
            points = grouped.get(("actor_score_recovery", coef, q), [])
            if points:
                draw(ax, points.copy(), colors[coef], markers[coef])
                panel_counts.append(int(points[0]["n_seeds"]))
        if not any(ax.lines):
            ax.text(
                0.5,
                0.5,
                "Awaiting logged updates",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="#5B6472",
                fontsize=8,
            )
        n_label = (
            f" · n={min(panel_counts)}"
            if panel_counts and min(panel_counts) == max(panel_counts)
            else f" · n={min(panel_counts)}–{max(panel_counts)}" if panel_counts else ""
        )
        ax.set_title(f"q updates = {q}{n_label}")
        ax.set_xlim(0, int(manifest["budgets"][task]) / 1e6)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.set_xlabel("Environment steps (millions)")
        ax.grid(axis="y", color="#D9DEE5", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(direction="out", width=0.8, length=3)
        if metric == "win_rate":
            ax.set_ylim(0, 1)
    axes[0].set_ylabel(
        "Training win rate" if metric == "win_rate" else "Training return"
    )
    fig.suptitle(f"SMAX · {task.replace('_vs_', ' vs ').replace('_', ' ')}", y=0.97)
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=3,
        frameon=False,
    )
    fig.text(
        0.5,
        0.055,
        "Live snapshot · per-curve seed count in CSV · pointwise 95% bootstrap band only when n ≥ 2",
        ha="center",
        fontsize=7,
        color="#5B6472",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        target = output.with_suffix(f".{suffix}")
        temporary = target.with_name(target.stem + ".tmp" + target.suffix)
        fig.savefig(temporary, dpi=300)
        temporary.replace(target)
    plt.close(fig)


def make_snapshot(
    root: Path,
    *,
    metric: str = "win_rate",
    bootstrap_samples: int = 20_000,
    bootstrap_seed: int = 20260923,
    output_dir: Path | None = None,
) -> list[Path]:
    manifest, rows, coverage = snapshot_curves(
        root,
        metric=metric,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    base = output_dir or root / "analysis" / "live"
    outputs = []
    for task in manifest["maps"]:
        task_dir = base / task
        task_dir.mkdir(parents=True, exist_ok=True)
        stem = f"smax-arec-sweep-live-{metric.replace('_', '-')}"
        task_rows = [row for row in rows if row["task"] == task]
        task_coverage = [row for row in coverage if row["task"] == task]
        _write_csv(
            task_dir / f"{stem}-curves.csv",
            task_rows,
            [
                "task",
                "condition",
                "coef",
                "q_steps",
                "q_learning_rate",
                "fisher_ridge",
                "env_step",
                "n_seeds",
                "seeds",
                f"mean_{metric}",
                "ci95_low",
                "ci95_high",
            ],
        )
        _write_csv(
            task_dir / f"{stem}-coverage.csv",
            task_coverage,
            [
                "task",
                "condition",
                "coef",
                "q_steps",
                "seed",
                "run_name",
                "status",
                "finite_logged_points",
                "latest_env_step",
                "included",
            ],
        )
        (task_dir / f"{stem}-metadata.json").write_text(
            json.dumps(
                {
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "protocol": PROTOCOL,
                    "task": task,
                    "metric": metric,
                    "bootstrap_samples": bootstrap_samples,
                    "bootstrap_seed": bootstrap_seed,
                    "ci": "pointwise percentile; whole-seed resampling; absent for n<2",
                    "alignment": "exact shared env_step within each curve; no interpolation or smoothing",
                    "note": "Exploratory in-progress snapshot; curves can have different seed counts and horizons.",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        output = task_dir / stem
        render_task(task, manifest, task_rows, output, metric)
        outputs.append(output.with_suffix(".png"))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--metric", choices=METRICS, default="win_rate")
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260923)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    for output in make_snapshot(
        args.run_root.expanduser().resolve(),
        metric=args.metric,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        output_dir=args.output_dir.expanduser().resolve() if args.output_dir else None,
    ):
        print(output)


if __name__ == "__main__":
    main()
