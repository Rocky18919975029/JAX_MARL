"""Redraw the current four-seed Mava evaluation-return curves from JSON logs.

Run repeatedly with ``watch``. The script never changes training files and
handles a metrics.json temporarily being rewritten by Mava's JSON logger.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


EXPECTED_PROTOCOL = "mava-jumanji-rec-mappo-arec-paired-optimal-v1"
CONDITIONS = ("none", "arec")
COLORS = {"none": "#374151", "arec": "#2563EB"}
LINE_STYLES = {"none": "--", "arec": "-"}
ENV_NAMES = {"lbf_15x15-4p-5f": "LevelBasedForaging", "rware_large-8ag": "RobotWarehouse"}
TASK_LABELS = {"lbf_15x15-4p-5f": "LBF · 15×15-4p-5f", "rware_large-8ag": "RWARE · large-8ag"}
ALGORITHM_NAMES = {"none": "rec_mappo", "arec": "rec_mappo_arec"}
OUTPUT_STEM = "mava-paired-return-4seed"


def read_live_json(path: Path, attempts: int = 5) -> dict:
    """Marl-eval rewrites the file in place; tolerate a transient partial read."""
    for attempt in range(attempts):
        try:
            payload = json.loads(path.read_text())
            if not isinstance(payload, dict):
                raise ValueError("metrics root must be a JSON object")
            return payload
        except (OSError, json.JSONDecodeError) as error:
            if attempt == attempts - 1:
                raise RuntimeError(f"Could not read a complete {path}: {error}") from error
            time.sleep(0.05)
    raise AssertionError("unreachable")


def metric_value(value: object) -> float:
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    result = float(value)
    if not np.isfinite(result):
        raise ValueError("Return must be finite")
    return result


def parse_evaluations(payload: dict, task: str, condition: str, seed: int) -> dict[int, float]:
    run = payload[ENV_NAMES[task]][task.split("_", 1)[1]][ALGORITHM_NAMES[condition]][f"seed_{seed}"]
    values = {}
    for key, row in run.items():
        if not key.startswith("step_") or not isinstance(row, dict):
            continue
        if "step_count" not in row or "mean_episode_return" not in row:
            continue
        step = int(row["step_count"])
        if step <= 0 or step in values:
            raise ValueError(f"Nonpositive or duplicate evaluation step in {task}/{condition}/seed{seed}: {step}")
        values[step] = metric_value(row["mean_episode_return"])
    return values


def metric_file(run_root: Path, job: dict) -> Path | None:
    run_dir = run_root / "runs" / job["name"] / "json"
    paths = list(run_dir.glob("**/metrics.json")) if run_dir.is_dir() else []
    if not paths:
        return None
    # A retried run may have multiple timestamped outputs. Never splice
    # evaluations from different attempts into one seed curve.
    return max(paths, key=lambda path: (path.stat().st_mtime_ns, str(path)))


def load_curves(run_root: Path) -> tuple[dict, dict, list[str], dict]:
    manifest_path = run_root / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("protocol") != EXPECTED_PROTOCOL or manifest.get("smoke"):
        raise ValueError("Expected a formal paired-optimal experiment manifest")
    seeds = sorted(manifest["seeds"])
    if len(seeds) != 4 or len(set(seeds)) != 4:
        raise ValueError(f"Expected four distinct training seeds; found {seeds}")
    tasks = manifest["tasks"]
    if any(task not in ENV_NAMES for task in tasks):
        raise ValueError(f"Unsupported tasks in manifest: {tasks}")
    curves = {(task, condition): {} for task in tasks for condition in CONDITIONS}
    warnings = []
    sources = {}
    expected = {(task, condition, seed) for task in tasks for condition in CONDITIONS for seed in seeds}
    found = set()
    for job in manifest["jobs"]:
        task, condition, seed = job["task"], job["condition"], job["seed"]
        key = task, condition, seed
        if key not in expected or key in found:
            raise ValueError(f"Unexpected or duplicated job in manifest: {key}")
        found.add(key)
        path = metric_file(run_root, job)
        if path is None:
            continue
        try:
            curves[(task, condition)][seed] = parse_evaluations(
                read_live_json(path), task, condition, seed
            )
            sources[job["name"]] = str(path)
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            warnings.append(f"{job['name']}: {error}")
    if found != expected:
        raise ValueError(f"Manifest missing jobs: {sorted(expected - found)}")
    return manifest, curves, warnings, sources


def bootstrap_interval(samples: np.ndarray, resamples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Pointwise percentile interval; each resample draws whole seed curves."""
    if samples.ndim != 2 or samples.shape[0] != 4 or resamples < 100:
        raise ValueError("Expected four seed trajectories and at least 100 bootstrap resamples")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, 4, size=(resamples, 4))
    means = samples[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975], axis=0)
    return lower, upper


def aggregate(curves: dict, tasks: list[str], seeds: list[int], resamples: int, seed: int) -> list[dict]:
    rows = []
    for task in tasks:
        for condition in CONDITIONS:
            by_seed = curves[(task, condition)]
            steps = sorted({step for series in by_seed.values() for step in series})
            if not steps:
                continue
            full_steps = [step for step in steps if all(step in by_seed.get(s, {}) for s in seeds)]
            intervals = {}
            if full_steps:
                samples = np.array(
                    [[by_seed[s][step] for step in full_steps] for s in seeds], dtype=float
                )
                low, high = bootstrap_interval(samples, resamples, seed)
                intervals = dict(zip(full_steps, zip(low, high, strict=True), strict=True))
            for step in steps:
                available = [s for s in seeds if step in by_seed.get(s, {})]
                values = [by_seed[s][step] for s in available]
                ci = intervals.get(step)
                rows.append({
                    "task": task,
                    "condition": condition,
                    "env_step": step,
                    "n_seeds": len(available),
                    "seed_ids": ",".join(map(str, available)),
                    "mean_return": float(np.mean(values)),
                    "ci_low": float(ci[0]) if ci is not None else None,
                    "ci_high": float(ci[1]) if ci is not None else None,
                })
    return rows


def render(rows: list[dict], tasks: list[str], output: Path) -> None:
    mm = 1 / 25.4
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.8,
        "savefig.transparent": False,
        "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(1, len(tasks), figsize=(178 * mm, 76 * mm), squeeze=False)
    max_step_millions = max((row["env_step"] for row in rows), default=0) / 1e6
    x_max = max(1.0, max_step_millions * 1.03)
    for ax, task in zip(axes[0], tasks, strict=True):
        ax.set_title(TASK_LABELS[task], pad=8)
        ax.set_xlabel("Environment steps (millions)")
        ax.set_ylabel("Evaluation episode return")
        ax.grid(axis="y", color="#CBD5E1", alpha=0.7, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        task_rows = [row for row in rows if row["task"] == task]
        for condition in CONDITIONS:
            series = [row for row in rows if row["task"] == task and row["condition"] == condition]
            if not series:
                continue
            x = np.array([row["env_step"] / 1e6 for row in series])
            y = np.array([row["mean_return"] for row in series])
            full = np.array([row["n_seeds"] == 4 for row in series])
            color = COLORS[condition]
            style = LINE_STYLES[condition]
            for mask, width, alpha in ((full, 1.35, 1.0), (~full, 1.2, 0.45)):
                ax.plot(x, np.where(mask, y, np.nan), color=color, linestyle=style,
                        linewidth=width, alpha=alpha)
                # A newly started run may have only one evaluation point in
                # the complete or provisional segment; a line cannot show it.
                if np.count_nonzero(mask) == 1:
                    ax.scatter(x[mask], y[mask], s=12, color=color, alpha=alpha, zorder=4)
            if np.any(full):
                lows = np.array([row["ci_low"] if row["ci_low"] is not None else np.nan for row in series])
                highs = np.array([row["ci_high"] if row["ci_high"] is not None else np.nan for row in series])
                ax.fill_between(x, lows, highs, where=full, color=color, alpha=0.17, linewidth=0)
        if task_rows:
            extrema = [value for row in task_rows
                       for value in (row["mean_return"], row["ci_low"], row["ci_high"])
                       if value is not None]
            lower = min(0.0, min(extrema))
            upper = max(0.0, max(extrema))
            padding = max(0.1, (upper - lower) * 0.07)
            ax.set_xlim(0, x_max)
            ax.set_ylim(lower - padding if lower < 0 else 0, upper + padding)
        else:
            ax.set_xlim(0, x_max)
            ax.set_ylim(0, 1)
            ax.text(0.5, 0.5, "Waiting for evaluations", transform=ax.transAxes,
                    ha="center", va="center", color="#64748B")
    handles = [
        Line2D([0], [0], color=COLORS[condition], linestyle=LINE_STYLES[condition],
               linewidth=1.4, label="Original recurrent MAPPO" if condition == "none" else "ARec")
        for condition in CONDITIONS
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.text(0.5, 0.01,
             "Shading: pointwise 95% seed-bootstrap CI (4/4 seeds); faint tails: <4 seeds, no CI. Unsmoothed.",
             ha="center", va="bottom", fontsize=7, color="#5B6472")
    fig.tight_layout(rect=(0, 0.07, 1, 0.9), w_pad=2.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix, fmt, kwargs in (
        (".png", "png", {"dpi": 300}),
        (".svg", "svg", {}),
        (".pdf", "pdf", {}),
    ):
        target = output.with_suffix(suffix)
        temporary = target.with_name(target.name + ".tmp")
        fig.savefig(temporary, format=fmt, **kwargs)
        os.replace(temporary, target)
    plt.close(fig)


def write_csv(rows: list[dict], path: Path) -> None:
    fields = ("task", "condition", "env_step", "n_seeds", "seed_ids", "mean_return", "ci_low", "ci_high")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260924)
    args = parser.parse_args()
    run_root = args.run_root.expanduser().resolve()
    output = (args.output_root or run_root / "plots").expanduser().resolve()
    manifest, curves, warnings, sources = load_curves(run_root)
    rows = aggregate(curves, manifest["tasks"], manifest["seeds"],
                     args.bootstrap_resamples, args.bootstrap_seed)
    output.mkdir(parents=True, exist_ok=True)
    stem = output / OUTPUT_STEM
    render(rows, manifest["tasks"], stem)
    write_csv(rows, stem.with_suffix(".csv"))
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root),
        "manifest": str(run_root / "experiment_manifest.json"),
        "seeds": manifest["seeds"],
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.bootstrap_seed,
        "ci_definition": "pointwise percentile bootstrap of four training-seed trajectories",
        "sources": sources,
        "warnings": warnings,
        "latest": {
            f"{task}/{condition}": (
                {"env_step": matching[-1]["env_step"], "n_seeds": matching[-1]["n_seeds"],
                 "last_full_four_seed_step": max(
                    (row["env_step"] for row in matching if row["n_seeds"] == 4), default=None
                 )}
                if matching else None
            )
            for task in manifest["tasks"] for condition in CONDITIONS
            for matching in [[row for row in rows if row["task"] == task and row["condition"] == condition]]
        },
    }
    metadata_path = stem.with_suffix(".json")
    temporary = metadata_path.with_name(metadata_path.name + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, metadata_path)
    print(f"Updated {stem.with_suffix('.png')}")
    for name, state in summary["latest"].items():
        if state:
            print(f"{name}: latest={state['env_step']:,} n={state['n_seeds']}/4 full4={state['last_full_four_seed_step']}")
        else:
            print(f"{name}: waiting for evaluations")
    for warning in warnings:
        print(f"Temporary/malformed log skipped: {warning}")


if __name__ == "__main__":
    main()
