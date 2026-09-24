"""Report the completed same-four-seed Mava ARec lambda sweep.

Use the immutable sweep manifest to join the original MAPPO and lambda=1e-4
runs with the 32 new runs. No interpolation, smoothing, or seed substitution.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import plot_matched_optimal_returns as paired_plot
import run_arec_lambda_sweep as sweep


STEM = "mava-arec-lambda-sweep-4seed"
BASELINE_COLOR = "#374151"
COEFFICIENT_COLORS = {
    3e-6: "#6A9EF5",
    1e-5: "#4E84E6",
    3e-5: "#2563EB",
    1e-4: "#1D4ED8",
    3e-4: "#1E3A8A",
}
MARKERS = {3e-6: "o", 1e-5: "s", 3e-5: "^", 1e-4: "D", 3e-4: "v"}


def latest_complete_metric(run_dir: Path, expected_final_step: int) -> Path:
    complete = []
    for path in (run_dir / "json").glob("**/metrics.json"):
        try:
            steps = set(sweep.metric_steps(sweep.paired.read_json(path)))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        if expected_final_step in steps:
            complete.append(path)
    if not complete:
        raise RuntimeError(f"No complete evaluation log in {run_dir / 'json'}")
    return max(complete, key=lambda path: (path.stat().st_mtime_ns, str(path)))


def validated_curves(run_root: Path) -> tuple[dict, dict, dict]:
    path = run_root / "experiment_manifest.json"
    manifest = sweep.paired.read_json(path)
    if manifest.get("protocol") != sweep.PROTOCOL:
        raise RuntimeError(f"Wrong sweep protocol in {path}")
    if tuple(manifest["tasks"]) != sweep.TASKS or tuple(manifest["seeds"]) != sweep.SEEDS:
        raise RuntimeError("Task or seed cohort differs from the planned four-seed sweep")
    if tuple(manifest["coefficients"]) != sweep.COEFFICIENTS:
        raise RuntimeError("Coefficient grid differs from the planned sweep")
    if manifest["benchmark_config_sha256"] != sweep.paired.digest(sweep.paired.CONFIG_PATH):
        raise RuntimeError("Benchmark configuration changed since the sweep")
    source_manifest = Path(manifest["source_root"]) / "experiment_manifest.json"
    if sweep.paired.digest(source_manifest) != manifest["source_manifest_sha256"]:
        raise RuntimeError("Original paired manifest changed since the sweep")

    config = sweep.paired.read_json(sweep.paired.CONFIG_PATH)
    expected_jobs = sweep.paired.make_jobs(config, sweep.TASKS, sweep.SEEDS, False)
    expected_reused = {(job["task"], job["condition"], job["seed"]) for job in expected_jobs}
    if len(manifest["reused"]) != len(expected_reused) or {
        (row["task"], row["condition"], row["seed"]) for row in manifest["reused"]
    } != expected_reused:
        raise RuntimeError("Reused baseline/lambda=1e-4 cohort is incomplete or duplicated")

    records = {}
    provenance = {}
    for row in manifest["reused"]:
        metric = Path(row["metric_file"])
        if sweep.paired.digest(metric) != row["metric_sha256"]:
            raise RuntimeError(f"Reused evaluation log changed: {metric}")
        key = (row["task"], row["coefficient"], row["seed"])
        if key in records:
            raise RuntimeError(f"Duplicate reused run: {key}")
        records[key] = paired_plot.parse_evaluations(
            sweep.paired.read_json(metric), row["task"], row["condition"], row["seed"]
        )
        provenance[str(key)] = {"metric_file": str(metric), "sha256": row["metric_sha256"]}

    expected_groups = set(sweep.COEFFICIENTS) - {sweep.REUSED_COEFFICIENT}
    if len(manifest["new_groups"]) != len(expected_groups) or {
        group["coefficient"] for group in manifest["new_groups"]
    } != expected_groups:
        raise RuntimeError("New coefficient groups are incomplete or duplicated")
    for group in manifest["new_groups"]:
        coefficient = group["coefficient"]
        child_root = Path(group["run_root"])
        child_manifest = sweep.paired.read_json(child_root / "experiment_manifest.json")
        if child_manifest.get("jobs") != group["jobs"]:
            raise RuntimeError(f"Child job set changed: {child_root}")
        if child_manifest.get("conditions") != ["arec"] or child_manifest["arec"] != {
            "coef": coefficient, "q_steps": sweep.Q_STEPS,
            "q_lr": sweep.Q_LR, "fisher_ridge": sweep.FISHER_RIDGE,
        }:
            raise RuntimeError(f"Child ARec configuration changed: {child_root}")
        if len(group["jobs"]) != len(sweep.TASKS) * len(sweep.SEEDS):
            raise RuntimeError(f"Child four-seed cohort is incomplete: {child_root}")
        for job in group["jobs"]:
            status = sweep.paired.status_of(child_root, job)
            if status.get("status") != "completed" or status.get("exit_code") != 0:
                raise RuntimeError(f"Sweep run not complete: {coefficient:g} {job['name']}")
            key = (job["task"], coefficient, job["seed"])
            if key in records:
                raise RuntimeError(f"Duplicate run: {key}")
            metric = latest_complete_metric(child_root / "runs" / job["name"], sweep.total_steps(job))
            records[key] = paired_plot.parse_evaluations(
                sweep.paired.read_json(metric), job["task"], "arec", job["seed"]
            )
            provenance[str(key)] = {"metric_file": str(metric), "sha256": sweep.paired.digest(metric)}

    expected = {
        (task, coefficient, seed)
        for task in sweep.TASKS for seed in sweep.SEEDS
        for coefficient in (None, *sweep.COEFFICIENTS)
    }
    if set(records) != expected:
        raise RuntimeError(f"Expected 48 unique curves; missing {sorted(expected - set(records), key=str)}")

    # Both pinned Mava learners evaluate the previous learner_state but log the
    # step after the newly completed update. Correct all conditions identically.
    steps_per_eval = {}
    for task in sweep.TASKS:
        job = next(job for job in expected_jobs if job["task"] == task)
        overrides = job["shared_overrides"]
        if overrides["system.num_updates"] % overrides["arch.num_evaluation"]:
            raise RuntimeError("Unequal update count per evaluation")
        stride = sweep.total_steps(job) // overrides["arch.num_evaluation"]
        logged_steps = tuple(stride * index for index in range(1, overrides["arch.num_evaluation"] + 1))
        steps_per_eval[task] = stride
        for coefficient in (None, *sweep.COEFFICIENTS):
            for seed in sweep.SEEDS:
                key = (task, coefficient, seed)
                series = records[key]
                if tuple(sorted(series)) != logged_steps:
                    raise RuntimeError(f"Missing/mismatched evaluation steps for {key}")
                records[key] = {step - stride: value for step, value in series.items()}
    return manifest, records, {"sources": provenance, "evaluation_lag_steps": steps_per_eval}


def mean_interval(values: np.ndarray, bootstrap_indices: np.ndarray) -> tuple[float, float, float]:
    means = values[bootstrap_indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def normalized_auc(steps: np.ndarray, returns: np.ndarray) -> float:
    if steps.ndim != 1 or steps.size < 2 or steps[-1] <= steps[0]:
        raise ValueError("AUC needs at least two increasing evaluation steps")
    return float(np.sum(np.diff(steps) * (returns[:-1] + returns[1:]) / 2) / (steps[-1] - steps[0]))


def aggregate(records: dict, resamples: int, bootstrap_seed: int) -> tuple[list[dict], list[dict], list[dict]]:
    if resamples < 100:
        raise ValueError("At least 100 seed bootstrap resamples are required")
    rng = np.random.default_rng(bootstrap_seed)
    draw = rng.integers(0, len(sweep.SEEDS), size=(resamples, len(sweep.SEEDS)))
    seed_rows = []
    summary_rows = []
    curve_rows = []
    for task in sweep.TASKS:
        baseline = None
        for coefficient in (None, *sweep.COEFFICIENTS):
            arrays = [records[(task, coefficient, seed)] for seed in sweep.SEEDS]
            steps = np.array(sorted(arrays[0]), dtype=np.int64)
            samples = np.array([[series[int(step)] for step in steps] for series in arrays])
            if baseline is None:
                baseline = samples
            aucs = np.array([normalized_auc(steps, sample) for sample in samples])
            finals = samples[:, -5:].mean(axis=1)
            baseline_aucs = np.array([normalized_auc(steps, sample) for sample in baseline])
            baseline_finals = baseline[:, -5:].mean(axis=1)
            delta_aucs = aucs - baseline_aucs
            delta_finals = finals - baseline_finals
            for index, seed in enumerate(sweep.SEEDS):
                seed_rows.append({
                    "task": task, "condition": "none" if coefficient is None else "arec",
                    "coefficient": "" if coefficient is None else coefficient,
                    "seed": seed, "normalized_auc": aucs[index],
                    "final_five_return": finals[index],
                    "paired_auc_delta": delta_aucs[index],
                    "paired_final_delta": delta_finals[index],
                })
            auc_mean, auc_low, auc_high = mean_interval(aucs, draw)
            final_mean, final_low, final_high = mean_interval(finals, draw)
            delta_auc_mean, delta_auc_low, delta_auc_high = mean_interval(delta_aucs, draw)
            delta_final_mean, delta_final_low, delta_final_high = mean_interval(delta_finals, draw)
            summary_rows.append({
                "task": task, "condition": "none" if coefficient is None else "arec",
                "coefficient": "" if coefficient is None else coefficient,
                "n_seeds": len(sweep.SEEDS),
                "auc_mean": auc_mean, "auc_ci_low": auc_low, "auc_ci_high": auc_high,
                "final_five_mean": final_mean, "final_five_ci_low": final_low,
                "final_five_ci_high": final_high,
                "paired_auc_delta_mean": delta_auc_mean,
                "paired_auc_delta_ci_low": delta_auc_low,
                "paired_auc_delta_ci_high": delta_auc_high,
                "paired_final_delta_mean": delta_final_mean,
                "paired_final_delta_ci_low": delta_final_low,
                "paired_final_delta_ci_high": delta_final_high,
            })
            boot_means = samples[draw].mean(axis=1)
            low, high = np.quantile(boot_means, [0.025, 0.975], axis=0)
            means = samples.mean(axis=0)
            for index, step in enumerate(steps):
                curve_rows.append({
                    "task": task, "condition": "none" if coefficient is None else "arec",
                    "coefficient": "" if coefficient is None else coefficient,
                    "policy_env_step": int(step), "n_seeds": len(sweep.SEEDS),
                    "mean_return": float(means[index]),
                    "ci_low": float(low[index]), "ci_high": float(high[index]),
                })
    for task in sweep.TASKS:
        candidates = [row for row in summary_rows if row["task"] == task and row["condition"] == "arec"]
        best = max(candidates, key=lambda row: (row["paired_auc_delta_mean"], row["paired_final_delta_mean"]))
        for row in summary_rows:
            if row["task"] == task:
                row["selected_by_auc"] = row is best
    return seed_rows, summary_rows, curve_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def render(curve_rows: list[dict], output_stem: Path) -> None:
    mm = 1 / 25.4
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.8, "svg.fonttype": "none", "pdf.fonttype": 42,
        "savefig.transparent": False,
    })
    fig, axes = plt.subplots(1, len(sweep.TASKS), figsize=(178 * mm, 100 * mm), squeeze=False)
    handles = []
    for ax, task in zip(axes[0], sweep.TASKS, strict=True):
        ax.set_title(paired_plot.TASK_LABELS[task], pad=7)
        ax.set_xlabel("Evaluated policy steps (millions)")
        ax.set_ylabel("Evaluation episode return")
        ax.grid(axis="y", color="#CBD5E1", linewidth=0.7, alpha=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        task_rows = [row for row in curve_rows if row["task"] == task]
        for coefficient in (None, *sweep.COEFFICIENTS):
            series = [row for row in task_rows if row["coefficient"] == ("" if coefficient is None else coefficient)]
            x = np.array([row["policy_env_step"] / 1e6 for row in series])
            y = np.array([row["mean_return"] for row in series])
            low = np.array([row["ci_low"] for row in series])
            high = np.array([row["ci_high"] for row in series])
            color = BASELINE_COLOR if coefficient is None else COEFFICIENT_COLORS[coefficient]
            style = "--" if coefficient is None else "-"
            marker = None if coefficient is None else MARKERS[coefficient]
            ax.fill_between(x, low, high, color=color, alpha=0.10 if coefficient is None else 0.07, linewidth=0)
            ax.plot(x, y, color=color, linestyle=style, linewidth=1.35,
                    marker=marker, markersize=2.6, markevery=12 if marker else None)
            if len(handles) < 6:
                label = "Original MAPPO" if coefficient is None else f"ARec λ={coefficient:g}"
                handles.append(Line2D([0], [0], color=color, linestyle=style,
                                      marker=marker, markersize=3.0, linewidth=1.35, label=label))
        all_values = [value for row in task_rows for value in (row["ci_low"], row["ci_high"])]
        upper = max(all_values)
        lower = min(0.0, min(all_values))
        padding = max(0.04, (upper - lower) * 0.06)
        ax.set_ylim(lower - padding if lower < 0 else 0, upper + padding)
        ax.set_xlim(0, 20)
    legend_order = (0, 3, 1, 4, 2, 5)
    fig.legend(handles=[handles[index] for index in legend_order], loc="upper center",
               ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01),
               handlelength=2.5, columnspacing=1.4)
    fig.text(0.5, 0.015,
             "Mean ± pointwise 95% seed-bootstrap CI (4 seeds); unsmoothed. "
             "Policy steps correct Mava's one-evaluation logging lag.",
             ha="center", va="bottom", fontsize=7, color="#4B5563")
    fig.tight_layout(rect=(0, 0.08, 1, 0.88), w_pad=2.0)
    for suffix, fmt, extra in ((".png", "png", {"dpi": 300}), (".svg", "svg", {}), (".pdf", "pdf", {})):
        destination = output_stem.with_suffix(suffix)
        temporary = destination.with_name(destination.name + ".tmp")
        fig.savefig(temporary, format=fmt, **extra)
        os.replace(temporary, destination)
    plt.close(fig)


def report(run_root: Path, output_root: Path, resamples: int, bootstrap_seed: int) -> tuple[Path, Path]:
    manifest, curves, provenance = validated_curves(run_root)
    seed_rows, summary_rows, curve_rows = aggregate(curves, resamples, bootstrap_seed)
    output_root.mkdir(parents=True, exist_ok=True)
    output_stem = output_root / STEM
    render(curve_rows, output_stem)
    write_csv(output_root / "seed_level.csv", seed_rows)
    write_csv(output_root / "summary.csv", summary_rows)
    write_csv(output_root / "pointwise_curves.csv", curve_rows)
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root),
        "manifest_sha256": sweep.paired.digest(run_root / "experiment_manifest.json"),
        "bootstrap_resamples": resamples,
        "bootstrap_seed": bootstrap_seed,
        "primary_selection_metric": "mean paired normalized return AUC over the same four training seeds",
        "selection_note": "Exploratory: hyperparameters selected on the same four seeds used for comparison",
        "auc_definition": "Trapezoidal mean evaluation episode return over evaluated-policy steps",
        "final_definition": "Mean of each seed's last five evaluated checkpoints, then mean over seeds",
        "ci_definition": "Percentile bootstrap resampling four whole training-seed trajectories; paired deltas resample paired seeds",
        "evaluation_lag_steps": provenance["evaluation_lag_steps"],
        "sources": provenance["sources"],
        "mava_commit": manifest["mava_commit"],
    }
    sweep.paired.write_json(output_root / "provenance.json", metadata)
    print(output_stem.with_suffix(".png"))
    print(output_root / "summary.csv")
    for row in summary_rows:
        if row["selected_by_auc"]:
            print(f"{row['task']}: selected λ={row['coefficient']:g} "
                  f"paired ΔAUC={row['paired_auc_delta_mean']:+.6f} "
                  f"paired Δfinal5={row['paired_final_delta_mean']:+.6f}")
    return output_stem.with_suffix(".png"), output_root / "summary.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260925)
    args = parser.parse_args()
    run_root = args.run_root.expanduser().resolve()
    output_root = (args.output_root or run_root / "report").expanduser().resolve()
    report(run_root, output_root, args.bootstrap_resamples, args.bootstrap_seed)


if __name__ == "__main__":
    main()
