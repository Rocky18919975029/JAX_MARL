#!/usr/bin/env python3
"""Create strictly task-separated VMAS return tables and learning curves."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.benchmarl_vmas.protocol import CONDITIONS, TASKS


LABELS = {
    "none": "Isolated",
    "c_to_a_mse": "C → A (LN-MSE)",
    "c_to_a_cka": "C → A (Linear CKA)",
}
COLORS = {"none": "#333333", "c_to_a_mse": "#D55E00", "c_to_a_cka": "#0072B2"}


def find_eval_json(output: Path) -> Path:
    candidates = [
        path
        for path in output.glob("*.json")
        if path.name not in {"completed.json", "protocol_metadata.json"}
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one BenchMARL evaluation JSON in {output}, got {candidates}"
        )
    return candidates[0]


def load_records(root: Path) -> list[dict]:
    records = []
    for status_path in sorted((root / "status").glob("*.json")):
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") != "completed":
            continue
        task = status["task"]
        condition = status["condition"]
        if task not in TASKS or condition not in CONDITIONS:
            continue
        output = Path(status["benchmarl_output"])
        payload = json.loads(find_eval_json(output).read_text(encoding="utf-8"))
        run_data = next(
            iter(next(iter(next(iter(payload.values())).values())).values())
        )
        run_data = next(iter(run_data.values()))
        for key, step in run_data.items():
            if not key.startswith("step_"):
                continue
            returns = np.asarray(step["return"], dtype=float)
            records.append(
                {
                    "task": task,
                    "condition": condition,
                    "seed": int(status["seed"]),
                    "env_step": int(step["step_count"]),
                    "return_mean": float(returns.mean()),
                    "return_std_episode": (
                        float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
                    ),
                    "evaluation_episodes": len(returns),
                }
            )
    if not records:
        raise RuntimeError(f"no completed VMAS evaluation records under {root}")
    return records


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze_task(task: str, records: list[dict], output_root: Path) -> None:
    task_records = [record for record in records if record["task"] == task]
    task_root = output_root / task
    task_root.mkdir(parents=True, exist_ok=True)
    write_csv(
        task_root / "checkpoint_returns.csv",
        list(task_records[0]),
        sorted(
            task_records,
            key=lambda row: (row["condition"], row["seed"], row["env_step"]),
        ),
    )

    by_cell = defaultdict(list)
    for record in task_records:
        by_cell[(record["condition"], record["env_step"])].append(record["return_mean"])
    curves = []
    for (condition, step), values in sorted(by_cell.items()):
        values = np.asarray(values)
        curves.append(
            {
                "task": task,
                "condition": condition,
                "env_step": step,
                "seed_count": len(values),
                "return_mean": float(values.mean()),
                "return_se": (
                    float(values.std(ddof=1) / math.sqrt(len(values)))
                    if len(values) > 1
                    else 0.0
                ),
            }
        )
    write_csv(task_root / "curve_summary.csv", list(curves[0]), curves)

    endpoints = []
    for condition in CONDITIONS:
        by_seed = defaultdict(list)
        for record in task_records:
            if record["condition"] == condition:
                by_seed[record["seed"]].append(record)
        seed_rows = []
        for seed, seed_records in by_seed.items():
            seed_records.sort(key=lambda row: row["env_step"])
            x = np.asarray([row["env_step"] for row in seed_records], dtype=float)
            y = np.asarray([row["return_mean"] for row in seed_records], dtype=float)
            seed_rows.append((seed, y[-1], float(np.trapezoid(y, x) / x[-1])))
        for metric_index, metric in ((1, "final_return"), (2, "return_auc")):
            values = np.asarray([row[metric_index] for row in seed_rows])
            endpoints.append(
                {
                    "task": task,
                    "condition": condition,
                    "metric": metric,
                    "seed_count": len(values),
                    "mean": float(values.mean()),
                    "se": (
                        float(values.std(ddof=1) / math.sqrt(len(values)))
                        if len(values) > 1
                        else 0.0
                    ),
                }
            )
    write_csv(task_root / "endpoint_summary.csv", list(endpoints[0]), endpoints)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axis = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    for condition in CONDITIONS:
        rows = [row for row in curves if row["condition"] == condition]
        if not rows:
            continue
        x = np.asarray([row["env_step"] for row in rows])
        mean = np.asarray([row["return_mean"] for row in rows])
        se = np.asarray([row["return_se"] for row in rows])
        axis.plot(
            x, mean, label=LABELS[condition], color=COLORS[condition], linewidth=2.4
        )
        axis.fill_between(
            x,
            mean - 1.96 * se,
            mean + 1.96 * se,
            color=COLORS[condition],
            alpha=0.16,
            linewidth=0,
        )
    axis.set_title(task.replace("_", " ").title(), fontsize=14, weight="semibold")
    axis.set_xlabel("Environment steps")
    axis.set_ylabel("Deterministic evaluation return")
    axis.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    axis.legend(frameon=False, ncol=1)
    axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(task_root / "learning_curve.png", dpi=300)
    fig.savefig(task_root / "learning_curve.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    root = args.run_root.expanduser().resolve()
    output = (
        args.output_root.expanduser().resolve()
        if args.output_root
        else root / "analysis"
    )
    records = load_records(root)
    for task in TASKS:
        if any(record["task"] == task for record in records):
            analyze_task(task, records, output)
    manifest = {
        "tasks_reported_separately": True,
        "cross_task_aggregation": False,
        "tasks": sorted({record["task"] for record in records}),
        "output_root": str(output),
    }
    (output / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(output)


if __name__ == "__main__":
    main()
