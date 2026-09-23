#!/usr/bin/env python3
"""Summarize an exploratory sweep against paired isolated runs, per map."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


PROTOCOL = "smax-nps-actor-score-recovery-sweep-v1.0"


def history(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    by_step = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        step = int(row["env_step"])
        if step <= 0:
            continue
        by_step[step] = row
    if len(by_step) < 2:
        raise RuntimeError(f"Insufficient training history: {path}")
    return [by_step[step] for step in sorted(by_step)]


def metric_summary(rows: list[dict], metric: str, budget: int) -> tuple[float, float]:
    points = []
    for row in rows:
        try:
            value = float(row[metric])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value) and int(row["env_step"]) <= budget:
            points.append((int(row["env_step"]), value))
    if len(points) < 2:
        raise RuntimeError(f"Fewer than two finite {metric} values")
    # Endpoint values are held constant outside the observed training-log span.
    grid = [(0, points[0][1]), *points, (budget, points[-1][1])]
    auc = sum(
        (right_step - left_step) * (left_value + right_value) / 2
        for (left_step, left_value), (right_step, right_value) in zip(
            grid, grid[1:]
        )
    ) / budget
    return auc, statistics.mean(value for _, value in points[-5:])


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(root: Path) -> dict:
    manifest = json.loads((root / "experiment_manifest.json").read_text())
    if manifest.get("protocol") != PROTOCOL:
        raise RuntimeError("Expected an actor-score-recovery sweep root")
    seed_rows = []
    missing = []
    for run in manifest["runs"]:
        name = run["run_name"]
        status_path = root / "status" / f"{name}.json"
        status = (
            json.loads(status_path.read_text()).get("status")
            if status_path.is_file()
            else None
        )
        if status != "completed":
            missing.append(f"{name}: {status or 'pending'}")
            continue
        rows = history(root / "metrics" / f"{name}.jsonl")
        budget = int(run["steps"])
        return_auc, return_last5 = metric_summary(rows, "returns", budget)
        win_auc, win_last5 = metric_summary(rows, "win_rate", budget)
        seed_rows.append(
            {
                "task": run["map_name"],
                "condition": run["condition"],
                "seed": run["seed"],
                "budget": budget,
                "coef": run["coef"],
                "q_steps": run["q_steps"],
                "q_learning_rate": run["q_learning_rate"],
                "fisher_ridge": run["fisher_ridge"],
                "run_name": name,
                "return_auc": return_auc,
                "win_rate_auc": win_auc,
                "return_last5_logged_updates": return_last5,
                "win_rate_last5_logged_updates": win_last5,
            }
        )
    if missing:
        raise RuntimeError(
            f"Sweep is not complete ({len(missing)} runs): {missing[:10]}"
        )

    selection = {}
    for task in manifest["maps"]:
        task_rows = [row for row in seed_rows if row["task"] == task]
        baseline = {
            row["seed"]: row
            for row in task_rows
            if row["condition"] == "none"
        }
        if set(baseline) != set(manifest["seeds"]):
            raise RuntimeError(f"Missing paired isolated baseline for {task}")
        grouped = defaultdict(list)
        for row in task_rows:
            key = (
                row["condition"],
                row["coef"],
                row["q_steps"],
                row["q_learning_rate"],
                row["fisher_ridge"],
            )
            grouped[key].append(row)
        summary_rows = []
        for (condition, coef, q_steps, q_lr, ridge), cell in sorted(grouped.items()):
            if set(row["seed"] for row in cell) != set(manifest["seeds"]):
                raise RuntimeError(f"Incomplete seed group for {task}/{condition}")
            summary_rows.append(
                {
                    "task": task,
                    "condition": condition,
                    "budget": int(cell[0]["budget"]),
                    "n_seeds": len(cell),
                    "coef": coef,
                    "q_steps": q_steps,
                    "q_learning_rate": q_lr,
                    "fisher_ridge": ridge,
                    "mean_win_rate_auc": statistics.mean(
                        row["win_rate_auc"] for row in cell
                    ),
                    "paired_delta_win_rate_auc_vs_none": statistics.mean(
                        row["win_rate_auc"]
                        - baseline[row["seed"]]["win_rate_auc"]
                        for row in cell
                    ),
                    "mean_return_auc": statistics.mean(
                        row["return_auc"] for row in cell
                    ),
                    "paired_delta_return_auc_vs_none": statistics.mean(
                        row["return_auc"] - baseline[row["seed"]]["return_auc"]
                        for row in cell
                    ),
                    "mean_win_rate_last5_logged_updates": statistics.mean(
                        row["win_rate_last5_logged_updates"] for row in cell
                    ),
                    "mean_return_last5_logged_updates": statistics.mean(
                        row["return_last5_logged_updates"] for row in cell
                    ),
                }
            )
        task_output = root / "analysis" / task
        write_csv(task_output / "seed_level.csv", task_rows)
        write_csv(task_output / "task_condition_summary.csv", summary_rows)
        candidates = [
            row for row in summary_rows if row["condition"] == "actor_score_recovery"
        ]
        best = max(
            candidates,
            key=lambda row: (
                row["paired_delta_win_rate_auc_vs_none"],
                row["paired_delta_return_auc_vs_none"],
                -row["coef"],
                -row["q_steps"],
            ),
        )
        selection[task] = {
            "selection_metric": "mean seed-paired training win-rate AUC gain",
            "screening_seeds": list(manifest["seeds"]),
            "training_budget": manifest["budgets"][task],
            "top_candidate": best,
            "promote_to_confirmatory": best[
                "paired_delta_win_rate_auc_vs_none"
            ] > 0,
            "warning": (
                "Exploratory training-curve selection only; evaluate the selected "
                "setting on separate, unused seeds at the full task budget."
            ),
        }
    output = root / "analysis" / "selection.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    return selection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    selection = summarize(args.run_root.expanduser().resolve())
    for task, result in selection.items():
        best = result["top_candidate"]
        print(
            f"{task}: coef={best['coef']:.10g} q_steps={best['q_steps']} "
            f"q_lr={best['q_learning_rate']:.10g} "
            f"ridge={best['fisher_ridge']:.10g} "
            f"paired win-AUC gain={best['paired_delta_win_rate_auc_vs_none']:.6g} "
            f"promote={result['promote_to_confirmatory']}"
        )
    print(args.run_root.expanduser().resolve() / "analysis" / "selection.json")


if __name__ == "__main__":
    main()
