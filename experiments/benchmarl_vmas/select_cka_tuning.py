#!/usr/bin/env python3
"""Rank VMAS CKA candidates against existing seed-paired isolated runs."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.benchmarl_vmas.analyze import find_eval_json
from experiments.benchmarl_vmas.protocol import TASKS, parse_csv, parse_seeds
from experiments.benchmarl_vmas.run_cka_tuning import TUNING_PROTOCOL_VERSION


def find_step_mapping(payload) -> dict:
    candidates = []

    def visit(value) -> None:
        if not isinstance(value, dict):
            return
        if any(str(key).startswith("step_") for key in value):
            candidates.append(value)
            return
        for child in value.values():
            visit(child)

    visit(payload)
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one evaluation step mapping, found {len(candidates)}"
        )
    return candidates[0]


def load_curve(status: dict) -> list[tuple[int, float]]:
    output = Path(status["benchmarl_output"])
    payload = json.loads(find_eval_json(output).read_text(encoding="utf-8"))
    step_mapping = find_step_mapping(payload)
    curve = []
    for key, value in step_mapping.items():
        if not str(key).startswith("step_"):
            continue
        returns = [float(item) for item in value["return"]]
        curve.append((int(value["step_count"]), statistics.fmean(returns)))
    curve.sort()
    if not curve:
        raise RuntimeError(f"no evaluation curve in {output}")
    return curve


def curve_metrics(curve: list[tuple[int, float]]) -> tuple[float, float]:
    final = curve[-1][1]
    if len(curve) == 1 or curve[-1][0] == curve[0][0]:
        return final, final
    area = sum(
        (right_x - left_x) * (left_y + right_y) / 2
        for (left_x, left_y), (right_x, right_y) in zip(curve, curve[1:])
    )
    return final, area / (curve[-1][0] - curve[0][0])


def load_statuses(root: Path) -> list[dict]:
    statuses = []
    for path in sorted((root / "status").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") == "completed":
            statuses.append(payload)
    return statuses


def standard_error(values: list[float]) -> float:
    return statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_candidate_rows(
    baseline_statuses: list[dict],
    tuning_statuses: list[dict],
    tasks: tuple[str, ...],
    seeds: tuple[int, ...],
) -> list[dict]:
    baselines = {}
    for status in baseline_statuses:
        key = (status.get("task"), int(status.get("seed", -1)))
        if status.get("condition") == "none" and key[0] in tasks and key[1] in seeds:
            if key in baselines:
                raise RuntimeError(f"duplicate isolated baseline for {key}")
            baselines[key] = curve_metrics(load_curve(status))
    expected_baselines = {(task, seed) for task in tasks for seed in seeds}
    missing = sorted(expected_baselines - set(baselines))
    if missing:
        raise RuntimeError(f"missing completed isolated baselines: {missing}")

    rows = []
    seen = set()
    for status in tuning_statuses:
        if status.get("condition") != "c_to_a_cka":
            continue
        if status.get("experiment_stage") != "cka_tuning":
            continue
        task = status["task"]
        seed = int(status["seed"])
        if task not in tasks or seed not in seeds:
            continue
        multiplier = float(status["cka_multiplier"])
        coefficient = float(status["alignment_coef"])
        key = (task, seed, multiplier)
        if key in seen:
            raise RuntimeError(f"duplicate CKA tuning run for {key}")
        seen.add(key)
        final, auc = curve_metrics(load_curve(status))
        baseline_final, baseline_auc = baselines[(task, seed)]
        rows.append(
            {
                "task": task,
                "seed": seed,
                "cka_multiplier": multiplier,
                "cka_coefficient": coefficient,
                "final_return": final,
                "return_auc": auc,
                "isolated_final_return": baseline_final,
                "isolated_return_auc": baseline_auc,
                "paired_final_difference": final - baseline_final,
                "paired_auc_difference": auc - baseline_auc,
            }
        )
    if not rows:
        raise RuntimeError("no completed CKA tuning runs found")
    return rows


def summarize(rows: list[dict], seeds: tuple[int, ...]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["task"], row["cka_multiplier"], row["cka_coefficient"])].append(
            row
        )
    summary = []
    for (task, multiplier, coefficient), values in sorted(grouped.items()):
        present = {int(row["seed"]) for row in values}
        if present != set(seeds):
            raise RuntimeError(
                f"incomplete tuning cell {(task, multiplier)}: seeds={sorted(present)}"
            )
        auc_differences = [row["paired_auc_difference"] for row in values]
        final_differences = [row["paired_final_difference"] for row in values]
        summary.append(
            {
                "task": task,
                "cka_multiplier": multiplier,
                "cka_coefficient": coefficient,
                "seed_count": len(values),
                "paired_auc_difference_mean": statistics.fmean(auc_differences),
                "paired_auc_difference_se": standard_error(auc_differences),
                "paired_final_difference_mean": statistics.fmean(final_differences),
                "paired_final_difference_se": standard_error(final_differences),
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--tuning-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--tasks", type=parse_csv, default=TASKS)
    parser.add_argument("--seeds", type=parse_seeds, default=(1, 2))
    args = parser.parse_args()
    tasks = tuple(args.tasks)
    invalid_tasks = set(tasks) - set(TASKS)
    if invalid_tasks:
        raise ValueError(f"unsupported tasks: {sorted(invalid_tasks)}")
    baseline_root = args.baseline_root.expanduser().resolve()
    tuning_root = args.tuning_root.expanduser().resolve()
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root
        else tuning_root / "selection"
    )
    manifest = json.loads((tuning_root / "tuning_manifest.json").read_text())
    if manifest.get("protocol_version") != TUNING_PROTOCOL_VERSION:
        raise RuntimeError("incompatible tuning manifest")

    rows = paired_candidate_rows(
        load_statuses(baseline_root),
        load_statuses(tuning_root),
        tasks,
        args.seeds,
    )
    summary = summarize(rows, args.seeds)
    selections = {}
    for task in tasks:
        task_rows = [row for row in rows if row["task"] == task]
        task_summary = [row for row in summary if row["task"] == task]
        if not task_summary:
            raise RuntimeError(f"no complete tuning cells for {task}")
        selected = max(
            task_summary,
            key=lambda row: (
                row["paired_auc_difference_mean"],
                row["paired_final_difference_mean"],
                -row["cka_coefficient"],
            ),
        )
        task_root = output_root / task
        write_csv(task_root / "paired_seed_scores.csv", task_rows)
        write_csv(task_root / "candidate_summary.csv", task_summary)
        selection = {
            "task": task,
            "selection_metric": "mean seed-paired normalized return AUC difference",
            "tie_breaker": "mean seed-paired final-return difference, then smaller coefficient",
            "tuning_seeds": list(args.seeds),
            "selected_cka_multiplier": selected["cka_multiplier"],
            "selected_cka_coefficient": selected["cka_coefficient"],
            "paired_auc_difference_mean": selected["paired_auc_difference_mean"],
            "paired_final_difference_mean": selected["paired_final_difference_mean"],
            "exploratory_return_tuning": True,
            "confirmatory_evidence": False,
        }
        (task_root / "selection.json").write_text(
            json.dumps(selection, indent=2, sort_keys=True) + "\n"
        )
        selections[task] = selection

    selection_manifest = {
        "schema_version": 1,
        "protocol_version": TUNING_PROTOCOL_VERSION,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "baseline_root": str(baseline_root),
        "tuning_root": str(tuning_root),
        "tasks_reported_separately": True,
        "cross_task_selection": False,
        "tuning_seeds": list(args.seeds),
        "selection_uses_return": True,
        "confirmatory_evidence": False,
        "selections": selections,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / "selected_cka.json"
    destination.write_text(
        json.dumps(selection_manifest, indent=2, sort_keys=True) + "\n"
    )
    print(destination)
    for task in tasks:
        selected = selections[task]
        print(
            f"{task}: multiplier={selected['selected_cka_multiplier']:.10g} "
            f"lambda={selected['selected_cka_coefficient']:.10g} "
            f"paired_auc_delta={selected['paired_auc_difference_mean']:.6g} "
            f"paired_final_delta={selected['paired_final_difference_mean']:.6g}"
        )


if __name__ == "__main__":
    main()
