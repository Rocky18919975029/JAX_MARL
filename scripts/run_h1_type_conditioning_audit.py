#!/usr/bin/env python3
"""Run and summarize the offline NPS H1 slot-by-type conditioning audit."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

try:
    from h1_type_conditioning_metrics import AUDIT_PROTOCOL
except ModuleNotFoundError:
    from scripts.h1_type_conditioning_metrics import AUDIT_PROTOCOL


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAPS = ("10m_vs_11m", "smacv2_10_units")
MSE_CONDITIONS = ("none", "c_to_a", "a_to_c", "joint")
CKA_CONDITIONS = ("c_to_a_cka", "a_to_c_cka", "joint_cka")
METRICS = (
    "heldout_return_mean",
    "epsilon_lat_slot",
    "epsilon_lat_slot_type",
    "linear_cka_distance_slot",
    "linear_cka_distance_slot_type",
)


@dataclass(frozen=True)
class Task:
    diagnostics_dir: Path
    output_json: Path
    log_path: Path


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path, rows):
    if not rows:
        raise RuntimeError(f"Refusing to write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_csv(value):
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("Expected a non-empty unique CSV list")
    return values


def canonical_condition(condition):
    return condition.removesuffix("_cka")


def discover(root, distance, conditions, maps, seeds, output_root):
    tasks = []
    seen = set()
    for metadata_path in sorted(
        (root / "diagnostics_raw").glob("H1-*/*/metadata.json")
    ):
        metadata = read_json(metadata_path)
        if metadata.get("actor_parameter_sharing"):
            continue
        if metadata.get("align_distance", "ln_mse") != distance:
            continue
        if metadata.get("condition") not in conditions:
            continue
        if metadata.get("map_name") not in maps:
            continue
        if int(metadata.get("training_seed", -1)) not in seeds:
            continue
        key = (
            metadata["map_name"],
            metadata["condition"],
            int(metadata["training_seed"]),
            metadata_path.parent.name,
        )
        if key in seen:
            raise RuntimeError(f"Duplicate diagnostic checkpoint: {key}")
        seen.add(key)
        destination = (
            output_root
            / "checkpoints"
            / metadata["run_name"]
            / metadata_path.parent.name
            / "type_conditioning.json"
        )
        tasks.append(
            Task(
                metadata_path.parent,
                destination,
                output_root
                / "logs"
                / f"{metadata['run_name']}-{metadata_path.parent.name}.log",
            )
        )
    return tasks


def valid_cached(path, ridge, cka_epsilon):
    if not path.is_file():
        return False
    try:
        payload = read_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("audit_protocol") == AUDIT_PROTOCOL
        and float(payload.get("fisher_ridge_absolute", math.nan)) == ridge
        and float(payload.get("linear_cka_epsilon", math.nan)) == cka_epsilon
    )


def run_task(task, ridge, cka_epsilon):
    task.output_json.parent.mkdir(parents=True, exist_ok=True)
    task.log_path.parent.mkdir(parents=True, exist_ok=True)
    command = (
        sys.executable,
        str(REPO_ROOT / "scripts" / "h1_type_conditioning_metrics.py"),
        "--diagnostics-dir",
        str(task.diagnostics_dir),
        "--output-json",
        str(task.output_json),
        "--fisher-ridge-absolute",
        str(ridge),
        "--cka-epsilon",
        str(cka_epsilon),
    )
    environment = os.environ.copy()
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[variable] = "1"
    with task.log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    return result.returncode


def flattened(payload, baseline_source="native"):
    return {
        "run_id": payload.get("run_id"),
        "run_name": payload["run_name"],
        "task": payload["task"],
        "actor_parameterization": "nps",
        "align_distance": payload["align_distance"],
        "condition": canonical_condition(payload["condition"]),
        "seed": payload["seed"],
        "nominal_step": payload["nominal_step"],
        "checkpoint_step": payload["checkpoint_step"],
        "alignment_coef": payload["alignment_coef"],
        "baseline_source": baseline_source,
        "heldout_episodes": payload["heldout_episodes"],
        "heldout_return_mean": payload["heldout_return_mean"],
        "heldout_return_std": payload["heldout_return_std"],
        "epsilon_lat_slot": payload["epsilon_lat_slot"],
        "epsilon_lat_slot_type": payload["epsilon_lat_slot_type"],
        "epsilon_lat_type_minus_slot": payload["epsilon_lat_type_minus_slot"],
        "linear_cka_distance_slot": payload["linear_cka_distance_slot"],
        "linear_cka_distance_slot_type": payload["linear_cka_distance_slot_type"],
        "linear_cka_type_minus_slot": payload["linear_cka_type_minus_slot"],
        "num_observed_unit_types": len(payload["observed_unit_types"]),
        "observed_unit_types": ";".join(map(str, payload["observed_unit_types"])),
        "fisher_ridge_absolute": payload["fisher_ridge_absolute"],
        "audit_protocol": payload["audit_protocol"],
    }


def summarize(values):
    values = [float(value) for value in values]
    mean = sum(values) / len(values)
    variance = (
        sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        if len(values) > 1
        else 0.0
    )
    std = math.sqrt(variance)
    stderr = std / math.sqrt(len(values))
    return {
        "mean": mean,
        "std": std,
        "stderr": stderr,
        "ci95_low": mean - 1.96 * stderr,
        "ci95_high": mean + 1.96 * stderr,
        "n_seeds": len(values),
    }


def validate_single_type(rows):
    selected = [row for row in rows if row["task"] == "10m_vs_11m"]
    if not selected:
        return {"checked": 0, "status": "not_requested"}
    failures = []
    for row in selected:
        if int(row["num_observed_unit_types"]) != 1:
            failures.append(f"{row['run_name']} has multiple unit types")
        for pooled, conditioned in (
            ("epsilon_lat_slot", "epsilon_lat_slot_type"),
            ("linear_cka_distance_slot", "linear_cka_distance_slot_type"),
        ):
            if not math.isclose(
                float(row[pooled]),
                float(row[conditioned]),
                rel_tol=1e-10,
                abs_tol=1e-12,
            ):
                failures.append(
                    f"{row['run_name']} {row['nominal_step']}: "
                    f"{pooled} != {conditioned}"
                )
    if failures:
        raise RuntimeError(
            "10m_vs_11m single-type consistency failed:\n" + "\n".join(failures[:10])
        )
    return {"checked": len(selected), "status": "pass"}


def analyze(tasks, output_root, maps, seeds, ridge, cka_epsilon):
    payloads = [read_json(task.output_json) for task in tasks]
    actual = [flattened(payload) for payload in payloads]
    validation = validate_single_type(actual)

    # The distance-free baseline is trained once under the LN-MSE root.  Reuse
    # the exact same held-out audit row in the Linear CKA stratum.
    reused = []
    has_cka = any(row["align_distance"] == "linear_cka" for row in actual)
    if has_cka:
        for row in actual:
            if row["align_distance"] == "ln_mse" and row["condition"] == "none":
                copy = dict(row)
                copy["align_distance"] = "linear_cka"
                copy["baseline_source"] = "reused_ln_mse_none"
                reused.append(copy)
    analysis_rows = sorted(
        actual + reused,
        key=lambda row: (
            row["task"],
            row["align_distance"],
            row["condition"],
            int(row["seed"]),
            int(row["nominal_step"]),
        ),
    )
    tables = output_root / "tables"
    write_csv(tables / "checkpoint_type_conditioning.csv", analysis_rows)

    grouped = defaultdict(list)
    for row in analysis_rows:
        for metric in METRICS:
            grouped[
                (
                    row["task"],
                    row["align_distance"],
                    row["condition"],
                    int(row["nominal_step"]),
                    metric,
                )
            ].append(row[metric])
    curve_rows = []
    for (task, distance, condition, step, metric), values in sorted(grouped.items()):
        curve_rows.append(
            {
                "task": task,
                "align_distance": distance,
                "condition": condition,
                "nominal_step": step,
                "metric": metric,
                **summarize(values),
            }
        )
    write_csv(tables / "curve_summary.csv", curve_rows)

    lookup = {
        (
            row["task"],
            row["align_distance"],
            row["condition"],
            int(row["seed"]),
            int(row["nominal_step"]),
        ): row
        for row in analysis_rows
    }
    paired_rows = []
    for row in analysis_rows:
        if row["condition"] == "none":
            continue
        baseline = lookup.get(
            (
                row["task"],
                row["align_distance"],
                "none",
                int(row["seed"]),
                int(row["nominal_step"]),
            )
        )
        if baseline is None:
            raise RuntimeError(f"Missing paired baseline for {row['run_name']}")
        paired_rows.append(
            {
                "task": row["task"],
                "align_distance": row["align_distance"],
                "condition": row["condition"],
                "seed": row["seed"],
                "nominal_step": row["nominal_step"],
                **{
                    f"delta_{metric}": float(row[metric]) - float(baseline[metric])
                    for metric in METRICS
                },
            }
        )
    write_csv(tables / "paired_seed_differences.csv", paired_rows)

    paired_grouped = defaultdict(list)
    for row in paired_rows:
        for metric in METRICS:
            paired_grouped[
                (
                    row["task"],
                    row["align_distance"],
                    row["condition"],
                    int(row["nominal_step"]),
                    metric,
                )
            ].append(row[f"delta_{metric}"])
    paired_summary = []
    for key, values in sorted(paired_grouped.items()):
        task, distance, condition, step, metric = key
        paired_summary.append(
            {
                "task": task,
                "align_distance": distance,
                "condition": condition,
                "nominal_step": step,
                "metric": metric,
                **summarize(values),
            }
        )
    write_csv(tables / "paired_curve_summary.csv", paired_summary)

    report = {
        "schema_version": 1,
        "audit_protocol": AUDIT_PROTOCOL,
        "actor_parameterization": "nps",
        "maps": list(maps),
        "seeds": list(seeds),
        "fisher_ridge_absolute": ridge,
        "linear_cka_epsilon": cka_epsilon,
        "native_checkpoint_rows": len(actual),
        "reused_baseline_rows": len(reused),
        "single_type_consistency": validation,
        "tables": {
            "checkpoint": str(tables / "checkpoint_type_conditioning.csv"),
            "curves": str(tables / "curve_summary.csv"),
            "paired_seed": str(tables / "paired_seed_differences.csv"),
            "paired_curves": str(tables / "paired_curve_summary.csv"),
        },
    }
    (output_root / "audit_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mse-root", type=Path, required=True)
    parser.add_argument("--cka-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--maps", type=parse_csv, default=DEFAULT_MAPS)
    parser.add_argument("--seeds", type=parse_csv, default=("1", "2", "3", "4"))
    parser.add_argument("--fisher-ridge-absolute", type=float, required=True)
    parser.add_argument("--cka-epsilon", type=float, default=1e-8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    seeds = tuple(map(int, args.seeds))
    maps = tuple(args.maps)
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    tasks = discover(
        args.mse_root.expanduser().resolve(),
        "ln_mse",
        MSE_CONDITIONS,
        maps,
        seeds,
        output_root,
    ) + discover(
        args.cka_root.expanduser().resolve(),
        "linear_cka",
        CKA_CONDITIONS,
        maps,
        seeds,
        output_root,
    )
    if not tasks:
        raise RuntimeError("No matching NPS diagnostics were discovered")
    pending = [
        task
        for task in tasks
        if args.overwrite
        or not valid_cached(
            task.output_json, args.fisher_ridge_absolute, args.cka_epsilon
        )
    ]
    print(
        f"Discovered={len(tasks)} pending={len(pending)} "
        f"cached={len(tasks) - len(pending)} "
        f"workers={args.workers}",
        flush=True,
    )
    if args.dry_run:
        for task in pending:
            print(task.diagnostics_dir)
        return
    failures = []
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_task,
                task,
                args.fisher_ridge_absolute,
                args.cka_epsilon,
            ): task
            for task in pending
        }
        for future in as_completed(futures):
            task = futures[future]
            status = future.result()
            completed += 1
            if status:
                failures.append((task, status))
            print(
                f"[{completed}/{len(pending)}] status={status} "
                f"{task.diagnostics_dir.parent.name}/{task.diagnostics_dir.name}",
                flush=True,
            )
    if failures:
        preview = "\n".join(
            f"{task.diagnostics_dir}: status={status}, log={task.log_path}"
            for task, status in failures[:10]
        )
        raise RuntimeError(f"{len(failures)} audit workers failed:\n{preview}")
    report = analyze(
        tasks,
        output_root,
        maps,
        seeds,
        args.fisher_ridge_absolute,
        args.cka_epsilon,
    )
    subprocess.run(
        (
            sys.executable,
            str(REPO_ROOT / "scripts" / "plot_h1_type_conditioning_audit.py"),
            "--audit-root",
            str(output_root),
        ),
        cwd=REPO_ROOT,
        check=True,
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
