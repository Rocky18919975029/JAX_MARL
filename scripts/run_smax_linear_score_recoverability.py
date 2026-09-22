#!/usr/bin/env python3
"""Fit action-conditioned linear probes using existing SMAX collections only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.smax_linear_score_recoverability import measure_from_collection


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def parse_selection(value: str | None, available: list, cast=str) -> list:
    if value is None:
        return list(available)
    selected = [cast(part.strip()) for part in value.split(",") if part.strip()]
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("Selection must contain unique nonempty items")
    missing = set(selected) - set(available)
    if missing:
        raise ValueError(f"Items are absent from source protocol: {sorted(missing)}")
    return selected


def build_jobs(source_root: Path, tasks: list, conditions: list, seeds: list):
    jobs = []
    for task in tasks:
        for condition in conditions:
            for seed in seeds:
                directory = source_root / "runs" / task / condition / f"seed_{seed}"
                summary_path = directory / "summary.json"
                collected = directory / "collected"
                metadata_path = collected / "metadata.json"
                if not summary_path.is_file() or not metadata_path.is_file():
                    raise FileNotFoundError(
                        f"Source measurement/collection is incomplete: {directory}"
                    )
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                expected = (task, condition, int(seed))
                actual = (
                    summary["task"],
                    summary["condition"],
                    int(summary["training_seed"]),
                )
                if actual != expected:
                    raise RuntimeError(f"Source cell mismatch at {directory}")
                jobs.append(
                    {
                        "task": task,
                        "condition": condition,
                        "seed": int(seed),
                        "source_dir": str(directory),
                        "source_summary_sha256": sha256(summary_path),
                        "collection_metadata_sha256": sha256(metadata_path),
                        "checkpoint": summary["checkpoint"],
                    }
                )
    return jobs


def run_job(job: dict, run_root: str, ridge: float, min_fit_per_action: int | None):
    source = Path(job["source_dir"])
    if sha256(source / "summary.json") != job["source_summary_sha256"]:
        raise RuntimeError(f"Source summary changed: {source}")
    if (
        sha256(source / "collected" / "metadata.json")
        != job["collection_metadata_sha256"]
    ):
        raise RuntimeError(f"Source collection metadata changed: {source}")
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    output = (
        Path(run_root) / "runs" / job["task"] / job["condition"] / f"seed_{job['seed']}"
    )
    result = measure_from_collection(
        source / "collected",
        summary,
        output,
        ridge=ridge,
        min_fit_per_action=min_fit_per_action,
    )
    result["source_summary_sha256"] = job["source_summary_sha256"]
    result["collection_metadata_sha256"] = job["collection_metadata_sha256"]
    atomic_json(output / "summary.json", result)
    return result


def completed_matches(job: dict, run_root: Path, ridge: float, minimum: int | None):
    path = (
        run_root
        / "runs"
        / job["task"]
        / job["condition"]
        / f"seed_{job['seed']}"
        / "summary.json"
    )
    if not path.is_file():
        return False
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("protocol") != "smax-linear-score-recoverability-v1.0":
        raise RuntimeError(f"Unrecognized existing result: {path}")
    expected = {
        "task": job["task"],
        "condition": job["condition"],
        "training_seed": job["seed"],
        "checkpoint": job["checkpoint"],
        "ridge": ridge,
        "source_summary_sha256": job["source_summary_sha256"],
        "collection_metadata_sha256": job["collection_metadata_sha256"],
    }
    if minimum is not None:
        expected["min_fit_per_action"] = minimum
    for key, value in expected.items():
        if result.get(key) != value:
            raise RuntimeError(f"Existing result disagrees on {key}: {path}")
    for name in ("agent_metrics.csv", "action_support.csv"):
        if not (path.parent / name).is_file():
            raise RuntimeError(f"Existing result is incomplete: {path.parent / name}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--min-fit-per-action", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.ridge) or args.ridge <= 0:
        parser.error("--ridge must be finite and positive")
    if args.min_fit_per_action is not None and args.min_fit_per_action <= 0:
        parser.error("--min-fit-per-action must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")

    source_root = args.source_root.expanduser().resolve()
    run_root = args.run_root.expanduser().resolve()
    if source_root == run_root or source_root in run_root.parents:
        parser.error("The linear-probe run root must not be inside the source run root")
    source_protocol_path = source_root / "protocol.json"
    source_protocol = json.loads(source_protocol_path.read_text(encoding="utf-8"))
    if source_protocol["protocol"] != "smax-score-recoverability-v1.0":
        raise RuntimeError("Expected the completed v1 score-recoverability collection")
    tasks = parse_selection(args.tasks, source_protocol["tasks"])
    conditions = parse_selection(None, source_protocol["conditions"])
    seeds = parse_selection(args.seeds, source_protocol["seeds"], int)
    jobs = build_jobs(source_root, tasks, conditions, seeds)
    protocol = {
        "schema_version": 1,
        "protocol": "smax-linear-score-recoverability-v1.0",
        "source_root": str(source_root),
        "source_protocol_sha256": sha256(source_protocol_path),
        "tasks": tasks,
        "conditions": conditions,
        "seeds": seeds,
        "selected_budgets": {
            task: source_protocol["selected_budgets"][task] for task in tasks
        },
        "ridge": args.ridge,
        "min_fit_per_action": (
            args.min_fit_per_action
            if args.min_fit_per_action is not None
            else "critic_latent_dimension_plus_one"
        ),
        "rare_action_policy": "zero_predictor_and_report_fraction",
        "critic_standardization": "fit_mean_and_fit_std_plus_1e-6",
        "fisher_source": "fit_split_only",
        "probe_selection_uses_test": False,
        "jobs": jobs,
    }
    protocol_path = run_root / "protocol.json"
    if protocol_path.is_file():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing != protocol:
            raise RuntimeError(
                f"Run root contains a different frozen protocol: {run_root}"
            )
    elif not args.dry_run:
        atomic_json(protocol_path, protocol)

    pending = [
        job
        for job in jobs
        if not completed_matches(job, run_root, args.ridge, args.min_fit_per_action)
    ]
    print(
        f"linear probe: selected={len(jobs)} completed={len(jobs)-len(pending)} "
        f"pending={len(pending)} workers={args.workers}",
        flush=True,
    )
    for job in pending:
        print(
            f"PENDING {job['task']} {job['condition']} seed={job['seed']}", flush=True
        )
    if args.dry_run:
        return

    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for job in pending:
            label = f"{job['task']}--{job['condition']}--seed{job['seed']}"
            status_path = run_root / "status" / f"{label}.json"
            atomic_json(status_path, {"status": "running", "job": job})
            future = executor.submit(
                run_job, job, str(run_root), args.ridge, args.min_fit_per_action
            )
            futures[future] = (label, job, status_path)
        for future in as_completed(futures):
            label, job, status_path = futures[future]
            try:
                result = future.result()
            except Exception as error:
                atomic_json(
                    status_path,
                    {"status": "failed", "job": job, "error": repr(error)},
                )
                failures.append(label)
                print(f"FAILED {label}: {error!r}", flush=True)
            else:
                atomic_json(
                    status_path,
                    {
                        "status": "completed",
                        "job": job,
                        "epsilon_rec_lin_normalized": result[
                            "epsilon_rec_lin_normalized"
                        ],
                        "fallback_test_fraction": result["fallback_test_fraction"],
                    },
                )
                print(
                    f"DONE {label} error={result['epsilon_rec_lin_normalized']:.5g} "
                    f"fallback={result['fallback_test_fraction']:.2%}",
                    flush=True,
                )
    if failures:
        raise RuntimeError(f"{len(failures)} linear-probe jobs failed: {failures}")
    if not args.skip_analysis:
        from scripts.analyze_smax_linear_score_recoverability import analyze

        analyze(run_root)


if __name__ == "__main__":
    main()
