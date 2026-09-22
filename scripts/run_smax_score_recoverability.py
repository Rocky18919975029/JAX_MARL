#!/usr/bin/env python3
"""Run final-checkpoint SMAX score recoverability for three NPS conditions."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_smax_three_condition_suite import (
    choose_complete_cohorts,
    discover_sources,
)
from scripts.h1_diagnostic_data import load_diagnostics
from scripts.smax_score_recoverability import available_samples_by_agent


TASKS = ("10m_vs_11m", "3s5z_vs_3s6z", "smacv2_10_units")
CONDITIONS = ("none", "c_to_a_mse", "c_to_a_cka")
SEEDS = (1, 2, 3, 4)


def task_seed(base: int, task: str) -> int:
    digest = hashlib.sha256(task.encode("utf-8")).digest()
    return base + int.from_bytes(digest[:2], "big")


def atomic_json(path: Path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def parse_items(raw, cast=str):
    return tuple(cast(item.strip()) for item in raw.split(",") if item.strip())


def common_sample_counts(
    jobs,
    *,
    fit_fraction,
    split_seed,
    fit_cap,
    test_cap,
    expected_episodes=None,
    collection_seed_base=None,
):
    """Choose equal per-agent counts for every condition and seed in a task."""

    by_task = {}
    for name, source, output in jobs:
        directory = output / "collected"
        metadata, arrays = load_diagnostics(directory, ("active", "alive"))
        if int(metadata["episodes"]) != len(arrays["active"]):
            raise RuntimeError(f"Incomplete collected episodes for {name}")
        if metadata.get("checkpoint") != str(source.checkpoint):
            raise RuntimeError(
                f"Cached collection belongs to another checkpoint: {name}"
            )
        if (
            expected_episodes is not None
            and int(metadata["episodes"]) != expected_episodes
        ):
            raise RuntimeError(f"Cached collection has the wrong episode count: {name}")
        if collection_seed_base is not None and int(
            metadata["diagnostic_seed"]
        ) != task_seed(collection_seed_base, source.task):
            raise RuntimeError(f"Cached collection has the wrong rollout seed: {name}")
        if metadata.get("array_profile") not in ("score_recoverability", None):
            raise RuntimeError(f"Cached collection has the wrong array profile: {name}")
        fit, test = available_samples_by_agent(arrays, fit_fraction, split_seed)
        by_task.setdefault(source.task, []).append(
            {
                "run_name": name,
                "fit_min": int(fit.min()),
                "test_min": int(test.min()),
                "fit_by_agent": fit.tolist(),
                "test_by_agent": test.tolist(),
            }
        )
    chosen = {}
    for task, census in by_task.items():
        fit_available = min(item["fit_min"] for item in census)
        test_available = min(item["test_min"] for item in census)
        fit_count = min(fit_cap, fit_available)
        test_count = min(test_cap, test_available)
        if fit_count < 128 or test_count < 32:
            raise RuntimeError(
                f"{task} has too few eligible transitions for a probe: "
                f"fit={fit_available}, test={test_available}; collect more episodes"
            )
        chosen[task] = {
            "fit_samples_per_agent": fit_count,
            "test_samples_per_agent": test_count,
            "fit_available_min": fit_available,
            "test_available_min": test_available,
            "census": census,
        }
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--seeds", default="1,2,3,4")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--max-runs-per-gpu", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--collection-batch-size", type=int, default=64)
    parser.add_argument("--collection-seed-base", type=int, default=730000)
    parser.add_argument("--fit-fraction", type=float, default=0.75)
    parser.add_argument("--split-seed", type=int, default=20260923)
    parser.add_argument("--sampling-seed", type=int, default=20260924)
    parser.add_argument("--fit-samples-per-agent", type=int, default=16384)
    parser.add_argument("--test-samples-per-agent", type=int, default=4096)
    parser.add_argument("--fisher-ridge", type=float, default=1e-3)
    parser.add_argument("--probe-hidden-dim", type=int, default=256)
    parser.add_argument("--probe-steps", type=int, default=2000)
    parser.add_argument("--probe-batch-size", type=int, default=512)
    parser.add_argument("--probe-learning-rate", type=float, default=1e-3)
    parser.add_argument("--probe-seed", type=int, default=20260925)
    parser.add_argument("--cpu-threads-per-worker", type=int, default=0)
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    tasks = parse_items(args.tasks)
    seeds = parse_items(args.seeds, int)
    gpu_ids = parse_items(args.gpus)
    if not tasks or set(tasks) - set(TASKS):
        parser.error(f"--tasks must be drawn from {TASKS}")
    if not seeds or len(set(seeds)) != len(seeds):
        parser.error("--seeds must contain unique seed IDs")
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
        parser.error("--gpus must contain unique GPU IDs")
    if args.max_runs_per_gpu <= 0:
        parser.error("--max-runs-per-gpu must be positive")
    if args.episodes < 2 or not 0 < args.fit_fraction < 1:
        parser.error("episode count and fit fraction are invalid")
    if args.fisher_ridge <= 0:
        parser.error("--fisher-ridge must be positive")

    sources = discover_sources(args.matrix_root.expanduser().resolve())
    selected, audit = choose_complete_cohorts(sources, seeds, CONDITIONS)
    selected_sources = []
    missing = []
    for task in tasks:
        budget = selected.get((task, "nps"))
        if budget is None:
            missing.append(task)
            continue
        for condition in CONDITIONS:
            for seed in seeds:
                key = (task, "nps", budget, condition, seed)
                source = sources.get(key)
                if source is None:
                    missing.append(f"{task}:{condition}:seed{seed}")
                else:
                    selected_sources.append(source)
    if missing:
        relevant = [
            row
            for row in audit
            if row["task"] in tasks and row["actor_parameterization"] == "nps"
        ]
        raise RuntimeError(
            f"No complete final-checkpoint NPS cohort for {missing}; audit={relevant}"
        )

    root = args.run_root.expanduser().resolve()
    for name in ("runs", "logs", "status", "analysis"):
        (root / name).mkdir(parents=True, exist_ok=True)
    protocol = {
        "schema_version": 1,
        "protocol": "smax-score-recoverability-v1.0",
        "matrix_root": str(args.matrix_root.expanduser().resolve()),
        "tasks": list(tasks),
        "conditions": list(CONDITIONS),
        "seeds": list(seeds),
        "actor_parameterization": "nps",
        "selected_budgets": {task: selected[(task, "nps")] for task in tasks},
        "sources": [
            {
                "task": source.task,
                "condition": source.condition,
                "seed": source.seed,
                "budget": source.budget,
                "checkpoint": str(source.checkpoint),
            }
            for source in selected_sources
        ],
        "episodes": args.episodes,
        "collection_batch_size": args.collection_batch_size,
        "collection_seed_base": args.collection_seed_base,
        "collection_seed_policy": "same_seed_within_task",
        "fit_fraction": args.fit_fraction,
        "split_seed": args.split_seed,
        "sampling_seed": args.sampling_seed,
        "fit_samples_per_agent": args.fit_samples_per_agent,
        "test_samples_per_agent": args.test_samples_per_agent,
        "fisher_ridge_absolute": args.fisher_ridge,
        "fisher_source": "fit_split_only",
        "probe_hidden_dim": args.probe_hidden_dim,
        "probe_steps": args.probe_steps,
        "probe_batch_size": args.probe_batch_size,
        "probe_learning_rate": args.probe_learning_rate,
        "probe_seed": args.probe_seed,
    }
    protocol_path = root / "protocol.json"
    if protocol_path.is_file():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing != protocol:
            raise RuntimeError(
                f"Run root contains a different frozen protocol: {protocol_path}"
            )
    else:
        atomic_json(protocol_path, protocol)

    all_jobs = []
    for source in selected_sources:
        output = root / "runs" / source.task / source.condition / f"seed_{source.seed}"
        name = f"{source.task}--{source.condition}--seed{source.seed}"
        all_jobs.append((name, source, output))
    collect_jobs = [
        job
        for job in all_jobs
        if not (job[2] / "collected" / "metadata.json").is_file()
    ]
    probe_jobs = [job for job in all_jobs if not (job[2] / "summary.json").is_file()]
    print(
        f"selected={len(selected_sources)} collect_pending={len(collect_jobs)} "
        f"probe_pending={len(probe_jobs)} "
        f"tasks={','.join(tasks)}",
        flush=True,
    )
    for task in tasks:
        print(
            f"cohort {task}: nps budget={selected[(task, 'nps')]:,} "
            f"runs={len(CONDITIONS) * len(seeds)}",
            flush=True,
        )
    for name, source, output in probe_jobs:
        print(f"{name}: {source.checkpoint} -> {output}", flush=True)
    if args.dry_run:
        return

    slots = len(gpu_ids) * args.max_runs_per_gpu
    cpu_threads = args.cpu_threads_per_worker or max(
        1, (os.cpu_count() or slots) // slots
    )
    launcher_log = root / "launcher.log"

    def log(message):
        timestamp = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        print(line, flush=True)
        with launcher_log.open("a", encoding="utf-8") as file:
            file.write(line + "\n")

    stopping = False

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    def run_stage(stage, jobs, sample_counts=None):
        pending = {gpu: deque() for gpu in gpu_ids}
        for index, job in enumerate(jobs):
            pending[gpu_ids[index % len(gpu_ids)]].append(job)
        running = {}
        failures = 0
        while any(pending.values()) or running:
            if stopping:
                for process, _, _, handle in running.values():
                    process.terminate()
                    handle.close()
                raise SystemExit(130)
            for gpu in gpu_ids:
                active = sum(item[1] == gpu for item in running.values())
                while pending[gpu] and active < args.max_runs_per_gpu:
                    name, source, output = pending[gpu].popleft()
                    output.mkdir(parents=True, exist_ok=True)
                    log_path = root / "logs" / f"{name}.log"
                    handle = log_path.open("a", encoding="utf-8")
                    command = [
                        sys.executable,
                        str(REPO_ROOT / "scripts/eval_smax_score_recoverability.py"),
                        "--checkpoint",
                        str(source.checkpoint),
                        "--output-dir",
                        str(output),
                        "--stage",
                        stage,
                        "--episodes",
                        str(args.episodes),
                        "--collection-batch-size",
                        str(args.collection_batch_size),
                        "--collection-seed",
                        str(task_seed(args.collection_seed_base, source.task)),
                        "--fit-fraction",
                        str(args.fit_fraction),
                        "--split-seed",
                        str(args.split_seed),
                        "--sampling-seed",
                        str(args.sampling_seed),
                        "--fit-samples-per-agent",
                        str(
                            sample_counts[source.task]["fit_samples_per_agent"]
                            if sample_counts is not None
                            else args.fit_samples_per_agent
                        ),
                        "--test-samples-per-agent",
                        str(
                            sample_counts[source.task]["test_samples_per_agent"]
                            if sample_counts is not None
                            else args.test_samples_per_agent
                        ),
                        "--fisher-ridge",
                        str(args.fisher_ridge),
                        "--probe-hidden-dim",
                        str(args.probe_hidden_dim),
                        "--probe-steps",
                        str(args.probe_steps),
                        "--probe-batch-size",
                        str(args.probe_batch_size),
                        "--probe-learning-rate",
                        str(args.probe_learning_rate),
                        "--probe-seed",
                        str(args.probe_seed),
                    ]
                    environment = dict(os.environ)
                    environment.pop("LD_LIBRARY_PATH", None)
                    environment.update(
                        {
                            "CUDA_VISIBLE_DEVICES": gpu,
                            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                            "JAX_ENABLE_X64": "false",
                            "OMP_NUM_THREADS": str(cpu_threads),
                            "OPENBLAS_NUM_THREADS": str(cpu_threads),
                            "MKL_NUM_THREADS": str(cpu_threads),
                            "NUMEXPR_NUM_THREADS": str(cpu_threads),
                        }
                    )
                    process = subprocess.Popen(
                        command,
                        cwd=REPO_ROOT,
                        env=environment,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                    )
                    running[process.pid] = (process, gpu, name, handle)
                    atomic_json(
                        root / "status" / f"{name}.json",
                        {
                            "status": "running",
                            "stage": stage,
                            "pid": process.pid,
                            "gpu": gpu,
                            "run_name": name,
                            "output_dir": str(output),
                        },
                    )
                    log(f"GPU {gpu} START {stage} {name} pid={process.pid}")
                    active += 1
            finished = [
                pid for pid, item in running.items() if item[0].poll() is not None
            ]
            for pid in finished:
                process, gpu, name, handle = running.pop(pid)
                handle.close()
                succeeded = process.returncode == 0
                failures += int(not succeeded)
                status_path = root / "status" / f"{name}.json"
                payload = json.loads(status_path.read_text(encoding="utf-8"))
                payload.update(
                    {
                        "status": (
                            ("collected" if stage == "collect" else "completed")
                            if succeeded
                            else "failed"
                        ),
                        "exit_code": process.returncode,
                    }
                )
                atomic_json(status_path, payload)
                log(f"GPU {gpu} END   {stage} {name} status={process.returncode}")
            if not finished:
                time.sleep(1)
        log(f"{stage} stage finished; failures={failures}")
        if failures:
            raise SystemExit(1)

    run_stage("collect", collect_jobs)
    sample_counts = common_sample_counts(
        all_jobs,
        fit_fraction=args.fit_fraction,
        split_seed=args.split_seed,
        fit_cap=args.fit_samples_per_agent,
        test_cap=args.test_samples_per_agent,
        expected_episodes=args.episodes,
        collection_seed_base=args.collection_seed_base,
    )
    counts_path = root / "sample_counts.json"
    if counts_path.is_file():
        if json.loads(counts_path.read_text(encoding="utf-8")) != sample_counts:
            raise RuntimeError(
                f"Collected sample census changed from the frozen one: {counts_path}"
            )
    else:
        atomic_json(counts_path, sample_counts)
    for task in tasks:
        chosen = sample_counts[task]
        log(
            f"{task} common fit={chosen['fit_samples_per_agent']:,} "
            f"test={chosen['test_samples_per_agent']:,} valid transitions per agent"
        )
    for _, source, output in all_jobs:
        summary_path = output / "summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            chosen = sample_counts[source.task]
            for key in ("fit_samples_per_agent", "test_samples_per_agent"):
                if int(summary[key]) != chosen[key]:
                    raise RuntimeError(
                        f"Existing probe result used a different sample count: {summary_path}"
                    )
    run_stage("probe", probe_jobs, sample_counts)
    log("recoverability jobs finished; failures=0")
    if not args.skip_analysis:
        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts/analyze_smax_score_recoverability.py"),
                "--run-root",
                str(root),
            ],
            cwd=REPO_ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
