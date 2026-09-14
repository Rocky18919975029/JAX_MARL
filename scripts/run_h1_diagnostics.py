#!/usr/bin/env python3
"""Schedule H1 diagnostic collection/analyses across GPUs."""

from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import json
import os
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

try:
    from eval_h1_checkpoints import discover_tasks
except ModuleNotFoundError:  # Imported as scripts.run_h1_diagnostics in tests.
    from scripts.eval_h1_checkpoints import discover_tasks


REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE_MARKERS = {
    "collect": "metadata.json",
    "latent": "latent_distortion_summary.json",
    "decision": "decision_summary.json",
    "bellman": "bellman_summary.json",
}


def worker_environment(base_environment, gpu):
    """Build a training-dtype-compatible environment for a diagnostic worker."""

    environment = dict(base_environment)
    environment.pop("LD_LIBRARY_PATH", None)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            # Frozen checkpoints and recurrent carries were trained in
            # float32. Statistical routines promote selected quantities to
            # NumPy float64 internally when required.
            "JAX_ENABLE_X64": "false",
        }
    )
    return environment


def append_jsonl(path, payload):
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, sort_keys=True) + "\n")
        file.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--max-runs-per-gpu", type=int, default=1)
    parser.add_argument("--stages", default="collect,latent,decision,bellman")
    parser.add_argument("--output-tree", default="diagnostics_raw")
    parser.add_argument("--run-name-glob", default="H1-*")
    parser.add_argument("--checkpoint-name-glob", default="*")
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--anchors", type=int, default=256)
    parser.add_argument("--continuations", type=int, default=32)
    parser.add_argument("--bellman-heads", type=int, default=32)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    run_root = args.run_root.expanduser().resolve()
    gpu_ids = tuple(item.strip() for item in args.gpus.split(",") if item.strip())
    stages = tuple(item.strip() for item in args.stages.split(",") if item.strip())
    unknown = set(stages) - set(STAGE_MARKERS)
    if unknown:
        raise ValueError(f"Unknown stages: {sorted(unknown)}")
    tasks, missing = discover_tasks(run_root)
    if missing and not args.allow_missing:
        raise RuntimeError(
            f"{len(missing)} preregistered checkpoints are missing; finish training "
            "or pass --allow-missing for a partial diagnostic batch"
        )
    tasks = [
        task
        for task in tasks
        if fnmatch.fnmatch(task.run_name, args.run_name_glob)
        and fnmatch.fnmatch(task.checkpoint_dir.name, args.checkpoint_name_glob)
    ]
    selected = []
    for task in tasks:
        output = run_root / args.output_tree / task.run_name / task.checkpoint_dir.name
        complete = all((output / STAGE_MARKERS[stage]).is_file() for stage in stages)
        if not complete:
            selected.append((task, output))
    pending = {gpu: deque() for gpu in gpu_ids}
    for index, item in enumerate(selected):
        pending[gpu_ids[index % len(gpu_ids)]].append(item)

    log_dir = run_root / "logs" / "diagnostics"
    log_dir.mkdir(parents=True, exist_ok=True)
    launcher_log = log_dir / "launcher.log"
    manifest = run_root / "diagnostics_summary" / "diagnostic_manifest.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)

    def log(message):
        timestamp = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        print(line, flush=True)
        with launcher_log.open("a", encoding="utf-8") as file:
            file.write(line + "\n")

    log(
        f"stages={','.join(stages)} discovered={len(tasks)} pending={len(selected)} "
        f"missing={len(missing)}"
    )
    running = {}
    stopping = False

    def stop_handler(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)
    failures = 0
    while any(pending.values()) or running:
        if stopping:
            for process, _, _, handle, _ in running.values():
                process.terminate()
                handle.close()
            raise SystemExit(130)
        for gpu in gpu_ids:
            active = sum(item[1] == gpu for item in running.values())
            while pending[gpu] and active < args.max_runs_per_gpu:
                task, output = pending[gpu].popleft()
                output.mkdir(parents=True, exist_ok=True)
                log_path = log_dir / f"{task.run_name}-{task.checkpoint_dir.name}.log"
                handle = log_path.open("w", encoding="utf-8")
                command = [
                    sys.executable,
                    str(REPO_ROOT / "scripts/h1_diagnostic_worker.py"),
                    "--checkpoint",
                    str(task.checkpoint_dir),
                    "--output-dir",
                    str(output),
                    "--training-seed",
                    str(task.training_seed),
                    "--checkpoint-index",
                    str(task.checkpoint_index),
                    "--stages",
                    ",".join(stages),
                    "--episodes",
                    str(args.episodes),
                    "--batch-size",
                    str(args.batch_size),
                    "--anchors",
                    str(args.anchors),
                    "--continuations",
                    str(args.continuations),
                    "--bellman-heads",
                    str(args.bellman_heads),
                ]
                environment = worker_environment(os.environ, gpu)
                started = dt.datetime.now(dt.timezone.utc).isoformat()
                process = subprocess.Popen(
                    command,
                    cwd=REPO_ROOT,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
                running[process.pid] = (process, gpu, task, handle, started)
                log(
                    f"GPU {gpu} START {task.run_name}/{task.checkpoint_dir.name} "
                    f"pid={process.pid}"
                )
                active += 1
        finished = [pid for pid, item in running.items() if item[0].poll() is not None]
        for pid in finished:
            process, gpu, task, handle, started = running.pop(pid)
            handle.close()
            failures += process.returncode != 0
            record = {
                "schema_version": 1,
                "run_name": task.run_name,
                "checkpoint": str(task.checkpoint_dir),
                "checkpoint_index": task.checkpoint_index,
                "training_seed": task.training_seed,
                "stages": stages,
                "gpu": gpu,
                "started_at_utc": started,
                "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "status": "completed" if process.returncode == 0 else "failed",
                "exit_code": process.returncode,
            }
            append_jsonl(manifest, record)
            log(
                f"GPU {gpu} END   {task.run_name}/{task.checkpoint_dir.name} "
                f"status={process.returncode}"
            )
        if not finished:
            time.sleep(1)
    log(f"diagnostics finished; failures={failures}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
