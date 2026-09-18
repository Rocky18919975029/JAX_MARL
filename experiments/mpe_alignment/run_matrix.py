#!/usr/bin/env python3
"""Launch the 12-run Simple Spread-5 NPS alignment matrix."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.mpe_alignment.protocol import (
    DEFAULT_SEEDS,
    PROTOCOL_VERSION,
    TASKS,
    load_cka_coefficient,
    matrix,
    parse_csv,
    parse_seeds,
)


TRAIN_SCRIPT = REPO_ROOT / "experiments" / "mpe_alignment" / "train_alignment.py"


def is_completed(root: Path, name: str) -> bool:
    path = root / "status" / f"{name}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))["status"] == "completed"
    except (OSError, ValueError, KeyError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--cka-calibration", type=Path, required=True)
    parser.add_argument("--tasks", type=parse_csv, default=TASKS)
    parser.add_argument("--seeds", type=parse_seeds, default=DEFAULT_SEEDS)
    parser.add_argument("--gpus", type=parse_csv, default=("0", "1", "2", "3"))
    parser.add_argument("--max-runs-per-gpu", type=int, default=3)
    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    parser.add_argument("--wandb-project", default="jaxmarl-mpe-spread5-alignment")
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="online"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_runs_per_gpu < 1:
        raise ValueError("max-runs-per-gpu must be positive")
    if set(args.tasks) - set(TASKS):
        raise ValueError(f"unsupported tasks: {args.tasks}")
    if subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--"], cwd=REPO_ROOT
    ).returncode:
        raise RuntimeError("tracked worktree is dirty; commit before launching")

    root = args.run_root.expanduser().resolve()
    (root / "logs").mkdir(parents=True, exist_ok=True)
    coefficient = load_cka_coefficient(args.cka_calibration)
    runs = matrix(args.seeds, args.tasks, coefficient)
    pending = [run for run in runs if not is_completed(root, run.name)]
    print(
        f"Protocol={PROTOCOL_VERSION} total={len(runs)} pending={len(pending)} "
        f"lambda_CKA={coefficient:.10g} max_parallel="
        f"{len(args.gpus) * args.max_runs_per_gpu}",
        flush=True,
    )
    slots = [gpu for gpu in args.gpus for _ in range(args.max_runs_per_gpu)]
    queues = [[] for _ in slots]
    for index, run in enumerate(pending):
        queues[index % len(slots)].append(run)
    lock = threading.Lock()
    launcher_log = root / "launcher.log"

    def event(message: str) -> None:
        line = f"[{dt.datetime.now().astimezone():%Y-%m-%d %H:%M:%S}] {message}"
        with lock:
            print(line, flush=True)
            with launcher_log.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def run_queue(index: int) -> None:
        gpu = slots[index]
        for run in queues[index]:
            command = [
                sys.executable,
                str(TRAIN_SCRIPT),
                "--run-root",
                str(root),
                "--run-name",
                run.name,
                "--task",
                run.task,
                "--seed",
                str(run.seed),
                "--condition",
                run.condition,
                "--align-mode",
                run.align_mode,
                "--align-distance",
                run.align_distance,
                "--alignment-coef",
                str(run.coefficient),
                "--total-timesteps",
                str(args.total_timesteps),
                "--wandb-project",
                args.wandb_project,
                "--wandb-mode",
                args.wandb_mode,
            ]
            event(f"GPU {gpu} START {run.name}")
            if args.dry_run:
                event("COMMAND " + " ".join(command))
                continue
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
            log = root / "logs" / f"{run.name}.log"
            with log.open("w", encoding="utf-8") as handle:
                result = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
            event(f"GPU {gpu} END   {run.name} status={result.returncode}")
            if result.returncode:
                raise RuntimeError(f"run failed; inspect {log}")

    with ThreadPoolExecutor(max_workers=len(slots)) as executor:
        futures = [executor.submit(run_queue, index) for index in range(len(slots))]
        for future in futures:
            future.result()
    if not args.dry_run:
        event("matrix finished; failures=0")


if __name__ == "__main__":
    main()
