#!/usr/bin/env python3
"""Launch the matched ShadowHandOver HAPPO/MAPPO/MADPO matrix."""

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
TRAIN_SCRIPT = REPO_ROOT / "experiments" / "harl_dexhands" / "train.py"


def training_environment(gpu: str) -> dict[str, str]:
    """Build an Isaac Gym child environment from the active Python runtime."""

    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    conda_prefix = Path(environment.get("CONDA_PREFIX", sys.prefix)).expanduser()
    runtime_lib = str(conda_prefix / "lib")
    current = environment.get("LD_LIBRARY_PATH", "")
    entries = [entry for entry in current.split(os.pathsep) if entry]
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(
        [runtime_lib, *(entry for entry in entries if entry != runtime_lib)]
    )
    return environment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--harl-root", type=Path, default=REPO_ROOT / "third_party" / "HARL"
    )
    parser.add_argument("--algorithms", default="happo,mappo,madpo")
    parser.add_argument("--seeds", default="1-4")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--max-runs-per-gpu", type=int, default=1)
    parser.add_argument("--num-env-steps", type=int)
    parser.add_argument("--n-rollout-threads", type=int)
    parser.add_argument("--div-coef", type=float, default=1000.0)
    parser.add_argument("--div-weight", type=float, default=0.05)
    parser.add_argument("--div-sigma", type=float, default=1.0)
    parser.add_argument("--div-max-samples", type=int, default=1024)
    parser.add_argument("--wandb-project", default="harl-dexhands-shadowhandover")
    parser.add_argument(
        "--wandb-mode", choices=("online", "disabled"), default="online"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT))
    from experiments.harl_dexhands.protocol import (
        ALGORITHMS,
        PROTOCOL_VERSION,
        parse_csv,
        parse_seeds,
        task_matrix,
    )

    if args.max_runs_per_gpu < 1:
        raise ValueError("max-runs-per-gpu must be positive")
    algorithms = parse_csv(args.algorithms, ALGORITHMS)
    seeds = parse_seeds(args.seeds)
    gpus = tuple(piece.strip() for piece in args.gpus.split(",") if piece.strip())
    if not gpus:
        raise ValueError("Select at least one GPU")
    root = args.run_root.expanduser().resolve()
    harl_root = args.harl_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    tasks = task_matrix(
        algorithms,
        seeds,
        div_coef=args.div_coef,
        div_weight=args.div_weight,
        div_sigma=args.div_sigma,
        div_max_samples=args.div_max_samples,
    )

    def is_completed(task) -> bool:
        path = root / "status" / f"{task.name}.json"
        if not path.is_file():
            return False
        try:
            return (
                json.loads(path.read_text(encoding="utf-8")).get("status")
                == "completed"
            )
        except (OSError, ValueError):
            return False

    pending = [task for task in tasks if not is_completed(task)]
    print(
        f"Protocol={PROTOCOL_VERSION} total={len(tasks)} pending={len(pending)} "
        f"algorithms={','.join(algorithms)} seeds={','.join(map(str, seeds))}",
        flush=True,
    )
    slots = [gpu for gpu in gpus for _ in range(args.max_runs_per_gpu)]
    queues = [[] for _ in slots]
    for index, task in enumerate(pending):
        queues[index % len(slots)].append(task)
    lock = threading.Lock()
    launcher_log = root / "launcher.log"

    def event(message: str) -> None:
        line = f"[{dt.datetime.now().astimezone():%Y-%m-%d %H:%M:%S}] {message}"
        with lock:
            print(line, flush=True)
            with launcher_log.open("a", encoding="utf-8") as file:
                file.write(line + "\n")

    def run_queue(slot: int) -> None:
        gpu = slots[slot]
        for task in queues[slot]:
            command = [
                sys.executable,
                str(TRAIN_SCRIPT),
                "--harl-root",
                str(harl_root),
                "--run-root",
                str(root),
                "--run-name",
                task.name,
                "--algorithm",
                task.algorithm,
                "--seed",
                str(task.seed),
                "--div-coef",
                str(args.div_coef),
                "--div-weight",
                str(args.div_weight),
                "--div-sigma",
                str(args.div_sigma),
                "--div-max-samples",
                str(args.div_max_samples),
                "--wandb-project",
                args.wandb_project,
                "--wandb-mode",
                args.wandb_mode,
            ]
            if args.num_env_steps is not None:
                command.extend(["--num-env-steps", str(args.num_env_steps)])
            if args.n_rollout_threads is not None:
                command.extend(["--n-rollout-threads", str(args.n_rollout_threads)])
            event(f"GPU {gpu} START {task.name}")
            if args.dry_run:
                event("COMMAND " + " ".join(command))
                continue
            environment = training_environment(gpu)
            output = root / "logs" / f"{task.name}.log"
            with output.open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            event(f"GPU {gpu} END   {task.name} status={result.returncode}")

    with ThreadPoolExecutor(max_workers=len(slots)) as executor:
        futures = [executor.submit(run_queue, index) for index in range(len(slots))]
        for future in futures:
            future.result()
    failures = 0
    for task in tasks:
        path = root / "status" / f"{task.name}.json"
        if not path.is_file():
            failures += int(not args.dry_run)
            continue
        try:
            failures += json.loads(path.read_text()).get("status") == "failed"
        except (OSError, ValueError):
            failures += 1
    event(f"matrix finished; failures={failures}")
    if failures:
        raise RuntimeError(f"{failures} runs failed; inspect {root / 'logs'}")


if __name__ == "__main__":
    main()
