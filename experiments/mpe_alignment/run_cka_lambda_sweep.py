#!/usr/bin/env python3
"""Launch a restart-safe C→A Linear CKA coefficient sweep on Spread-5."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.mpe_alignment.protocol import (
    DEFAULT_SEEDS,
    TASKS,
    float_label,
    load_cka_coefficient,
    parse_csv,
    parse_seeds,
)


SWEEP_PROTOCOL = "mpe-simple-spread5-nps-cka-lambda-sweep-v1.0"
TRAIN_SCRIPT = REPO_ROOT / "experiments" / "mpe_alignment" / "train_alignment.py"


def parse_multipliers(value: str) -> tuple[float, ...]:
    result = tuple(float(piece.strip()) for piece in value.split(",") if piece.strip())
    if (
        not result
        or len(result) != len(set(result))
        or any(not math.isfinite(item) or item <= 0 for item in result)
    ):
        raise argparse.ArgumentTypeError(
            "multipliers must be unique finite positive numbers"
        )
    return result


@dataclass(frozen=True)
class SweepRun:
    seed: int
    base_coefficient: float
    multiplier: float

    @property
    def coefficient(self) -> float:
        return self.base_coefficient * self.multiplier

    @property
    def name(self) -> str:
        return (
            "MPE-simple_spread_5-nps-c_to_a_cka-"
            f"lam{float_label(self.coefficient)}-seed{self.seed}"
        )


def sweep_matrix(
    base_coefficient: float,
    multipliers: tuple[float, ...],
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> list[SweepRun]:
    if not math.isfinite(base_coefficient) or base_coefficient <= 0:
        raise ValueError("base coefficient must be finite and positive")
    return [
        SweepRun(seed, base_coefficient, multiplier)
        for multiplier in multipliers
        for seed in seeds
    ]


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
    parser.add_argument(
        "--multipliers",
        type=parse_multipliers,
        default=(0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0),
    )
    parser.add_argument("--seeds", type=parse_seeds, default=DEFAULT_SEEDS)
    parser.add_argument("--gpus", type=parse_csv, default=("0", "1", "2", "3"))
    parser.add_argument("--max-runs-per-gpu", type=int, default=3)
    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    parser.add_argument(
        "--wandb-project", default="jaxmarl-mpe-spread5-cka-lambda-sweep"
    )
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="online"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_runs_per_gpu < 1:
        raise ValueError("max-runs-per-gpu must be positive")
    if args.total_timesteps < 1:
        raise ValueError("total-timesteps must be positive")
    if subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--"], cwd=REPO_ROOT
    ).returncode:
        raise RuntimeError("tracked worktree is dirty; commit before launching")

    root = args.run_root.expanduser().resolve()
    (root / "logs").mkdir(parents=True, exist_ok=True)
    base_coefficient = load_cka_coefficient(args.cka_calibration)
    runs = sweep_matrix(base_coefficient, args.multipliers, args.seeds)
    pending = [run for run in runs if not is_completed(root, run.name)]
    print(
        f"Protocol={SWEEP_PROTOCOL} total={len(runs)} pending={len(pending)} "
        f"base_lambda={base_coefficient:.10g} "
        f"multipliers={','.join(map(str, args.multipliers))} "
        f"max_parallel={len(args.gpus) * args.max_runs_per_gpu}",
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
                TASKS[0],
                "--seed",
                str(run.seed),
                "--condition",
                "c_to_a_cka",
                "--align-mode",
                "c_to_a",
                "--align-distance",
                "linear_cka",
                "--alignment-coef",
                str(run.coefficient),
                "--sweep-base-coefficient",
                str(run.base_coefficient),
                "--sweep-multiplier",
                str(run.multiplier),
                "--protocol-version",
                SWEEP_PROTOCOL,
                "--wandb-group",
                "simple_spread_5-nps-cka-lambda-sweep",
                "--total-timesteps",
                str(args.total_timesteps),
                "--wandb-project",
                args.wandb_project,
                "--wandb-mode",
                args.wandb_mode,
            ]
            event(f"GPU {gpu} START {run.name} multiplier={run.multiplier:g}")
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
        event("lambda sweep finished; failures=0")


if __name__ == "__main__":
    main()
