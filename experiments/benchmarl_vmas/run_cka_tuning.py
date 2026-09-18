#!/usr/bin/env python3
"""Launch an exploratory, CKA-only coefficient sweep on VMAS."""

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
from dataclasses import asdict, dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.benchmarl_vmas.protocol import (
    TASKS,
    float_label,
    load_cka_coefficients,
    parse_csv,
    parse_seeds,
)


TRAIN_SCRIPT = REPO_ROOT / "experiments" / "benchmarl_vmas" / "train_alignment.py"
TUNING_PROTOCOL_VERSION = "benchmarl-vmas-nps-cka-tuning-v1.0"
DEFAULT_MULTIPLIERS = (
    1 / 128,
    1 / 64,
    1 / 32,
    1 / 16,
    1 / 8,
    1 / 4,
    1 / 2,
    1.0,
)


def parse_multipliers(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("multipliers must be numbers") from error
    if (
        not values
        or len(values) != len(set(values))
        or any(not math.isfinite(item) or item <= 0 for item in values)
    ):
        raise argparse.ArgumentTypeError(
            "multipliers must be unique, finite, and positive"
        )
    return values


@dataclass(frozen=True)
class Candidate:
    task: str
    seed: int
    calibration_coefficient: float
    multiplier: float
    coefficient: float

    @property
    def name(self) -> str:
        return (
            f"VMAS-CKATUNE-{self.task}-nps-c_to_a_cka-"
            f"mul{float_label(self.multiplier)}-"
            f"lam{float_label(self.coefficient)}-seed{self.seed}"
        )


def candidate_matrix(
    coefficients: dict[str, float],
    tasks: tuple[str, ...],
    seeds: tuple[int, ...],
    multipliers: tuple[float, ...],
) -> list[Candidate]:
    candidates = []
    for task in tasks:
        if task not in TASKS:
            raise ValueError(f"unsupported task: {task}")
        base = coefficients[task]
        for seed in seeds:
            for multiplier in multipliers:
                candidates.append(
                    Candidate(task, seed, base, multiplier, base * multiplier)
                )
    return candidates


def completed(root: Path, run_name: str) -> bool:
    path = root / "status" / f"{run_name}.json"
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "completed"
    except (OSError, ValueError):
        return False


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--cka-calibration", type=Path, required=True)
    parser.add_argument("--tasks", type=parse_csv, default=TASKS)
    parser.add_argument("--seeds", type=parse_seeds, default=(1, 2))
    parser.add_argument(
        "--multipliers",
        type=parse_multipliers,
        default=DEFAULT_MULTIPLIERS,
        help="positive multipliers applied to each task's calibrated CKA coefficient",
    )
    parser.add_argument("--gpus", type=parse_csv, default=("0", "1", "2", "3"))
    parser.add_argument("--max-runs-per-gpu", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=10_000_000)
    parser.add_argument("--wandb-project", default="benchmarl-vmas-nps-cka-tuning-v1")
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="online"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_runs_per_gpu < 1:
        raise ValueError("max-runs-per-gpu must be positive")
    if args.max_frames <= 0:
        raise ValueError("max-frames must be positive")
    invalid_tasks = set(args.tasks) - set(TASKS)
    if invalid_tasks:
        raise ValueError(f"unsupported tasks: {sorted(invalid_tasks)}")
    if subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--"], cwd=REPO_ROOT
    ).returncode:
        raise RuntimeError(
            "tracked worktree is dirty; commit the tuning protocol before launching"
        )

    root = args.run_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    coefficients = load_cka_coefficients(args.cka_calibration)
    candidates = candidate_matrix(
        coefficients, args.tasks, args.seeds, args.multipliers
    )
    pending = [
        candidate for candidate in candidates if not completed(root, candidate.name)
    ]

    manifest = {
        "schema_version": 1,
        "protocol_version": TUNING_PROTOCOL_VERSION,
        "git_commit": git_commit(),
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "exploratory_return_tuning": True,
        "confirmatory_evidence": False,
        "actor_parameterization": "nps",
        "condition": "c_to_a_cka",
        "tasks_reported_separately": True,
        "tasks": list(args.tasks),
        "seeds": list(args.seeds),
        "max_frames": args.max_frames,
        "calibration_artifact": str(args.cka_calibration.expanduser().resolve()),
        "task_calibration_coefficients": {
            task: coefficients[task] for task in args.tasks
        },
        "multipliers": list(args.multipliers),
        "candidates": [
            asdict(candidate) | {"run_name": candidate.name} for candidate in candidates
        ],
    }
    (root / "tuning_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"Protocol={TUNING_PROTOCOL_VERSION} total={len(candidates)} "
        f"pending={len(pending)} tasks={','.join(args.tasks)} "
        f"seeds={','.join(map(str, args.seeds))} "
        f"multipliers={','.join(f'{value:.10g}' for value in args.multipliers)}",
        flush=True,
    )

    slots = [gpu for gpu in args.gpus for _ in range(args.max_runs_per_gpu)]
    queues = [[] for _ in slots]
    for index, candidate in enumerate(pending):
        queues[index % len(slots)].append(candidate)
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
        for candidate in queues[index]:
            command = [
                sys.executable,
                str(TRAIN_SCRIPT),
                "--run-root",
                str(root),
                "--run-name",
                candidate.name,
                "--task",
                candidate.task,
                "--seed",
                str(candidate.seed),
                "--condition",
                "c_to_a_cka",
                "--align-mode",
                "c_to_a",
                "--align-distance",
                "linear_cka",
                "--alignment-coef",
                str(candidate.coefficient),
                "--cka-calibration-coef",
                str(candidate.calibration_coefficient),
                "--cka-multiplier",
                str(candidate.multiplier),
                "--experiment-stage",
                "cka_tuning",
                "--max-frames",
                str(args.max_frames),
                "--wandb-project",
                args.wandb_project,
                "--wandb-mode",
                args.wandb_mode,
            ]
            event(f"GPU {gpu} START {candidate.name}")
            if args.dry_run:
                event("COMMAND " + " ".join(command))
                continue
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            log_path = root / "logs" / f"{candidate.name}.log"
            with log_path.open("w", encoding="utf-8") as handle:
                result = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
            event(f"GPU {gpu} END   {candidate.name} status={result.returncode}")
            if result.returncode:
                raise RuntimeError(f"run failed; inspect {log_path}")

    with ThreadPoolExecutor(max_workers=len(slots)) as executor:
        futures = [executor.submit(run_queue, index) for index in range(len(slots))]
        for future in futures:
            future.result()
    if not args.dry_run:
        event("CKA tuning matrix finished; failures=0")


if __name__ == "__main__":
    main()
