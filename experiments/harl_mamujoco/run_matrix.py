#!/usr/bin/env python3
"""Launch the four-seed Humanoid-v2-17x1 alignment matrix over four GPUs."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = REPO_ROOT / "experiments" / "harl_mamujoco" / "train_alignment.py"
PROTOCOL = "harl-mamujoco-alignment-v1.0"


def parse_seeds(value: str) -> tuple[int, ...]:
    seeds = []
    for piece in value.split(","):
        piece = piece.strip()
        if "-" in piece:
            start, end = map(int, piece.split("-", 1))
            seeds.extend(range(start, end + 1))
        elif piece:
            seeds.append(int(piece))
    if not seeds or len(seeds) != len(set(seeds)) or any(seed < 0 for seed in seeds):
        raise argparse.ArgumentTypeError(
            "Seeds must be a unique non-negative list/range"
        )
    return tuple(seeds)


def label_float(value: float) -> str:
    return f"{value:.10g}".replace("-", "m").replace(".", "p")


@dataclass(frozen=True)
class Task:
    actor_label: str
    sharing: bool
    mode: str
    distance: str
    coefficient: float
    seed: int

    @property
    def condition(self) -> str:
        if self.mode == "none":
            return "none"
        suffix = "mse" if self.distance == "ln_mse" else "cka"
        return f"{self.mode}_{suffix}"

    @property
    def name(self) -> str:
        return (
            f"HARL-Humanoid-v2-17x1-{self.actor_label}-{self.condition}-"
            f"lam{label_float(self.coefficient)}-seed{self.seed}"
        )


def load_cka_coefficient(path: Path) -> float:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if payload.get("selection_uses_return") is not False:
        raise ValueError("CKA calibration must not use returns")
    if payload.get("performance_fields_persisted") is not False:
        raise ValueError("CKA calibration artifact must exclude performance fields")
    if payload.get("task") != "Humanoid-v2-17x1":
        raise ValueError(
            "Use the HARL Humanoid-v2-17x1 calibration, not SMAX calibration"
        )
    if (
        payload.get("reference_distance") != "ln_mse"
        or payload.get("target_distance") != "linear_cka"
    ):
        raise ValueError("Invalid CKA calibration distance pair")
    value = float(payload["global_alignment_coef"])
    if not (value > 0 and value < float("inf")):
        raise ValueError("Calibrated CKA coefficient must be finite and positive")
    return value


def task_matrix(seeds: tuple[int, ...], cka_coefficient: float) -> list[Task]:
    tasks = []
    for actor_label, sharing in (("ps", True), ("nps", False)):
        for seed in seeds:
            tasks.append(Task(actor_label, sharing, "none", "ln_mse", 0.1, seed))
            for distance, coefficient in (
                ("ln_mse", 0.1),
                ("linear_cka", cka_coefficient),
            ):
                for mode in ("c_to_a", "a_to_c", "joint"):
                    tasks.append(
                        Task(actor_label, sharing, mode, distance, coefficient, seed)
                    )
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--harl-root", type=Path, default=REPO_ROOT / "third_party" / "HARL"
    )
    parser.add_argument("--cka-calibration", type=Path, required=True)
    parser.add_argument("--seeds", type=parse_seeds, default=(1, 2, 3, 4))
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--max-runs-per-gpu", type=int, default=1)
    parser.add_argument("--wandb-project", default="harl-mamujoco-alignment")
    parser.add_argument("--checkpoint-interval-steps", type=int, default=500_000)
    parser.add_argument("--upload-checkpoints", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_runs_per_gpu < 1:
        raise ValueError("max-runs-per-gpu must be positive")
    gpu_ids = tuple(piece.strip() for piece in args.gpus.split(",") if piece.strip())
    if not gpu_ids:
        raise ValueError("Select at least one GPU")

    root = args.run_root.expanduser().resolve()
    harl_root = args.harl_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    cka_coefficient = load_cka_coefficient(args.cka_calibration)
    tasks = task_matrix(args.seeds, cka_coefficient)
    pending = [
        task
        for task in tasks
        if not (root / "checkpoints" / task.name / "completed.json").is_file()
    ]
    print(
        f"Protocol={PROTOCOL} unique_runs={len(tasks)} pending={len(pending)} "
        f"seeds={','.join(map(str, args.seeds))} lambda_CKA={cka_coefficient:.10g}",
        flush=True,
    )

    slots = [gpu for gpu in gpu_ids for _ in range(args.max_runs_per_gpu)]
    queues = [[] for _ in slots]
    for index, task in enumerate(pending):
        queues[index % len(slots)].append(task)
    log_lock = threading.Lock()
    launcher_log = root / "launcher.log"

    def event(message: str) -> None:
        line = f"[{dt.datetime.now().astimezone():%Y-%m-%d %H:%M:%S}] {message}"
        with log_lock:
            print(line, flush=True)
            with launcher_log.open("a", encoding="utf-8") as file:
                file.write(line + "\n")

    def run_queue(slot_index: int) -> None:
        gpu = slots[slot_index]
        for task in queues[slot_index]:
            command = [
                sys.executable,
                str(TRAIN_SCRIPT),
                "--harl-root",
                str(harl_root),
                "--run-root",
                str(root),
                "--run-name",
                task.name,
                "--seed",
                str(task.seed),
                "--actor-parameter-sharing",
                str(task.sharing).lower(),
                "--align-mode",
                task.mode,
                "--align-distance",
                task.distance,
                "--alignment-coef",
                str(task.coefficient),
                "--checkpoint-interval-steps",
                str(args.checkpoint_interval_steps),
                "--wandb-project",
                args.wandb_project,
                "--wandb-mode",
                "online",
                "--protocol-version",
                PROTOCOL,
            ]
            if args.upload_checkpoints:
                command.append("--wandb-upload-checkpoints")
            event(f"GPU {gpu} START {task.name}")
            if args.dry_run:
                event("COMMAND " + " ".join(command))
                continue
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            output = root / "logs" / f"{task.name}.log"
            with output.open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            event(f"GPU {gpu} END   {task.name} status={completed.returncode}")
            if completed.returncode != 0:
                raise RuntimeError(f"Run failed; inspect {output}")

    with ThreadPoolExecutor(max_workers=len(slots)) as executor:
        futures = [executor.submit(run_queue, index) for index in range(len(slots))]
        for future in futures:
            future.result()
    if not args.dry_run:
        event("matrix finished; failures=0")


if __name__ == "__main__":
    main()
