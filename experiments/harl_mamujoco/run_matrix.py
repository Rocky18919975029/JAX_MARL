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
        suffix = {
            "ln_mse": "mse",
            "linear_cka": "cka",
            "containment": "dsc",
        }[self.distance]
        return f"{self.mode}_{suffix}"

    @property
    def name(self) -> str:
        return (
            f"HARL-Humanoid-v2-17x1-{self.actor_label}-{self.condition}-"
            f"lam{label_float(self.coefficient)}-seed{self.seed}"
        )


def load_alignment_coefficient(path: Path, target_distance: str) -> float:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if payload.get("selection_uses_return") is not False:
        raise ValueError("Alignment calibration must not use returns")
    if payload.get("performance_fields_persisted") is not False:
        raise ValueError("CKA calibration artifact must exclude performance fields")
    if payload.get("task") != "Humanoid-v2-17x1":
        raise ValueError(
            "Use the HARL Humanoid-v2-17x1 calibration, not SMAX calibration"
        )
    if (
        payload.get("reference_distance") != "ln_mse"
        or payload.get("target_distance") != target_distance
    ):
        raise ValueError("Invalid alignment calibration distance pair")
    value = float(payload["global_alignment_coef"])
    if not (value > 0 and value < float("inf")):
        raise ValueError("Calibrated coefficient must be finite and positive")
    return value


def task_matrix(
    seeds: tuple[int, ...],
    cka_coefficient: float | None,
    distances=("ln_mse", "linear_cka"),
    containment_coefficient: float | None = None,
    actor_variants=("ps", "nps"),
    directions=("c_to_a", "a_to_c", "joint"),
) -> list[Task]:
    tasks = []
    sharing_by_label = {"ps": True, "nps": False}
    for actor_label in actor_variants:
        sharing = sharing_by_label[actor_label]
        for seed in seeds:
            tasks.append(Task(actor_label, sharing, "none", "ln_mse", 0.1, seed))
            coefficients = {
                "ln_mse": 0.1,
                "linear_cka": cka_coefficient,
                "containment": containment_coefficient,
            }
            for distance in distances:
                coefficient = coefficients[distance]
                if coefficient is None:
                    raise ValueError(f"Missing calibrated coefficient for {distance}")
                for mode in directions:
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
    parser.add_argument("--cka-calibration", type=Path)
    parser.add_argument("--containment-calibration", type=Path)
    parser.add_argument("--distances", default="ln_mse,linear_cka")
    parser.add_argument("--actor-variants", default="ps,nps")
    parser.add_argument("--directions", default="c_to_a,a_to_c,joint")
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
    distances = tuple(
        piece.strip() for piece in args.distances.split(",") if piece.strip()
    )
    if (
        not distances
        or len(distances) != len(set(distances))
        or not set(distances).issubset({"ln_mse", "linear_cka", "containment"})
    ):
        raise ValueError("Invalid --distances selection")
    actor_variants = tuple(
        piece.strip() for piece in args.actor_variants.split(",") if piece.strip()
    )
    if (
        not actor_variants
        or len(actor_variants) != len(set(actor_variants))
        or not set(actor_variants).issubset({"ps", "nps"})
    ):
        raise ValueError("Invalid --actor-variants selection")
    directions = tuple(
        piece.strip() for piece in args.directions.split(",") if piece.strip()
    )
    if (
        not directions
        or len(directions) != len(set(directions))
        or not set(directions).issubset({"c_to_a", "a_to_c", "joint"})
    ):
        raise ValueError("Invalid --directions selection")
    cka_coefficient = None
    containment_coefficient = None
    if "linear_cka" in distances:
        if args.cka_calibration is None:
            raise ValueError("--cka-calibration is required for linear_cka")
        cka_coefficient = load_alignment_coefficient(args.cka_calibration, "linear_cka")
    if "containment" in distances:
        if args.containment_calibration is None:
            raise ValueError("--containment-calibration is required for containment")
        containment_coefficient = load_alignment_coefficient(
            args.containment_calibration, "containment"
        )
    tasks = task_matrix(
        args.seeds,
        cka_coefficient,
        distances,
        containment_coefficient,
        actor_variants,
        directions,
    )
    pending = [
        task
        for task in tasks
        if not (root / "checkpoints" / task.name / "completed.json").is_file()
    ]
    print(
        f"Protocol={PROTOCOL} unique_runs={len(tasks)} pending={len(pending)} "
        f"seeds={','.join(map(str, args.seeds))} "
        f"lambda_CKA={cka_coefficient} lambda_DSC={containment_coefficient}",
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
                "--containment-ridge-ratio",
                "0.001",
                "--containment-epsilon",
                "0.000001",
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
