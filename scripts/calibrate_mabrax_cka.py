#!/usr/bin/env python3
"""Calibrate one MABrax linear-CKA coefficient from initial gradients."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path


PROTOCOL_VERSION = "mabrax-cka-gradient-calibration-v1.0"
ACTOR_VARIANTS = (("ps", True), ("nps", False))
DIRECTIONS = ("c_to_a", "a_to_c")


def repository_root():
    return Path(__file__).resolve().parents[1]


def parse_csv(value):
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("expected a non-empty list without duplicates")
    return values


@dataclass(frozen=True)
class Task:
    actor_label: str
    sharing: bool
    direction: str
    seed: int

    @property
    def name(self):
        return (
            f"MABRAX-CKA-calibration-halfcheetah_6x1-{self.actor_label}-"
            f"{self.direction}-seed{self.seed}"
        )


def load_metrics(path):
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != 1:
        raise RuntimeError(f"Expected one update in {path}, found {len(records)}")
    return records[0]


def extract_cell(task, metrics):
    recipient = "actor" if task.direction == "c_to_a" else "critic"
    ln_key = f"calibration_ln_mse_{recipient}_cross_to_rl_ratio"
    cka_key = f"calibration_linear_cka_{recipient}_cross_to_rl_ratio"
    ln_ratio = float(metrics[ln_key])
    cka_ratio = float(metrics[cka_key])
    if not all(math.isfinite(value) and value > 0 for value in (ln_ratio, cka_ratio)):
        raise RuntimeError(f"Invalid calibration ratios for {task.name}")
    return {
        "actor_parameterization": task.actor_label,
        "actor_parameter_sharing": task.sharing,
        "direction": task.direction,
        "gradient_recipient": recipient,
        "pilot_seed": task.seed,
        "ln_mse_cross_to_rl_ratio": ln_ratio,
        "linear_cka_cross_to_rl_ratio": cka_ratio,
        "cell_matching_coef": 0.1 * ln_ratio / cka_ratio,
    }


def pooled_rms_coefficient(cells, reference_coef=0.1):
    reference_squared = sum(cell["ln_mse_cross_to_rl_ratio"] ** 2 for cell in cells)
    cka_squared = sum(cell["linear_cka_cross_to_rl_ratio"] ** 2 for cell in cells)
    if not cells or reference_squared <= 0 or cka_squared <= 0:
        raise ValueError("Calibration cells and gradient ratios must be positive")
    return reference_coef * math.sqrt(reference_squared / cka_squared)


def build_command(task, output_root, metrics_path):
    return [
        sys.executable,
        str(repository_root() / "baselines/MAPPO/mappo_ff_mabrax.py"),
        "ENV_NAME=halfcheetah_6x1",
        f"SEED={task.seed}",
        f"ACTOR_PARAMETER_SHARING={str(task.sharing).lower()}",
        "MATCHED_COMPARISON=true",
        f"ALIGN_MODE={task.direction}",
        "ALIGN_DISTANCE=ln_mse",
        "ALIGN_DISTANCE_EPS=1e-8",
        "ALIGNMENT_COEF=0",
        "ALIGN_GRADIENT_CALIBRATION=true",
        "NUM_ENVS=64",
        "NUM_STEPS=300",
        "TOTAL_TIMESTEPS=19200",
        "UPDATE_EPOCHS=1",
        "NUM_MINIBATCHES=4",
        "LR=0",
        "ANNEAL_LR=false",
        "SAVE_CHECKPOINTS=false",
        f"METRICS_JSONL={metrics_path}",
        f"MATRIX_PROFILE={PROTOCOL_VERSION}",
        f"PROTOCOL_VERSION={PROTOCOL_VERSION}",
        "WANDB_MODE=disabled",
        "PROJECT=jaxmarl-mabrax-cka-calibration",
        f"hydra.run.dir={output_root / 'hydra' / task.name}",
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pilot-seed", type=int, default=9001)
    parser.add_argument("--gpus", type=parse_csv, default=("0", "1", "2", "3"))
    parser.add_argument("--reference-coef", type=float, default=0.1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.reference_coef <= 0:
        parser.error("--reference-coef must be positive")
    output_root = args.output_root.expanduser().resolve()
    metrics_root = output_root / "metrics"
    logs_root = output_root / "logs"
    metrics_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)
    tasks = [
        Task(label, sharing, direction, args.pilot_seed)
        for label, sharing in ACTOR_VARIANTS
        for direction in DIRECTIONS
    ]

    queues = {gpu: [] for gpu in args.gpus}
    for index, task in enumerate(tasks):
        queues[args.gpus[index % len(args.gpus)]].append(task)

    def run_queue(gpu, tasks_for_gpu):
        for task in tasks_for_gpu:
            metrics = metrics_root / f"{task.name}.jsonl"
            log = logs_root / f"{task.name}.log"
            command = build_command(task, output_root, metrics)
            print(f"GPU {gpu} START {task.name}", flush=True)
            if args.dry_run:
                print(" ".join(map(str, command)), flush=True)
                continue
            metrics.unlink(missing_ok=True)
            environment = os.environ.copy()
            environment.pop("LD_LIBRARY_PATH", None)
            environment.update(
                {
                    "CUDA_VISIBLE_DEVICES": gpu,
                    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                    "HYDRA_FULL_ERROR": "1",
                    "WANDB_NAME": task.name,
                }
            )
            with log.open("w", encoding="utf-8") as handle:
                completed = subprocess.run(
                    command,
                    cwd=repository_root(),
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
            print(
                f"GPU {gpu} END   {task.name} status={completed.returncode}", flush=True
            )
            if completed.returncode:
                raise RuntimeError(f"Calibration failed; inspect {log}")

    with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        futures = [
            executor.submit(run_queue, gpu, queue) for gpu, queue in queues.items()
        ]
        for future in futures:
            future.result()
    if args.dry_run:
        return

    cells = [
        extract_cell(task, load_metrics(metrics_root / f"{task.name}.jsonl"))
        for task in tasks
    ]
    coefficient = pooled_rms_coefficient(cells, args.reference_coef)
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repository_root(),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    result = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": commit,
        "environment": "halfcheetah_6x1",
        "pilot_seed": args.pilot_seed,
        "selection_uses_return": False,
        "reference_distance": "ln_mse",
        "reference_alignment_coef": args.reference_coef,
        "target_distance": "linear_cka",
        "aggregation": "equal-cell pooled RMS of initial cross/RL gradient ratios",
        "global_alignment_coef": coefficient,
        "cells": cells,
    }
    destination = output_root / "cka_gradient_calibration.json"
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Global lambda_CKA: {coefficient:.10g}")
    print(destination)


if __name__ == "__main__":
    main()
