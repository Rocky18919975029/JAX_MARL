#!/usr/bin/env python3
"""Calibrate one HARL Linear-CKA coefficient without looking at returns."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = REPO_ROOT / "experiments" / "harl_mamujoco" / "train_alignment.py"


@dataclass(frozen=True)
class Task:
    actor_label: str
    sharing: bool
    direction: str
    seed: int

    @property
    def name(self) -> str:
        return f"HARL-CKA-calibration-Humanoid17x1-{self.actor_label}-{self.direction}-seed{self.seed}"


def load_result(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "ln_mse_cross_to_rl_ratio",
        "linear_cka_cross_to_rl_ratio",
        "selection_uses_return",
    }
    if not required.issubset(payload) or payload["selection_uses_return"] is not False:
        raise ValueError(f"Incomplete calibration result: {path}")
    for key in ("ln_mse_cross_to_rl_ratio", "linear_cka_cross_to_rl_ratio"):
        if not math.isfinite(float(payload[key])) or float(payload[key]) <= 0:
            raise ValueError(f"Invalid {key} in {path}")
    return payload


def pooled_rms_coefficient(cells: list[dict], reference_coef: float) -> float:
    ln_squared = sum(float(cell["ln_mse_cross_to_rl_ratio"]) ** 2 for cell in cells)
    cka_squared = sum(
        float(cell["linear_cka_cross_to_rl_ratio"]) ** 2 for cell in cells
    )
    if ln_squared <= 0 or cka_squared <= 0:
        raise ValueError("Calibration ratios must be positive")
    return reference_coef * math.sqrt(ln_squared / cka_squared)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--harl-root", type=Path, default=REPO_ROOT / "third_party" / "HARL"
    )
    parser.add_argument("--pilot-seed", type=int, default=9001)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--reference-coef", type=float, default=0.1)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.pilot_seed in {1, 2, 3, 4} or args.pilot_seed < 0:
        raise ValueError("Use an independent, non-negative pilot seed")
    if args.reference_coef <= 0:
        raise ValueError("reference-coef must be positive")
    gpu_ids = tuple(piece.strip() for piece in args.gpus.split(",") if piece.strip())
    if not gpu_ids:
        raise ValueError("Select at least one GPU")

    root = args.run_root.expanduser().resolve()
    harl_root = args.harl_root.expanduser().resolve()
    metrics_root = root / "metrics"
    logs_root = root / "logs"
    metrics_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)
    tasks = [
        Task(actor_label, sharing, direction, args.pilot_seed)
        for actor_label, sharing in (("ps", True), ("nps", False))
        for direction in ("c_to_a", "a_to_c")
    ]

    def run_task(index: int, task: Task) -> None:
        gpu = gpu_ids[index % len(gpu_ids)]
        output = metrics_root / f"{task.name}.json"
        if output.exists() and not args.rerun:
            try:
                load_result(output)
                print(f"GPU {gpu} SKIP  {task.name}", flush=True)
                return
            except (OSError, ValueError, KeyError):
                pass
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
            task.direction,
            "--align-distance",
            "ln_mse",
            "--alignment-coef",
            "0",
            "--wandb-mode",
            "disabled",
            "--calibration-output",
            str(output),
        ]
        print(f"GPU {gpu} START {task.name}", flush=True)
        if args.dry_run:
            print(" ".join(command), flush=True)
            return
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = gpu
        log_path = logs_root / f"{task.name}.log"
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        print(f"GPU {gpu} END   {task.name} status={result.returncode}", flush=True)
        if result.returncode != 0:
            raise RuntimeError(f"Calibration failed; inspect {log_path}")

    with ThreadPoolExecutor(max_workers=min(len(tasks), len(gpu_ids))) as executor:
        futures = [
            executor.submit(run_task, index, task) for index, task in enumerate(tasks)
        ]
        for future in futures:
            future.result()
    if args.dry_run:
        return

    cells = [load_result(metrics_root / f"{task.name}.json") for task in tasks]
    coefficient = pooled_rms_coefficient(cells, args.reference_coef)
    source_config = (
        harl_root / "tuned_configs/mamujoco/Humanoid-v2-17x1/mappo/config.json"
    )
    payload = {
        "schema_version": 1,
        "protocol_version": "harl-mamujoco-cka-calibration-v1.0",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "task": "Humanoid-v2-17x1",
        "pilot_seed": args.pilot_seed,
        "selection_uses_return": False,
        "performance_fields_persisted": False,
        "reference_distance": "ln_mse",
        "reference_alignment_coef": args.reference_coef,
        "target_distance": "linear_cka",
        "aggregation": "equal-cell pooled RMS of initial cross/RL gradient ratios",
        "global_alignment_coef": coefficient,
        "source_config": str(source_config),
        "source_config_sha256": hashlib.sha256(source_config.read_bytes()).hexdigest(),
        "cells": cells,
    }
    destination = root / "cka_gradient_calibration.json"
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Global lambda_CKA: {coefficient:.10g}")
    print(destination)


if __name__ == "__main__":
    main()
