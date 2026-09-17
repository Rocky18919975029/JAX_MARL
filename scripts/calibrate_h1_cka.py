#!/usr/bin/env python3
"""Calibrate one global relational-alignment coefficient from initial gradients."""

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

try:
    from h1_protocol import FROZEN_KEYS, repository_root
except ModuleNotFoundError:  # Imported as scripts.calibrate_h1_cka in tests.
    from scripts.h1_protocol import FROZEN_KEYS, repository_root


MAPS = ("10m_vs_11m", "smacv2_10_units")
CALIBRATION_MAPS = (
    *MAPS,
    "2s3z",
    "3s5z_vs_3s6z",
    "6h_vs_8z",
)
ACTOR_VARIANTS = (("ps", True), ("nps", False))
DIRECTIONS = ("c_to_a", "a_to_c")
CALIBRATION_PROTOCOL = "h1-cka-gradient-calibration-v1.0"
TARGET_DISTANCES = ("linear_cka", "containment")


@dataclass(frozen=True)
class CalibrationTask:
    map_name: str
    actor_label: str
    sharing: bool
    direction: str
    pilot_seed: int
    calibration_label: str = "CKA"

    @property
    def name(self):
        return (
            f"{self.calibration_label}-calibration-{self.map_name}-{self.actor_label}-"
            f"{self.direction}-seed{self.pilot_seed}"
        )


def parse_csv(value, allowed):
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(values) - set(allowed))
    if not values or unknown:
        raise argparse.ArgumentTypeError(
            "selection must be non-empty"
            + (f"; unknown values: {', '.join(unknown)}" if unknown else "")
        )
    return values


def hydra_value(value):
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def load_frozen_config(path):
    path = path.expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    frozen = payload.get("training_config", payload)
    missing = [key for key in FROZEN_KEYS if key not in frozen]
    if missing:
        raise ValueError(f"Frozen config is missing: {', '.join(missing)}")
    return frozen, path


def build_command(
    task, frozen, output_root, metrics_path, commit, protocol=CALIBRATION_PROTOCOL
):
    probe = dict(frozen)
    probe.update(
        {
            "MATCHED_COMPARISON": True,
            "ALIGNMENT_COEF": 0.0,
            "TOTAL_TIMESTEPS": int(frozen["NUM_ENVS"] * frozen["NUM_STEPS"]),
            "UPDATE_EPOCHS": 1,
            # Keep the frozen NUM_MINIBATCHES. LR=0 holds parameters fixed, so
            # all actual-size minibatches probe the same initial parameters.
            "LR": 0.0,
            "ANNEAL_LR": False,
        }
    )
    command = [
        sys.executable,
        str(repository_root() / "baselines/MAPPO/mappo_rnn_smax.py"),
    ]
    for key in FROZEN_KEYS:
        if key == "ENV_KWARGS":
            for nested_key, value in probe[key].items():
                command.append(f"ENV_KWARGS.{nested_key}={hydra_value(value)}")
        else:
            command.append(f"{key}={hydra_value(probe[key])}")
    command.extend(
        (
            f"MAP_NAME={task.map_name}",
            f"SEED={task.pilot_seed}",
            f"ACTOR_PARAMETER_SHARING={hydra_value(task.sharing)}",
            f"ALIGN_MODE={task.direction}",
            "ALIGN_DISTANCE=ln_mse",
            "ALIGN_DISTANCE_EPS=1e-8",
            "ALIGN_CONTAINMENT_RIDGE_RATIO=1e-3",
            "ALIGN_CONTAINMENT_EPS=1e-6",
            "ALIGN_CONTAINMENT_GROUP_BY_UNIT_TYPE=true",
            "ALIGN_CONTAINMENT_MIN_GROUP_SAMPLES=64",
            "ALIGN_GRADIENT_CALIBRATION=true",
            "ALIGN_TARGET_SHUFFLE=false",
            "SAVE_CHECKPOINTS=false",
            f"METRICS_JSONL={metrics_path}",
            f"MATRIX_PROFILE={protocol}",
            f"PROTOCOL_VERSION={protocol}",
            f"GIT_COMMIT={commit}",
            "WANDB_MODE=disabled",
            f"PROJECT=h1-smax-{task.calibration_label.lower()}-calibration",
            f"hydra.run.dir={output_root / 'hydra' / task.name}",
        )
    )
    return command


def load_last_metrics(path):
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != 1:
        raise RuntimeError(
            f"Expected one calibration update in {path}, got {len(records)}"
        )
    return records[0]


def metrics_complete(path, target_distance="linear_cka"):
    target_prefix = f"calibration_{target_distance}"
    required = {
        "calibration_ln_mse_actor_cross_to_rl_ratio",
        "calibration_ln_mse_critic_cross_to_rl_ratio",
        f"{target_prefix}_actor_cross_to_rl_ratio",
        f"{target_prefix}_critic_cross_to_rl_ratio",
    }
    try:
        return required.issubset(load_last_metrics(path))
    except (OSError, ValueError, RuntimeError):
        return False


def extract_cell(task, metrics, target_distance="linear_cka"):
    if task.direction == "c_to_a":
        reference_key = "calibration_ln_mse_actor_cross_to_rl_ratio"
        target_key = f"calibration_{target_distance}_actor_cross_to_rl_ratio"
        recipient = "actor"
    else:
        reference_key = "calibration_ln_mse_critic_cross_to_rl_ratio"
        target_key = f"calibration_{target_distance}_critic_cross_to_rl_ratio"
        recipient = "critic"
    reference_ratio = float(metrics[reference_key])
    target_ratio = float(metrics[target_key])
    if not (
        math.isfinite(reference_ratio)
        and math.isfinite(target_ratio)
        and reference_ratio > 0
        and target_ratio > 0
    ):
        raise RuntimeError(
            f"Invalid gradient ratios for {task.name}: "
            f"LN-MSE={reference_ratio}, {target_distance}={target_ratio}"
        )
    return {
        "map_name": task.map_name,
        "actor_parameterization": task.actor_label,
        "actor_parameter_sharing": task.sharing,
        "direction": task.direction,
        "gradient_recipient": recipient,
        "pilot_seed": task.pilot_seed,
        "ln_mse_cross_to_rl_ratio": reference_ratio,
        f"{target_distance}_cross_to_rl_ratio": target_ratio,
        "cell_matching_coef": 0.1 * reference_ratio / target_ratio,
    }


def pooled_rms_coefficient(
    cells, reference_coef=0.1, target_distance="linear_cka"
):
    """Match pooled RMS cross/RL ratios with equal weight per pilot cell."""

    if not cells or reference_coef <= 0:
        raise ValueError("Calibration needs cells and a positive reference coefficient")
    reference_squared = sum(
        float(cell["ln_mse_cross_to_rl_ratio"]) ** 2 for cell in cells
    )
    target_squared = sum(
        float(cell[f"{target_distance}_cross_to_rl_ratio"]) ** 2 for cell in cells
    )
    if reference_squared <= 0 or target_squared <= 0:
        raise ValueError("Calibration gradient ratios must be positive")
    return reference_coef * math.sqrt(reference_squared / target_squared)


def git_commit(repo):
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pilot-seed", type=int, default=9001)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument(
        "--maps",
        type=lambda value: parse_csv(value, CALIBRATION_MAPS),
        default=MAPS,
    )
    parser.add_argument(
        "--actor-variants",
        type=lambda value: parse_csv(value, dict(ACTOR_VARIANTS)),
        default=tuple(dict(ACTOR_VARIANTS)),
    )
    parser.add_argument(
        "--directions",
        type=lambda value: parse_csv(value, DIRECTIONS),
        default=DIRECTIONS,
    )
    parser.add_argument("--reference-coef", type=float, default=0.1)
    parser.add_argument(
        "--target-distance", choices=TARGET_DISTANCES, default="linear_cka"
    )
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.pilot_seed < 0 or args.reference_coef <= 0:
        raise ValueError("Pilot seed must be non-negative and coefficient positive")
    if args.pilot_seed in {*range(1, 5), *range(101, 111)}:
        raise ValueError("Pilot seed must be independent from all experiment seeds")
    gpu_ids = tuple(item.strip() for item in args.gpus.split(",") if item.strip())
    if not gpu_ids:
        raise ValueError("Select at least one GPU")

    frozen, frozen_path = load_frozen_config(args.frozen_config)
    output_root = args.output_root.expanduser().resolve()
    metrics_root = output_root / "metrics"
    logs_root = output_root / "logs"
    metrics_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)
    actor_lookup = dict(ACTOR_VARIANTS)
    calibration_label = "CKA" if args.target_distance == "linear_cka" else "DSC"
    calibration_protocol = (
        CALIBRATION_PROTOCOL
        if args.target_distance == "linear_cka"
        else "h1-containment-gradient-calibration-v1.0"
    )
    tasks = [
        CalibrationTask(
            map_name,
            actor_label,
            actor_lookup[actor_label],
            direction,
            args.pilot_seed,
            calibration_label,
        )
        for map_name in args.maps
        for actor_label in args.actor_variants
        for direction in args.directions
    ]
    commit = git_commit(repository_root())

    pending_by_gpu = {gpu: [] for gpu in gpu_ids}
    for index, task in enumerate(tasks):
        metrics_path = metrics_root / f"{task.name}.jsonl"
        if metrics_complete(metrics_path, args.target_distance) and not args.rerun:
            continue
        pending_by_gpu[gpu_ids[index % len(gpu_ids)]].append(task)

    def run_gpu_queue(gpu, pending):
        for task in pending:
            metrics_path = metrics_root / f"{task.name}.jsonl"
            log_path = logs_root / f"{task.name}.log"
            command = build_command(
                task,
                frozen,
                output_root,
                metrics_path,
                commit,
                calibration_protocol,
            )
            print(f"GPU {gpu} START {task.name}", flush=True)
            if args.dry_run:
                print(" ".join(map(str, command)), flush=True)
                continue
            metrics_path.unlink(missing_ok=True)
            environment = dict(os.environ)
            environment.pop("LD_LIBRARY_PATH", None)
            environment.update(
                {
                    "CUDA_VISIBLE_DEVICES": gpu,
                    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                    "HYDRA_FULL_ERROR": "1",
                    "WANDB_NAME": task.name,
                }
            )
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    command,
                    cwd=repository_root(),
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            print(
                f"GPU {gpu} END   {task.name} status={completed.returncode}",
                flush=True,
            )
            if completed.returncode != 0:
                raise RuntimeError(f"Calibration task failed; inspect {log_path}")

    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = [
            executor.submit(run_gpu_queue, gpu, pending)
            for gpu, pending in pending_by_gpu.items()
        ]
        for future in futures:
            future.result()
    if args.dry_run:
        return

    cells = []
    for task in tasks:
        metrics = load_last_metrics(metrics_root / f"{task.name}.jsonl")
        cells.append(extract_cell(task, metrics, args.target_distance))
    coefficient = pooled_rms_coefficient(
        cells, args.reference_coef, args.target_distance
    )
    frozen_hash = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    result = {
        "schema_version": 1,
        "protocol_version": calibration_protocol,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": commit,
        "frozen_config": str(frozen_path),
        "frozen_config_sha256": frozen_hash,
        "pilot_seed": args.pilot_seed,
        "selection_uses_return": False,
        "performance_fields_persisted": False,
        "reference_distance": "ln_mse",
        "reference_alignment_coef": args.reference_coef,
        "target_distance": args.target_distance,
        "containment_ridge_ratio": 1e-3,
        "containment_epsilon": 1e-6,
        "containment_group_by_unit_type": True,
        "containment_min_group_samples": 64,
        "aggregation": "equal-cell pooled RMS of initial cross/RL gradient ratios",
        "global_alignment_coef": coefficient,
        "cells": cells,
    }
    stem = "cka" if args.target_distance == "linear_cka" else "containment"
    destination = output_root / f"{stem}_gradient_calibration.json"
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Global lambda_{calibration_label}: {coefficient:.10g}")
    print(destination)


if __name__ == "__main__":
    main()
