#!/usr/bin/env python3
"""Calibrate one global VMAS CKA coefficient from initial actor gradients."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.benchmarl_vmas.protocol import (
    CALIBRATION_PROTOCOL_VERSION,
    REFERENCE_ALIGNMENT_COEF,
    TASKS,
    parse_csv,
)


WORKER_SCRIPT = REPO_ROOT / "experiments" / "benchmarl_vmas" / "calibration_worker.py"


def gradient_norm(loss, parameters, retain_graph=True) -> float:
    import torch

    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=retain_graph, allow_unused=True
    )
    squared = sum(
        gradient.detach().square().sum()
        for gradient in gradients
        if gradient is not None
    )
    return float(squared.sqrt().cpu())


def run_cell(args, task: str, output: Path) -> None:
    # Imports remain inside workers so the launcher itself does not initialize CUDA.
    from experiments.benchmarl_vmas.train_alignment import build_experiment

    cell_args = argparse.Namespace(
        run_root=output.parent,
        run_name=f"VMAS-CKA-calibration-{task}-nps-seed{args.pilot_seed}",
        task=task,
        seed=args.pilot_seed,
        condition="c_to_a_mse",
        align_mode="c_to_a",
        align_distance="ln_mse",
        alignment_coef=0.0,
        alignment_epsilon=1e-8,
        device="cuda",
        max_frames=60_000,
        frames_per_batch=60_000,
        num_envs=600,
        minibatch_iters=1,
        minibatch_size=4096,
        evaluation_interval=120_000,
        evaluation_episodes=1,
        checkpoint_interval=0,
        wandb_project="benchmarl-vmas-cka-calibration",
        wandb_mode="disabled",
        disable_evaluation=True,
        disable_logging=True,
    )
    experiment = build_experiment(cell_args)
    try:
        batch = next(iter(experiment.collector))
        group = next(iter(experiment.group_map))
        group_batch = batch.exclude(*experiment._get_excluded_keys(group)).to(
            experiment.config.train_device
        )
        group_batch = experiment.algorithm.process_batch(group, group_batch).reshape(-1)
        minibatch = group_batch[: experiment.config.on_policy_minibatch_size]
        loss_module = experiment.losses[group]
        actor_parameters = list(
            loss_module.actor_network_params.flatten_keys().values()
        )
        ppo = loss_module(minibatch)["loss_objective"]
        mse = loss_module.alignment_loss(minibatch, "ln_mse")
        cka = loss_module.alignment_loss(minibatch, "linear_cka")
        rl_norm = gradient_norm(ppo, actor_parameters)
        mse_norm = gradient_norm(mse, actor_parameters)
        cka_norm = gradient_norm(cka, actor_parameters, retain_graph=False)
        if min(rl_norm, mse_norm, cka_norm) <= 0 or not all(
            math.isfinite(value) for value in (rl_norm, mse_norm, cka_norm)
        ):
            raise RuntimeError("non-positive or non-finite calibration gradient")
        payload = {
            "task": task,
            "pilot_seed": args.pilot_seed,
            "actor_parameterization": "nps",
            "gradient_recipient": "actor",
            "rl_gradient_norm": rl_norm,
            "ln_mse_gradient_norm": mse_norm,
            "linear_cka_gradient_norm": cka_norm,
            "ln_mse_cross_to_rl_ratio": mse_norm / rl_norm,
            "linear_cka_cross_to_rl_ratio": cka_norm / rl_norm,
            "cell_matching_coef": REFERENCE_ALIGNMENT_COEF * mse_norm / cka_norm,
            "selection_uses_return": False,
        }
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    finally:
        experiment.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pilot-seed", type=int, default=9001)
    parser.add_argument("--gpus", type=parse_csv, default=("0", "1", "2", "3"))
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.pilot_seed in {1, 2, 3, 4} or args.pilot_seed < 0:
        raise ValueError("use an independent non-negative pilot seed")
    root = args.output_root.expanduser().resolve()
    metrics = root / "metrics"
    logs = root / "logs"
    metrics.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    def launch(index: int, task: str) -> None:
        gpu = args.gpus[index % len(args.gpus)]
        output = metrics / f"{task}.json"
        if output.exists() and not args.rerun:
            print(f"GPU {gpu} SKIP  {task}", flush=True)
            return
        command = [
            sys.executable,
            str(WORKER_SCRIPT),
            "--task",
            task,
            "--output",
            str(output),
            "--pilot-seed",
            str(args.pilot_seed),
        ]
        print(f"GPU {gpu} START {task}", flush=True)
        if args.dry_run:
            print(" ".join(command), flush=True)
            return
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = gpu
        log = logs / f"{task}.log"
        with log.open("w", encoding="utf-8") as handle:
            result = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        print(f"GPU {gpu} END   {task} status={result.returncode}", flush=True)
        if result.returncode:
            raise RuntimeError(f"calibration failed; inspect {log}")

    with ThreadPoolExecutor(max_workers=min(len(TASKS), len(args.gpus))) as executor:
        futures = [
            executor.submit(launch, index, task) for index, task in enumerate(TASKS)
        ]
        for future in futures:
            future.result()
    if args.dry_run:
        return

    cells = [json.loads((metrics / f"{task}.json").read_text()) for task in TASKS]
    reference_squared = sum(cell["ln_mse_cross_to_rl_ratio"] ** 2 for cell in cells)
    target_squared = sum(cell["linear_cka_cross_to_rl_ratio"] ** 2 for cell in cells)
    coefficient = REFERENCE_ALIGNMENT_COEF * math.sqrt(
        reference_squared / target_squared
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    payload = {
        "schema_version": 1,
        "protocol_version": CALIBRATION_PROTOCOL_VERSION,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": commit,
        "tasks": list(TASKS),
        "actor_parameterization": "nps",
        "pilot_seed": args.pilot_seed,
        "selection_uses_return": False,
        "performance_fields_persisted": False,
        "reference_distance": "ln_mse",
        "reference_alignment_coef": REFERENCE_ALIGNMENT_COEF,
        "target_distance": "linear_cka",
        "aggregation": "equal-task pooled RMS of initial cross/RL gradient ratios",
        "global_alignment_coef": coefficient,
        "cells": cells,
    }
    destination = root / "cka_gradient_calibration.json"
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Global lambda_CKA: {coefficient:.10g}")
    print(destination)


if __name__ == "__main__":
    main()
