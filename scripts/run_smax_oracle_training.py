#!/usr/bin/env python3
"""Launch seed-matched SMAX MAPPO baseline versus oracle distortion training."""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


PROTOCOL_VERSION = "smax-nps-independent-mc-oracle-v2.0"


@dataclass(frozen=True)
class Task:
    map_name: str
    seed: int
    condition: str

    @property
    def run_name(self):
        return (
            f"SMAX-ORACLE-{self.map_name}-nps-{self.condition}-seed{self.seed}"
        )


def csv_values(value):
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result or len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("Expected a non-empty unique CSV")
    return result


def seeds(value):
    if "-" in value and "," not in value:
        start, end = (int(item) for item in value.split("-", 1))
        result = tuple(range(start, end + 1))
    else:
        result = tuple(int(item) for item in csv_values(value))
    if not result or any(item < 0 for item in result):
        raise argparse.ArgumentTypeError("Seeds must be nonnegative")
    return result


def log(path, message):
    line = f"[{dt.datetime.now().astimezone():%Y-%m-%d %H:%M:%S}] {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(line + "\n")


def completed(run_root, task):
    return next(
        iter((run_root / "checkpoints").glob(
            f"**/{task.run_name}-*/final/model.safetensors"
        )),
        None,
    )


def command(repo, run_root, args, task):
    oracle = task.condition == "oracle_latent_distortion"
    return [
        sys.executable,
        str(repo / "baselines/MAPPO/mappo_rnn_smax.py"),
        f"MAP_NAME={args.map_name}",
        f"SEED={task.seed}",
        "ACTOR_PARAMETER_SHARING=false",
        "MATCHED_COMPARISON=true",
        "ALIGN_MODE=none",
        "ALIGN_DISTANCE=ln_mse",
        "ALIGNMENT_COEF=0.1",
        f"ORACLE_LATENT_DISTORTION={'true' if oracle else 'false'}",
        f"ORACLE_DISTORTION_COEF={args.oracle_coef if oracle else 0}",
        f"ORACLE_FISHER_RIDGE={args.fisher_ridge}",
        f"ORACLE_REFERENCE_MULTIPLIER={args.reference_multiplier}",
        f"ORACLE_REFERENCE_BASELINE={args.reference_baseline}",
        f"ORACLE_REFERENCE_SEED_OFFSET={args.reference_seed_offset}",
        f"TOTAL_TIMESTEPS={args.total_timesteps}",
        f"UPDATE_EPOCHS={args.update_epochs}",
        f"LR={args.learning_rate}",
        "SAVE_CHECKPOINTS=true",
        f"CHECKPOINT_INTERVAL_TIMESTEPS={args.checkpoint_interval}",
        f"CHECKPOINT_DIR={run_root / 'checkpoints'}",
        "WANDB_UPLOAD_CHECKPOINTS=false",
        f"WANDB_MODE={args.wandb_mode}",
        f"PROJECT={args.project}",
        f"EXPERIMENT_CONDITION={task.condition}",
        f"MATRIX_PROFILE={PROTOCOL_VERSION}",
        f"PROTOCOL_VERSION={PROTOCOL_VERSION}",
        f"METRICS_JSONL={run_root / 'metrics' / (task.run_name + '.jsonl')}",
        f"hydra.run.dir={run_root / 'hydra' / task.run_name}",
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--map-name", default="10m_vs_11m")
    parser.add_argument("--seeds", type=seeds, default=(1, 2, 3, 4))
    parser.add_argument(
        "--conditions",
        type=csv_values,
        default=("none", "oracle_latent_distortion"),
    )
    parser.add_argument("--oracle-coef", type=float, required=True)
    parser.add_argument("--fisher-ridge", type=float, default=1e-3)
    parser.add_argument("--reference-multiplier", type=int, default=4)
    parser.add_argument(
        "--reference-baseline",
        choices=("frozen_critic", "zero"),
        default="frozen_critic",
    )
    parser.add_argument("--reference-seed-offset", type=int, default=900_000)
    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    parser.add_argument("--checkpoint-interval", type=int, default=1_000_000)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--gpus", type=csv_values, default=("0", "1", "2", "3"))
    parser.add_argument("--max-runs-per-gpu", type=int, default=1)
    parser.add_argument("--project", default="jaxmarl-smax-oracle-distortion")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    allowed = {"none", "oracle_latent_distortion"}
    if set(args.conditions) - allowed:
        parser.error(f"Conditions must be a subset of {sorted(allowed)}")
    if args.oracle_coef <= 0 or args.fisher_ridge <= 0:
        parser.error("Oracle coefficient and Fisher ridge must be positive")
    if args.reference_multiplier < 2:
        parser.error("--reference-multiplier must be at least 2")
    if args.reference_seed_offset <= 0:
        parser.error("--reference-seed-offset must be positive")
    if args.max_runs_per_gpu < 1:
        parser.error("--max-runs-per-gpu must be positive")

    repo = Path(__file__).resolve().parents[1]
    run_root = args.run_root.expanduser().resolve()
    dirs = {name: run_root / name for name in ("logs", "status", "metrics", "checkpoints", "hydra", "wandb")}
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    matrix = [
        Task(args.map_name, seed, condition)
        for seed in args.seeds
        for condition in args.conditions
    ]
    process_text = subprocess.run(
        ("pgrep", "-af", "baselines/MAPPO/mappo_rnn_smax.py"),
        text=True,
        capture_output=True,
    ).stdout
    pending = [
        task for task in matrix
        if completed(run_root, task) is None and task.run_name not in process_text
    ]
    manifest = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "map_name": args.map_name,
        "actor_parameterization": "nps",
        "seeds": list(args.seeds),
        "conditions": list(args.conditions),
        "oracle_distortion_coef": args.oracle_coef,
        "fisher_ridge_absolute": args.fisher_ridge,
        "reference_multiplier": args.reference_multiplier,
        "reference_baseline": args.reference_baseline,
        "reference_seed_offset": args.reference_seed_offset,
        "total_timesteps": args.total_timesteps,
        "update_epochs": args.update_epochs,
        "learning_rate": args.learning_rate,
        "reference_signal": "independent_complete_mc_return_minus_action_independent_baseline",
        "reference_sampling": "fresh_frozen_preupdate_policy_rollouts",
        "critic_signal": "unnormalized_training_gae",
        "signals_are_stop_gradient": True,
    }
    manifest_path = run_root / "experiment_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise RuntimeError(f"Experiment settings changed: {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    launcher_log = run_root / "launcher.log"
    log(launcher_log, f"selected={len(matrix)} pending={len(pending)}")
    for task in matrix:
        state = "COMPLETE" if completed(run_root, task) else "ACTIVE" if task.run_name in process_text else "PENDING"
        print(f"{state:8s} {task.run_name}")
    if args.dry_run or not pending:
        return

    queues = {gpu: collections.deque() for gpu in args.gpus}
    for index, task in enumerate(pending):
        queues[args.gpus[index % len(args.gpus)]].append(task)
    running = {}
    stopping = False

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    failures = 0
    while any(queues.values()) or running:
        if stopping:
            for process, _, _, handle in running.values():
                process.terminate()
                handle.close()
            raise SystemExit(130)
        for gpu in args.gpus:
            active = sum(row[1] == gpu for row in running.values())
            while queues[gpu] and active < args.max_runs_per_gpu:
                task = queues[gpu].popleft()
                output = (dirs["logs"] / f"{task.run_name}.log").open("a", encoding="utf-8")
                environment = os.environ.copy()
                environment.pop("LD_LIBRARY_PATH", None)
                environment.update({
                    "CUDA_VISIBLE_DEVICES": gpu,
                    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                    "HYDRA_FULL_ERROR": "1",
                    "WANDB_DIR": str(dirs["wandb"]),
                    "WANDB_NAME": task.run_name,
                    "WANDB_RUN_GROUP": f"SMAX-{args.map_name}-nps-oracle-comparison",
                    "WANDB_TAGS": f"smax,{args.map_name},nps,oracle-study,{task.condition},seed-{task.seed}",
                })
                process = subprocess.Popen(command(repo, run_root, args, task), cwd=repo, env=environment, stdout=output, stderr=subprocess.STDOUT)
                running[process.pid] = (process, gpu, task, output)
                (dirs["status"] / f"{task.run_name}.json").write_text(json.dumps({"status": "running", "pid": process.pid, "gpu": gpu, "run_name": task.run_name, "condition": task.condition, "seed": task.seed}, indent=2) + "\n")
                log(launcher_log, f"GPU {gpu} START {task.run_name} pid={process.pid}")
                active += 1
        time.sleep(2)
        for pid, (process, gpu, task, output) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            output.close()
            status = "completed" if code == 0 else "failed"
            failures += int(code != 0)
            (dirs["status"] / f"{task.run_name}.json").write_text(json.dumps({"status": status, "exit_code": code, "gpu": gpu, "run_name": task.run_name, "condition": task.condition, "seed": task.seed}, indent=2) + "\n")
            log(launcher_log, f"GPU {gpu} END   {task.run_name} status={code}")
            del running[pid]
    log(launcher_log, f"matrix finished; failures={failures}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
