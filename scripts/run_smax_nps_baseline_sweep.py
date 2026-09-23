#!/usr/bin/env python3
"""Sweep PPO learning rate and update epochs for isolated 6s9z SMAX MAPPO."""

from __future__ import annotations

import argparse
import collections
import importlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    from scripts.run_smax_actor_score_recovery_sweep import (
        hydra_float,
        positive_float_items,
        positive_int_items,
    )
    from scripts.run_smax_score_recovery_training import (
        atomic_json,
        coef_tag,
        csv_items,
        log,
        seed_items,
    )
except ModuleNotFoundError:  # Direct execution from scripts/.
    from run_smax_actor_score_recovery_sweep import (
        hydra_float,
        positive_float_items,
        positive_int_items,
    )
    from run_smax_score_recovery_training import (
        atomic_json,
        coef_tag,
        csv_items,
        log,
        seed_items,
    )


PROTOCOL = "smax-6s9z-nps-isolated-ppo-sweep-v1.0"
MAP_NAME = "6s9z_vs_6s10z"
DEFAULT_BUDGET = 20_000_000
INCUMBENT = (0.002, 4)


@dataclass(frozen=True)
class BaselineRun:
    seed: int
    steps: int
    learning_rate: float
    update_epochs: int

    @property
    def name(self) -> str:
        return (
            f"SMAX-NONE-SWEEP-{MAP_NAME}-nps-lr{coef_tag(self.learning_rate)}"
            f"-ep{self.update_epochs}-seed{self.seed}"
        )


def run_matrix(seeds, learning_rates, update_epochs, steps) -> list[BaselineRun]:
    runs = [
        BaselineRun(seed, steps, lr, epochs)
        for seed in seeds
        for lr in learning_rates
        for epochs in update_epochs
    ]
    if len({run.name for run in runs}) != len(runs):
        raise ValueError("Baseline grid produces duplicate run names")
    return runs


def command(repo: Path, root: Path, args, run: BaselineRun) -> list[str]:
    return [
        sys.executable,
        str(repo / "baselines/MAPPO/mappo_rnn_smax.py"),
        f"MAP_NAME={MAP_NAME}",
        f"SEED={run.seed}",
        "ACTOR_PARAMETER_SHARING=false",
        "MATCHED_COMPARISON=true",
        "ALIGN_MODE=none",
        "ALIGN_DISTANCE=ln_mse",
        "ALIGNMENT_COEF=0",
        "ALIGN_GRADIENT_CALIBRATION=false",
        "ALIGN_TARGET_SHUFFLE=false",
        "ORACLE_LATENT_DISTORTION=false",
        "ORACLE_DISTORTION_COEF=0",
        "SCORE_RECOVERY=false",
        "SCORE_RECOVERY_COEF=0",
        "ACTOR_SCORE_RECOVERY=false",
        "ACTOR_SCORE_RECOVERY_COEF=0",
        "ACTOR_SCORE_RECOVERY_FISHER_RIDGE=0.001",
        "ACTOR_SCORE_RECOVERY_Q_LR=0.001",
        "ACTOR_SCORE_RECOVERY_Q_STEPS=8",
        f"TOTAL_TIMESTEPS={run.steps}",
        f"UPDATE_EPOCHS={run.update_epochs}",
        f"LR={hydra_float(run.learning_rate)}",
        f"NUM_ENVS={args.num_envs}",
        f"NUM_MINIBATCHES={args.num_minibatches}",
        "SAVE_CHECKPOINTS=true",
        f"CHECKPOINT_INTERVAL_TIMESTEPS={args.checkpoint_interval}",
        f"CHECKPOINT_DIR={root / 'checkpoints'}",
        "WANDB_UPLOAD_CHECKPOINTS=false",
        f"WANDB_MODE={args.wandb_mode}",
        f"PROJECT={args.project}",
        "EXPERIMENT_CONDITION=none",
        f"MATRIX_PROFILE={PROTOCOL}",
        f"PROTOCOL_VERSION={PROTOCOL}",
        f"METRICS_JSONL={root / 'metrics' / (run.name + '.jsonl')}",
        f"hydra.run.dir={root / 'hydra' / run.name}",
    ]


def completed(root: Path, run: BaselineRun) -> bool:
    status_path = root / "status" / f"{run.name}.json"
    if not status_path.is_file():
        return False
    if json.loads(status_path.read_text(encoding="utf-8")).get("status") != "completed":
        return False
    return any((root / "checkpoints").glob(f"**/{run.name}-*/final/model.safetensors"))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seeds", type=seed_items, default=(1, 2, 3, 4))
    parser.add_argument(
        "--learning-rates",
        type=positive_float_items,
        default=(0.0005, 0.001, 0.002),
    )
    parser.add_argument("--update-epochs-grid", type=positive_int_items, default=(2, 4))
    parser.add_argument("--total-timesteps", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--checkpoint-interval", type=int, default=1_000_000)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--gpus", type=csv_items, default=("0", "1", "2", "3"))
    parser.add_argument("--max-runs-per-gpu", type=int, default=2)
    parser.add_argument("--project", default="jaxmarl-smax-none-6s9z-ppo-sweep")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.seeds or not args.learning_rates or not args.update_epochs_grid:
        parser.error("Seeds, learning rates and epochs grids must be nonempty")
    if not args.gpus:
        parser.error("At least one GPU is required")
    if args.total_timesteps <= 0 or args.checkpoint_interval <= 0:
        parser.error("Training budget and checkpoint interval must be positive")
    if args.num_envs <= 0 or args.num_minibatches <= 0:
        parser.error("Environment and minibatch counts must be positive")
    if args.num_envs % args.num_minibatches:
        parser.error("NUM_ENVS must be divisible by NUM_MINIBATCHES")
    if args.max_runs_per_gpu <= 0:
        parser.error("--max-runs-per-gpu must be positive")
    if any(not math.isfinite(lr) or lr <= 0 for lr in args.learning_rates):
        parser.error("Learning rates must be finite and positive")
    return args


def status_payload(run: BaselineRun, status: str, **extra) -> dict:
    return {
        "status": status,
        "run_name": run.name,
        "map_name": MAP_NAME,
        "condition": "none",
        "seed": run.seed,
        "budget": run.steps,
        "learning_rate": run.learning_rate,
        "update_epochs": run.update_epochs,
        **extra,
    }


def main() -> None:
    args = parse_args()
    if not args.dry_run:
        try:
            importlib.import_module("jax")
        except Exception as exc:
            raise SystemExit(
                "JAX could not be imported in this Python environment. "
                "Activate the jaxmarl conda environment before launching. "
                f"Original error: {exc!r}"
            ) from exc
    repo = Path(__file__).resolve().parents[1]
    root = args.run_root.expanduser().resolve()
    for directory in ("logs", "status", "metrics", "checkpoints", "hydra", "wandb"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    runs = run_matrix(
        args.seeds,
        args.learning_rates,
        args.update_epochs_grid,
        args.total_timesteps,
    )
    manifest = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "map_name": MAP_NAME,
        "condition": "none",
        "seeds": list(args.seeds),
        "learning_rates": list(args.learning_rates),
        "update_epochs_grid": list(args.update_epochs_grid),
        "budget": args.total_timesteps,
        "num_envs": args.num_envs,
        "num_minibatches": args.num_minibatches,
        "checkpoint_interval": args.checkpoint_interval,
        "project": args.project,
        "runs": [dict(asdict(run), run_name=run.name) for run in runs],
    }
    manifest_path = root / "experiment_manifest.json"
    if manifest_path.is_file():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise RuntimeError(f"Run root has a different frozen protocol: {root}")
    elif not args.dry_run:
        atomic_json(manifest_path, manifest)

    active_processes = subprocess.run(
        ("pgrep", "-af", "baselines/MAPPO/mappo_rnn_smax.py"),
        text=True,
        capture_output=True,
        check=False,
    ).stdout
    pending = [
        run
        for run in runs
        if not completed(root, run) and run.name not in active_processes
    ]
    for run in runs:
        state = (
            "COMPLETE"
            if completed(root, run)
            else "ACTIVE" if run.name in active_processes else "PENDING"
        )
        print(f"{state:8s} {run.name} budget={run.steps:,}", flush=True)
    if args.dry_run or not pending:
        return
    launcher_log = root / "launcher.log"
    log(launcher_log, f"selected={len(runs)} pending={len(pending)}")
    # Clear stale failed statuses before dispatch; the monitor must show queued runs.
    for run in pending:
        atomic_json(
            root / "status" / f"{run.name}.json", status_payload(run, "pending")
        )

    queues = {gpu: collections.deque() for gpu in args.gpus}
    for index, run in enumerate(pending):
        queues[args.gpus[index % len(args.gpus)]].append(run)
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
            for process, _, run, output in running.values():
                process.terminate()
                output.close()
                atomic_json(
                    root / "status" / f"{run.name}.json",
                    status_payload(run, "interrupted"),
                )
            raise SystemExit(130)
        for gpu in args.gpus:
            active = sum(item[1] == gpu for item in running.values())
            while queues[gpu] and active < args.max_runs_per_gpu:
                run = queues[gpu].popleft()
                metrics_path = root / "metrics" / f"{run.name}.jsonl"
                if metrics_path.is_file():
                    retries = root / "retries"
                    retries.mkdir(exist_ok=True)
                    backup = retries / f"{run.name}.{time.time_ns()}.jsonl"
                    metrics_path.rename(backup)
                    log(
                        launcher_log,
                        f"ARCHIVE partial metrics {metrics_path} -> {backup}",
                    )
                output = (root / "logs" / f"{run.name}.log").open("a", encoding="utf-8")
                environment = dict(os.environ)
                environment.pop("LD_LIBRARY_PATH", None)
                environment.update(
                    {
                        "CUDA_VISIBLE_DEVICES": gpu,
                        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                        "HYDRA_FULL_ERROR": "1",
                        "WANDB_DIR": str(root / "wandb"),
                        "WANDB_NAME": run.name,
                        "WANDB_RUN_GROUP": f"SMAX-{MAP_NAME}-nps-none-ppo-sweep",
                        "WANDB_TAGS": (
                            f"smax,{MAP_NAME},nps,none,ppo-sweep,seed-{run.seed},"
                            f"lr-{coef_tag(run.learning_rate)},epochs-{run.update_epochs}"
                        ),
                    }
                )
                process = subprocess.Popen(
                    command(repo, root, args, run),
                    cwd=repo,
                    env=environment,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                )
                running[process.pid] = (process, gpu, run, output)
                atomic_json(
                    root / "status" / f"{run.name}.json",
                    status_payload(run, "running", pid=process.pid, gpu=gpu),
                )
                log(launcher_log, f"GPU {gpu} START {run.name} pid={process.pid}")
                active += 1
        time.sleep(2)
        for pid, (process, gpu, run, output) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            output.close()
            failures += int(code != 0)
            atomic_json(
                root / "status" / f"{run.name}.json",
                status_payload(
                    run,
                    "completed" if code == 0 else "failed",
                    exit_code=code,
                    gpu=gpu,
                ),
            )
            log(launcher_log, f"GPU {gpu} END   {run.name} status={code}")
            del running[pid]
    log(launcher_log, f"matrix finished; failures={failures}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
