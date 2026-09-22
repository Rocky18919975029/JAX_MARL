#!/usr/bin/env python3
"""Launch NPS SMAX MAPPO with critic-only policy-score recovery."""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import math
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


PROTOCOL = "smax-nps-critic-score-recovery-v1.0"
DEFAULT_BUDGETS = {
    "10m_vs_11m": 10_000_000,
    "3s5z_vs_3s6z": 20_000_000,
    "smacv2_10_units": 10_000_000,
}


def csv_items(value: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items or len(items) != len(set(items)):
        raise argparse.ArgumentTypeError("Expected unique comma-separated items")
    return items


def seed_items(value: str) -> tuple[int, ...]:
    if "-" in value and "," not in value:
        start, end = (int(item) for item in value.split("-", 1))
        items = tuple(range(start, end + 1))
    else:
        items = tuple(map(int, csv_items(value)))
    if not items or len(items) != len(set(items)) or min(items) < 0:
        raise argparse.ArgumentTypeError("Seeds must be unique nonnegative integers")
    return items


def coef_tag(value: float) -> str:
    return f"{value:.10g}".replace(".", "p").replace("-", "m")


@dataclass(frozen=True)
class Run:
    map_name: str
    seed: int
    steps: int
    coef: float

    @property
    def name(self) -> str:
        return (
            f"SMAX-REC-{self.map_name}-nps-score_recovery-"
            f"lam{coef_tag(self.coef)}-seed{self.seed}"
        )


def command(repo: Path, root: Path, args, run: Run) -> list[str]:
    return [
        sys.executable,
        str(repo / "baselines/MAPPO/mappo_rnn_smax.py"),
        f"MAP_NAME={run.map_name}",
        f"SEED={run.seed}",
        "ACTOR_PARAMETER_SHARING=false",
        "MATCHED_COMPARISON=true",
        "ALIGN_MODE=none",
        "ALIGN_DISTANCE=ln_mse",
        "ALIGNMENT_COEF=0",
        "ORACLE_LATENT_DISTORTION=false",
        "ORACLE_DISTORTION_COEF=0",
        "SCORE_RECOVERY=true",
        f"SCORE_RECOVERY_COEF={run.coef}",
        f"SCORE_RECOVERY_FISHER_RIDGE={args.fisher_ridge}",
        f"TOTAL_TIMESTEPS={run.steps}",
        f"UPDATE_EPOCHS={args.update_epochs}",
        f"LR={args.learning_rate}",
        f"NUM_ENVS={args.num_envs}",
        f"NUM_MINIBATCHES={args.num_minibatches}",
        "SAVE_CHECKPOINTS=true",
        f"CHECKPOINT_INTERVAL_TIMESTEPS={args.checkpoint_interval}",
        f"CHECKPOINT_DIR={root / 'checkpoints'}",
        "WANDB_UPLOAD_CHECKPOINTS=false",
        f"WANDB_MODE={args.wandb_mode}",
        f"PROJECT={args.project}",
        "EXPERIMENT_CONDITION=score_recovery",
        f"MATRIX_PROFILE={PROTOCOL}",
        f"PROTOCOL_VERSION={PROTOCOL}",
        f"METRICS_JSONL={root / 'metrics' / (run.name + '.jsonl')}",
        f"hydra.run.dir={root / 'hydra' / run.name}",
    ]


def completed(root: Path, run: Run) -> bool:
    return any((root / "checkpoints").glob(f"**/{run.name}-*/final/model.safetensors"))


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def log(path: Path, message: str) -> None:
    line = f"[{dt.datetime.now().astimezone():%Y-%m-%d %H:%M:%S}] {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(line + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--maps", type=csv_items, default=tuple(DEFAULT_BUDGETS))
    parser.add_argument("--seeds", type=seed_items, default=(1, 2, 3, 4))
    parser.add_argument("--score-recovery-coef", type=float, required=True)
    parser.add_argument("--fisher-ridge", type=float, default=1e-3)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=1_000_000)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--gpus", type=csv_items, default=("0", "1", "2", "3"))
    parser.add_argument("--max-runs-per-gpu", type=int, default=1)
    parser.add_argument("--project", default="jaxmarl-smax-score-recovery")
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="online"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if set(args.maps) - set(DEFAULT_BUDGETS):
        parser.error(f"--maps must be drawn from {tuple(DEFAULT_BUDGETS)}")
    if not math.isfinite(args.score_recovery_coef) or args.score_recovery_coef <= 0:
        parser.error("--score-recovery-coef must be finite and positive")
    if not math.isfinite(args.fisher_ridge) or args.fisher_ridge <= 0:
        parser.error("--fisher-ridge must be finite and positive")
    if not args.gpus or args.max_runs_per_gpu <= 0:
        parser.error("At least one GPU and a positive per-GPU limit are required")
    if args.num_envs <= 0 or args.num_minibatches <= 0:
        parser.error("Environment and minibatch counts must be positive")
    if args.num_envs % args.num_minibatches:
        parser.error("--num-envs must be divisible by --num-minibatches")
    if args.total_timesteps is not None and args.total_timesteps <= 0:
        parser.error("--total-timesteps must be positive")
    if args.checkpoint_interval <= 0 or args.update_epochs <= 0:
        parser.error("Checkpoint interval and update epochs must be positive")

    repo = Path(__file__).resolve().parents[1]
    root = args.run_root.expanduser().resolve()
    for name in ("logs", "status", "metrics", "checkpoints", "hydra", "wandb"):
        (root / name).mkdir(parents=True, exist_ok=True)
    budgets = {
        task: args.total_timesteps or DEFAULT_BUDGETS[task] for task in args.maps
    }
    runs = [
        Run(task, seed, budgets[task], args.score_recovery_coef)
        for task in args.maps
        for seed in args.seeds
    ]
    manifest = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "maps": list(args.maps),
        "seeds": list(args.seeds),
        "budgets": budgets,
        "score_recovery_coef": args.score_recovery_coef,
        "fisher_ridge": args.fisher_ridge,
        "update_epochs": args.update_epochs,
        "learning_rate": args.learning_rate,
        "num_envs": args.num_envs,
        "num_minibatches": args.num_minibatches,
        "checkpoint_interval": args.checkpoint_interval,
        "actor_parameterization": "nps",
        "matched_comparison": True,
        "alignment_coef": 0,
        "target": "frozen_rollout_score_with_full_rollout_agent_fisher",
        "optimizer_scope": "critic_encoder_and_recovery_head_only",
    }
    manifest_path = root / "experiment_manifest.json"
    if manifest_path.is_file():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise RuntimeError(f"Run root has a different frozen protocol: {root}")
    else:
        atomic_json(manifest_path, manifest)

    process_text = subprocess.run(
        ("pgrep", "-af", "baselines/MAPPO/mappo_rnn_smax.py"),
        text=True,
        capture_output=True,
    ).stdout
    pending = [
        run for run in runs if not completed(root, run) and run.name not in process_text
    ]
    launcher_log = root / "launcher.log"
    log(launcher_log, f"selected={len(runs)} pending={len(pending)}")
    for run in runs:
        state = (
            "COMPLETE"
            if completed(root, run)
            else "ACTIVE" if run.name in process_text else "PENDING"
        )
        print(f"{state:8s} {run.name} budget={run.steps:,}", flush=True)
    if args.dry_run or not pending:
        return

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
            for process, _, _, output in running.values():
                process.terminate()
                output.close()
            raise SystemExit(130)
        for gpu in args.gpus:
            active = sum(item[1] == gpu for item in running.values())
            while queues[gpu] and active < args.max_runs_per_gpu:
                run = queues[gpu].popleft()
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
                        "WANDB_RUN_GROUP": f"SMAX-{run.map_name}-nps-score-recovery",
                        "WANDB_TAGS": f"smax,{run.map_name},nps,score-recovery,seed-{run.seed}",
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
                    {
                        "status": "running",
                        "pid": process.pid,
                        "gpu": gpu,
                        "run_name": run.name,
                        "map_name": run.map_name,
                        "seed": run.seed,
                        "budget": run.steps,
                    },
                )
                log(launcher_log, f"GPU {gpu} START {run.name} pid={process.pid}")
                active += 1
        time.sleep(2)
        for pid, (process, gpu, run, output) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            output.close()
            status = "completed" if code == 0 else "failed"
            failures += int(code != 0)
            atomic_json(
                root / "status" / f"{run.name}.json",
                {
                    "status": status,
                    "exit_code": code,
                    "gpu": gpu,
                    "run_name": run.name,
                    "map_name": run.map_name,
                    "seed": run.seed,
                    "budget": run.steps,
                },
            )
            log(launcher_log, f"GPU {gpu} END   {run.name} status={code}")
            del running[pid]
    log(launcher_log, f"matrix finished; failures={failures}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
