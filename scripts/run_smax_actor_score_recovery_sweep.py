#!/usr/bin/env python3
"""Run isolated MAPPO and an actor-score-recovery hyperparameter grid."""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import math
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path

try:
    from scripts.run_smax_score_recovery_training import (
        atomic_json,
        coef_tag,
        csv_items,
        log,
        seed_items,
    )
    from scripts.run_smax_actor_score_recovery_training import DEFAULT_BUDGETS
except ModuleNotFoundError:  # Direct execution from scripts/.
    from run_smax_score_recovery_training import (
        atomic_json,
        coef_tag,
        csv_items,
        log,
        seed_items,
    )
    from run_smax_actor_score_recovery_training import DEFAULT_BUDGETS


PROTOCOL = "smax-nps-actor-score-recovery-sweep-v1.0"


def hydra_float(value: float) -> str:
    """Avoid scientific-notation ambiguity in Hydra command-line overrides."""
    return format(Decimal(str(value)), "f")


def positive_float_items(raw: str) -> tuple[float, ...]:
    values = tuple(float(value) for value in csv_items(raw))
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise argparse.ArgumentTypeError("Values must be finite and positive")
    if len({coef_tag(value) for value in values}) != len(values):
        raise argparse.ArgumentTypeError("Values must have distinct run-name tags")
    return values


def positive_int_items(raw: str) -> tuple[int, ...]:
    values = tuple(int(value) for value in csv_items(raw))
    if any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("Values must be positive integers")
    return values


@dataclass(frozen=True)
class SweepRun:
    map_name: str
    seed: int
    steps: int
    condition: str
    coef: float = 0.0
    q_steps: int = 8
    q_learning_rate: float = 1e-3
    fisher_ridge: float = 1e-3

    @property
    def name(self) -> str:
        prefix = f"SMAX-AREC-SWEEP-{self.map_name}-nps"
        if self.condition == "none":
            return f"{prefix}-none-seed{self.seed}"
        return (
            f"{prefix}-actor_score_recovery-lam{coef_tag(self.coef)}"
            f"-qs{self.q_steps}-qlr{coef_tag(self.q_learning_rate)}"
            f"-ridge{coef_tag(self.fisher_ridge)}-seed{self.seed}"
        )


def run_matrix(
    maps,
    seeds,
    budgets,
    coefs,
    q_steps,
    q_learning_rates,
    ridges,
    conditions=("none", "actor_score_recovery"),
):
    runs = []
    for map_name in maps:
        for seed in seeds:
            if "none" in conditions:
                runs.append(SweepRun(map_name, seed, budgets[map_name], "none"))
            if "actor_score_recovery" in conditions:
                for coef, steps, learning_rate, ridge in itertools.product(
                    coefs, q_steps, q_learning_rates, ridges
                ):
                    runs.append(
                        SweepRun(
                            map_name,
                            seed,
                            budgets[map_name],
                            "actor_score_recovery",
                            coef,
                            steps,
                            learning_rate,
                            ridge,
                        )
                    )
    if len({run.name for run in runs}) != len(runs):
        raise ValueError("Sweep grid produces duplicate run names")
    return runs


def command(repo: Path, root: Path, args, run: SweepRun) -> list[str]:
    enabled = run.condition == "actor_score_recovery"
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
        "ALIGN_GRADIENT_CALIBRATION=false",
        "ALIGN_TARGET_SHUFFLE=false",
        "ORACLE_LATENT_DISTORTION=false",
        "ORACLE_DISTORTION_COEF=0",
        "SCORE_RECOVERY=false",
        "SCORE_RECOVERY_COEF=0",
        f"ACTOR_SCORE_RECOVERY={'true' if enabled else 'false'}",
        f"ACTOR_SCORE_RECOVERY_COEF={hydra_float(run.coef) if enabled else '0'}",
        f"ACTOR_SCORE_RECOVERY_FISHER_RIDGE={hydra_float(run.fisher_ridge)}",
        f"ACTOR_SCORE_RECOVERY_Q_LR={hydra_float(run.q_learning_rate)}",
        f"ACTOR_SCORE_RECOVERY_Q_STEPS={run.q_steps}",
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
        f"EXPERIMENT_CONDITION={run.condition}",
        f"MATRIX_PROFILE={PROTOCOL}",
        f"PROTOCOL_VERSION={PROTOCOL}",
        f"METRICS_JSONL={root / 'metrics' / (run.name + '.jsonl')}",
        f"hydra.run.dir={root / 'hydra' / run.name}",
    ]


def completed(root: Path, run: SweepRun) -> bool:
    return any(
        (root / "checkpoints").glob(f"**/{run.name}-*/final/model.safetensors")
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--maps", type=csv_items, default=("10m_vs_11m", "3s5z_vs_3s6z")
    )
    parser.add_argument("--seeds", type=seed_items, default=(9001, 9002))
    parser.add_argument(
        "--conditions", type=csv_items, default=("none", "actor_score_recovery")
    )
    parser.add_argument(
        "--coefs", type=positive_float_items, default=(3e-5, 1e-4, 3e-4)
    )
    parser.add_argument("--q-steps-grid", type=positive_int_items, default=(4, 8))
    parser.add_argument(
        "--q-learning-rates", type=positive_float_items, default=(1e-3,)
    )
    parser.add_argument(
        "--fisher-ridges", type=positive_float_items, default=(1e-3,)
    )
    parser.add_argument("--budget-fraction", type=float, default=1.0)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=1_000_000)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--gpus", type=csv_items, default=("0", "1", "2", "3"))
    parser.add_argument("--max-runs-per-gpu", type=int, default=1)
    parser.add_argument("--project", default="jaxmarl-smax-actor-score-recovery")
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="online"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if set(args.maps) - set(DEFAULT_BUDGETS):
        parser.error(f"--maps must be drawn from {tuple(DEFAULT_BUDGETS)}")
    if set(args.conditions) - {"none", "actor_score_recovery"}:
        parser.error("--conditions must be none and/or actor_score_recovery")
    if not math.isfinite(args.budget_fraction) or not 0 < args.budget_fraction <= 1:
        parser.error("--budget-fraction must lie in (0, 1]")
    if args.total_timesteps is not None and args.total_timesteps <= 0:
        parser.error("--total-timesteps must be positive")
    if args.total_timesteps is not None and args.budget_fraction != 1.0:
        parser.error("Choose --total-timesteps or --budget-fraction, not both")
    if args.update_epochs <= 0 or args.learning_rate <= 0:
        parser.error("PPO epochs and learning rate must be positive")
    if args.num_envs <= 0 or args.num_minibatches <= 0:
        parser.error("Environment and minibatch counts must be positive")
    if args.num_envs % args.num_minibatches:
        parser.error("NUM_ENVS must be divisible by NUM_MINIBATCHES")
    if args.checkpoint_interval <= 0 or args.max_runs_per_gpu <= 0:
        parser.error("Checkpoint interval and concurrency must be positive")
    if not args.gpus:
        parser.error("At least one GPU is required")
    return args


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    root = args.run_root.expanduser().resolve()
    for directory in ("logs", "status", "metrics", "checkpoints", "hydra", "wandb"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    budgets = {
        name: args.total_timesteps
        or int(DEFAULT_BUDGETS[name] * args.budget_fraction)
        for name in args.maps
    }
    runs = run_matrix(
        args.maps,
        args.seeds,
        budgets,
        args.coefs,
        args.q_steps_grid,
        args.q_learning_rates,
        args.fisher_ridges,
        args.conditions,
    )
    manifest = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "maps": list(args.maps),
        "conditions": list(args.conditions),
        "seeds": list(args.seeds),
        "budgets": budgets,
        "budget_fraction": args.budget_fraction,
        "total_timesteps_override": args.total_timesteps,
        "coefs": list(args.coefs),
        "q_steps_grid": list(args.q_steps_grid),
        "q_learning_rates": list(args.q_learning_rates),
        "fisher_ridges": list(args.fisher_ridges),
        "update_epochs": args.update_epochs,
        "learning_rate": args.learning_rate,
        "num_envs": args.num_envs,
        "num_minibatches": args.num_minibatches,
        "checkpoint_interval": args.checkpoint_interval,
        "actor_parameterization": "nps",
        "project": args.project,
        "runs": [dict(asdict(run), run_name=run.name) for run in runs],
    }
    manifest_path = root / "experiment_manifest.json"
    if manifest_path.is_file():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise RuntimeError(f"Run root has a different frozen protocol: {root}")
    else:
        atomic_json(manifest_path, manifest)

    active_processes = subprocess.run(
        ("pgrep", "-af", "baselines/MAPPO/mappo_rnn_smax.py"),
        text=True,
        capture_output=True,
    ).stdout
    pending = [
        run
        for run in runs
        if not completed(root, run) and run.name not in active_processes
    ]
    launcher_log = root / "launcher.log"
    log(launcher_log, f"selected={len(runs)} pending={len(pending)}")
    for run in runs:
        state = (
            "COMPLETE"
            if completed(root, run)
            else "ACTIVE" if run.name in active_processes else "PENDING"
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
                output = (root / "logs" / f"{run.name}.log").open(
                    "a", encoding="utf-8"
                )
                environment = dict(os.environ)
                environment.pop("LD_LIBRARY_PATH", None)
                environment.update(
                    {
                        "CUDA_VISIBLE_DEVICES": gpu,
                        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                        "HYDRA_FULL_ERROR": "1",
                        "WANDB_DIR": str(root / "wandb"),
                        "WANDB_NAME": run.name,
                        "WANDB_RUN_GROUP": f"SMAX-{run.map_name}-nps-arec-sweep",
                        "WANDB_TAGS": (
                            f"smax,{run.map_name},nps,arec-sweep,"
                            f"{run.condition},seed-{run.seed}"
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
                    {
                        "status": "running",
                        "pid": process.pid,
                        "gpu": gpu,
                        "run_name": run.name,
                        "map_name": run.map_name,
                        "condition": run.condition,
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
            failures += int(code != 0)
            atomic_json(
                root / "status" / f"{run.name}.json",
                {
                    "status": "completed" if code == 0 else "failed",
                    "exit_code": code,
                    "gpu": gpu,
                    "run_name": run.name,
                    "map_name": run.map_name,
                    "condition": run.condition,
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
