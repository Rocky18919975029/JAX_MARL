#!/usr/bin/env python3
"""Train one NPS MAPPO condition on five-agent MPE Simple Spread."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import jax
from omegaconf import OmegaConf

import wandb
from baselines.MAPPO.mappo_rnn_mpe import make_train
from experiments.mpe_alignment.protocol import PROTOCOL_VERSION, TASKS
from jaxmarl.wrappers.baselines import save_params


OFFICIAL_CONFIG = (
    REPO_ROOT / "baselines" / "MAPPO" / "config" / "mappo_homogenous_rnn_mpe.yaml"
)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--condition", choices=("none", "c_to_a_mse", "c_to_a_cka"), required=True
    )
    parser.add_argument("--align-mode", choices=("none", "c_to_a"), required=True)
    parser.add_argument(
        "--align-distance", choices=("ln_mse", "linear_cka"), required=True
    )
    parser.add_argument("--alignment-coef", type=float, required=True)
    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--wandb-project", default="jaxmarl-mpe-spread5-alignment")
    parser.add_argument("--wandb-group", default="simple_spread_5-nps")
    parser.add_argument("--protocol-version", default=PROTOCOL_VERSION)
    parser.add_argument("--sweep-base-coefficient", type=float)
    parser.add_argument("--sweep-multiplier", type=float)
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="online"
    )
    return parser.parse_args()


def validate(args: argparse.Namespace) -> None:
    expected_mode = "none" if args.condition == "none" else "c_to_a"
    expected_distance = "linear_cka" if args.condition == "c_to_a_cka" else "ln_mse"
    if (args.align_mode, args.align_distance) != (expected_mode, expected_distance):
        raise ValueError("condition, direction, and distance are inconsistent")
    if args.condition == "none" and args.alignment_coef != 0:
        raise ValueError("isolated runs must have coefficient zero")
    if args.condition != "none" and args.alignment_coef <= 0:
        raise ValueError("aligned runs require a positive coefficient")
    if args.num_envs % args.num_minibatches:
        raise ValueError("num-envs must be divisible by num-minibatches")
    if args.total_timesteps < args.num_envs * args.num_steps:
        raise ValueError("total-timesteps must contain at least one rollout")
    sweep_values = (args.sweep_base_coefficient, args.sweep_multiplier)
    if any(value is not None for value in sweep_values):
        if args.condition != "c_to_a_cka" or not all(
            value is not None and value > 0 for value in sweep_values
        ):
            raise ValueError(
                "sweep metadata requires positive base/multiplier on CKA runs"
            )


def build_config(args: argparse.Namespace, status_path: Path, metadata: dict) -> dict:
    config = OmegaConf.to_container(OmegaConf.load(OFFICIAL_CONFIG), resolve=True)
    config.update(
        SEED=args.seed,
        NUM_SEEDS=1,
        ENV_NAME="MPE_simple_spread_v3",
        ENV_KWARGS={"num_agents": 5, "num_landmarks": 5},
        TOTAL_TIMESTEPS=int(args.total_timesteps),
        NUM_ENVS=args.num_envs,
        NUM_STEPS=args.num_steps,
        UPDATE_EPOCHS=args.update_epochs,
        NUM_MINIBATCHES=args.num_minibatches,
        LR=args.learning_rate,
        ACTOR_PARAMETER_SHARING=False,
        MATCHED_COMPARISON=True,
        ALIGN_MODE=args.align_mode,
        ALIGN_DISTANCE=args.align_distance,
        ALIGNMENT_COEF=args.alignment_coef,
        ALIGN_DISTANCE_EPS=1e-8,
        EXPERIMENT_CONDITION=args.condition,
        PROTOCOL_VERSION=args.protocol_version,
        HYPERPARAMETER_SWEEP=args.sweep_multiplier is not None,
        CKA_SWEEP_BASE_COEFFICIENT=args.sweep_base_coefficient,
        CKA_COEFFICIENT_MULTIPLIER=args.sweep_multiplier,
        WANDB_MODE=args.wandb_mode,
        PROJECT=args.wandb_project,
        METRICS_JSONL=str(args.run_root / "metrics" / f"{args.run_name}.jsonl"),
        STATUS_JSON=str(status_path),
        STATUS_METADATA=metadata,
    )
    return config


def main() -> None:
    args = parse_args()
    validate(args)
    args.run_root = args.run_root.expanduser().resolve()
    status_path = args.run_root / "status" / f"{args.run_name}.json"
    metadata = {
        "schema_version": 1,
        "protocol_version": args.protocol_version,
        "run_name": args.run_name,
        "task": args.task,
        "environment": "MPE_simple_spread_v3",
        "num_agents": 5,
        "num_landmarks": 5,
        "seed": args.seed,
        "actor_parameterization": "nps",
        "centralized_critic": True,
        "condition": args.condition,
        "align_mode": args.align_mode,
        "align_distance": args.align_distance,
        "alignment_coefficient": args.alignment_coef,
        "hyperparameter_sweep": args.sweep_multiplier is not None,
        "cka_sweep_base_coefficient": args.sweep_base_coefficient,
        "cka_coefficient_multiplier": args.sweep_multiplier,
        "total_timesteps": args.total_timesteps,
        "official_config": str(OFFICIAL_CONFIG.relative_to(REPO_ROOT)),
        "git_commit": git_commit(),
    }
    atomic_json(status_path, {**metadata, "status": "initializing", "env_steps": 0})
    config = build_config(args, status_path, metadata)
    checkpoint = args.run_root / "checkpoints" / args.run_name / "final"
    checkpoint.mkdir(parents=True, exist_ok=True)

    try:
        run = wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            group=args.wandb_group,
            tags=[
                "MPE",
                "MAPPO",
                "NPS",
                "simple_spread_5",
                args.condition,
                args.protocol_version,
            ],
            config=config,
            mode=args.wandb_mode,
        )
        wandb.define_metric("env_step")
        wandb.define_metric("returns", step_metric="env_step")
        wandb.define_metric("*", step_metric="env_step")
        train = jax.jit(make_train(config))
        output = train(jax.random.PRNGKey(args.seed))
        model = checkpoint / "model.safetensors"
        temporary_model = checkpoint / ".model.tmp.safetensors"
        save_params(
            {"actor": output["actor_params"], "critic": output["critic_params"]},
            temporary_model,
        )
        os.replace(temporary_model, model)
        atomic_json(checkpoint / "config.json", config)
        atomic_json(checkpoint / "metadata.json", metadata)
        atomic_json(checkpoint.parent / "completed.json", metadata)
        atomic_json(
            status_path,
            {**metadata, "status": "completed", "env_steps": args.total_timesteps},
        )
        run.finish()
    except Exception as error:
        atomic_json(
            status_path,
            {
                **metadata,
                "status": "failed",
                "env_steps": 0,
                "error": repr(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    main()
