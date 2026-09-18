#!/usr/bin/env python3
"""Train aligned HARL MAPPO on MA-MuJoCo Humanoid-v2-17x1."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HARL_ROOT = REPO_ROOT / "third_party" / "HARL"
TUNED_CONFIG = Path("tuned_configs/mamujoco/Humanoid-v2-17x1/mappo/config.json")


def parse_bool(value: str) -> bool:
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}")


def git_commit(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def lambda_label(value: float) -> str:
    return f"{value:.10g}".replace("-", "m").replace(".", "p")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_configuration(args):
    config_path = (
        args.config.expanduser().resolve()
        if args.config is not None
        else args.harl_root / TUNED_CONFIG
    )
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    main_args = copy.deepcopy(payload["main_args"])
    algo_args = copy.deepcopy(payload["algo_args"])
    env_args = copy.deepcopy(payload["env_args"])

    if main_args["algo"] != "mappo" or main_args["env"] != "mamujoco":
        raise ValueError("Configuration must be HARL MAPPO on mamujoco")
    if (
        env_args.get("scenario") != "Humanoid-v2"
        or env_args.get("agent_conf") != "17x1"
    ):
        raise ValueError("Configuration must be Humanoid-v2-17x1")

    main_args["exp_name"] = args.run_name
    main_args["load_config"] = str(config_path)
    algo_args["seed"]["seed_specify"] = True
    algo_args["seed"]["seed"] = args.seed
    algo_args["algo"]["share_param"] = args.actor_parameter_sharing
    algo_args["logger"]["log_dir"] = str(args.run_root / "harl_results")
    if args.num_env_steps is not None:
        algo_args["train"]["num_env_steps"] = args.num_env_steps
    if args.n_rollout_threads is not None:
        algo_args["train"]["n_rollout_threads"] = args.n_rollout_threads
    if args.episode_length is not None:
        algo_args["train"]["episode_length"] = args.episode_length
    if args.disable_eval or args.calibration_output is not None:
        algo_args["eval"]["use_eval"] = False
    if args.calibration_output is not None:
        algo_args["train"]["num_env_steps"] = (
            algo_args["train"]["episode_length"]
            * algo_args["train"]["n_rollout_threads"]
        )
        algo_args["train"]["use_linear_lr_decay"] = False
    return main_args, algo_args, env_args, config_path


def collect_one_rollout(runner) -> None:
    runner.warmup()
    runner.prep_rollout()
    for step in range(runner.algo_args["train"]["episode_length"]):
        values, actions, action_log_probs, rnn_states, rnn_states_critic = (
            runner.collect(step)
        )
        obs, share_obs, rewards, dones, infos, available_actions = runner.envs.step(
            actions
        )
        runner.insert(
            (
                obs,
                share_obs,
                rewards,
                dones,
                infos,
                available_actions,
                values,
                actions,
                action_log_probs,
                rnn_states,
                rnn_states_critic,
            )
        )
    runner.compute()
    runner.prep_training()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harl-root", type=Path, default=DEFAULT_HARL_ROOT)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--actor-parameter-sharing", type=parse_bool, required=True)
    parser.add_argument(
        "--align-mode", choices=("none", "c_to_a", "a_to_c", "joint"), required=True
    )
    parser.add_argument(
        "--align-distance",
        choices=("ln_mse", "linear_cka", "containment"),
        default="ln_mse",
    )
    parser.add_argument("--alignment-coef", type=float, default=0.1)
    parser.add_argument("--alignment-epsilon", type=float, default=1e-8)
    parser.add_argument("--containment-ridge-ratio", type=float, default=1e-3)
    parser.add_argument("--containment-epsilon", type=float, default=1e-6)
    parser.add_argument("--checkpoint-interval-steps", type=int, default=500_000)
    parser.add_argument("--wandb-project", default="harl-mamujoco-alignment")
    parser.add_argument("--wandb-entity")
    parser.add_argument(
        "--wandb-mode", choices=("online", "disabled"), default="online"
    )
    parser.add_argument("--wandb-upload-checkpoints", action="store_true")
    parser.add_argument("--protocol-version", default="harl-mamujoco-alignment-v1.0")
    parser.add_argument("--num-env-steps", type=int)
    parser.add_argument("--n-rollout-threads", type=int)
    parser.add_argument("--episode-length", type=int)
    parser.add_argument("--disable-eval", action="store_true")
    parser.add_argument(
        "--calibration-output",
        type=Path,
        help="collect one initial rollout, save gradient scales, and exit without training",
    )
    args = parser.parse_args()

    args.harl_root = args.harl_root.expanduser().resolve()
    args.run_root = args.run_root.expanduser().resolve()
    if not (args.harl_root / "harl").is_dir():
        raise FileNotFoundError(
            f"HARL is missing at {args.harl_root}; run git submodule update --init third_party/HARL"
        )
    sys.path.insert(0, str(args.harl_root))
    sys.path.insert(0, str(REPO_ROOT))

    actor_label = "ps" if args.actor_parameter_sharing else "nps"
    distance_suffix = {
        "ln_mse": "mse",
        "linear_cka": "cka",
        "containment": "dsc",
    }[args.align_distance]
    condition = (
        "none" if args.align_mode == "none" else f"{args.align_mode}_{distance_suffix}"
    )
    if args.run_name is None:
        args.run_name = (
            f"HARL-Humanoid-v2-17x1-{actor_label}-{condition}-"
            f"lam{lambda_label(args.alignment_coef)}-seed{args.seed}"
        )
    args.checkpoint_root = (
        args.checkpoint_root.expanduser().resolve()
        if args.checkpoint_root is not None
        else args.run_root / "checkpoints"
    )
    args.run_root.mkdir(parents=True, exist_ok=True)
    args.checkpoint_root.mkdir(parents=True, exist_ok=True)

    main_args, algo_args, env_args, config_path = load_configuration(args)
    parent_commit = git_commit(REPO_ROOT)
    harl_commit = git_commit(args.harl_root)
    status_path = args.run_root / "status" / f"{args.run_name}.json"
    experiment = {
        "protocol_version": args.protocol_version,
        "run_name": args.run_name,
        "task": "Humanoid-v2-17x1",
        "seed": args.seed,
        "actor_parameter_sharing": args.actor_parameter_sharing,
        "actor_parameterization": actor_label,
        "centralized_critic": True,
        "align_mode": args.align_mode,
        "align_distance": args.align_distance,
        "alignment_coef": args.alignment_coef,
        "alignment_epsilon": args.alignment_epsilon,
        "containment_ridge_ratio": args.containment_ridge_ratio,
        "containment_epsilon": args.containment_epsilon,
        "matched_update": True,
        "checkpoint_root": str(args.checkpoint_root),
        "status_path": str(status_path),
        "checkpoint_interval_steps": args.checkpoint_interval_steps,
        "wandb_upload_checkpoints": args.wandb_upload_checkpoints,
        "source_config": str(config_path),
        "jaxmarl_git_commit": parent_commit,
        "harl_git_commit": harl_commit,
        "algo_args": algo_args,
        "env_args": env_args,
    }

    wandb_run = None
    if args.wandb_mode == "online" and args.calibration_output is None:
        import wandb

        wandb_dir = args.run_root / "wandb"
        wandb_dir.mkdir(parents=True, exist_ok=True)
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            group=f"Humanoid-v2-17x1-{actor_label}",
            job_type="mappo-alignment",
            tags=["HARL", "MA-MuJoCo", "Humanoid-v2-17x1", actor_label, condition],
            config=experiment,
            mode="online",
            dir=str(wandb_dir),
        )
        wandb_run.define_metric("env_step")
        wandb_run.define_metric("train/*", step_metric="env_step")
        wandb_run.define_metric("eval/*", step_metric="env_step")

    from experiments.harl_mamujoco.runner import AlignedMAMuJoCoRunner

    write_json(
        status_path,
        {
            "status": "running",
            "run_name": args.run_name,
            "env_steps": 0,
            "total_env_steps": int(algo_args["train"]["num_env_steps"]),
            "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "pid": os.getpid(),
            **{
                key: experiment[key]
                for key in (
                    "task",
                    "seed",
                    "actor_parameterization",
                    "align_mode",
                    "align_distance",
                    "alignment_coef",
                    "protocol_version",
                )
            },
        },
    )
    runner = None
    try:
        runner = AlignedMAMuJoCoRunner(
            main_args, algo_args, env_args, experiment, wandb_run=wandb_run
        )
        if args.calibration_output is not None:
            if args.align_mode not in {"c_to_a", "a_to_c"}:
                raise ValueError("Calibration requires c_to_a or a_to_c")
            collect_one_rollout(runner)
            metrics = runner.calibration_metrics()
            write_json(
                args.calibration_output.expanduser().resolve(),
                {**experiment, **metrics, "selection_uses_return": False},
            )
            write_json(
                status_path,
                {
                    "status": "completed",
                    "run_name": args.run_name,
                    "env_steps": 0,
                    "total_env_steps": 0,
                },
            )
        else:
            runner.run()
            write_json(
                status_path,
                {
                    "status": "completed",
                    "run_name": args.run_name,
                    "env_steps": int(algo_args["train"]["num_env_steps"]),
                    "total_env_steps": int(algo_args["train"]["num_env_steps"]),
                },
            )
    except BaseException as error:
        write_json(
            status_path,
            {
                "status": "failed",
                "run_name": args.run_name,
                "error": repr(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    finally:
        if runner is not None:
            runner.close()
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
