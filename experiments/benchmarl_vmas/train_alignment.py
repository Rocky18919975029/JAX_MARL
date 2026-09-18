#!/usr/bin/env python3
"""Train NPS MAPPO with optional C→A alignment on an official VMAS task."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.benchmarl_vmas.protocol import PROTOCOL_VERSION, TASKS
from benchmarl.experiment.callback import Callback


TASK_ENUM_NAMES = {
    "discovery": "DISCOVERY",
    "passage": "PASSAGE",
    "football": "FOOTBALL",
}


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def package_versions() -> dict[str, str]:
    versions = {}
    for package in ("benchmarl", "torch", "torchrl", "tensordict", "vmas"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unknown"
    return versions


class StatusCallback(Callback):
    """Persist restart-safe progress without changing BenchMARL training."""

    def __init__(self, status_path: Path, metadata: dict):
        super().__init__()
        self.status_path = status_path
        self.metadata = metadata

    def on_setup(self):
        groups = self.experiment.group_map
        if len(groups) != 1 or not next(iter(groups.values())):
            raise RuntimeError(
                f"expected exactly one non-empty learning group, got {groups}"
            )
        atomic_json(
            self.status_path,
            {
                **self.metadata,
                "status": "running",
                "env_steps": 0,
                "benchmarl_output": str(self.experiment.folder_name),
            },
        )

    def on_batch_collected(self, batch):
        for loss_module in self.experiment.losses.values():
            request = getattr(loss_module, "request_gradient_audit", None)
            if request is not None:
                request()
        atomic_json(
            self.status_path,
            {
                **self.metadata,
                "status": "running",
                "env_steps": self.experiment.total_frames,
                "benchmarl_output": str(self.experiment.folder_name),
            },
        )


def build_task(task_name: str):
    from benchmarl.environments import VmasTask

    # Deliberately return the official YAML unchanged.  In particular, do not
    # rewrite agent counts to manufacture a five-agent benchmark.
    return getattr(VmasTask, TASK_ENUM_NAMES[task_name]).get_from_yaml()


def build_experiment(args, callbacks=None):
    from benchmarl.algorithms import MappoConfig
    from benchmarl.experiment import Experiment, ExperimentConfig

    from experiments.benchmarl_vmas.algorithm import AlignmentMappoConfig
    from experiments.benchmarl_vmas.model import AlignmentMlpConfig

    base_algorithm = MappoConfig.get_from_yaml()
    algorithm = AlignmentMappoConfig(
        **base_algorithm.__dict__,
        align_mode=args.align_mode,
        align_distance=args.align_distance,
        alignment_coef=args.alignment_coef,
        alignment_epsilon=args.alignment_epsilon,
    )
    config = ExperimentConfig.get_from_yaml()
    # Official BenchMARL fine_tuned/vmas/conf/config.yaml values.
    config.sampling_device = args.device
    config.train_device = args.device
    config.buffer_device = args.device
    config.share_policy_params = False
    config.prefer_continuous_actions = True
    config.gamma = 0.9
    config.lr = 5e-5
    config.clip_grad_norm = True
    config.clip_grad_val = 5
    config.exploration_anneal_frames = 1_000_000
    config.max_n_iters = None
    config.max_n_frames = args.max_frames
    config.on_policy_collected_frames_per_batch = args.frames_per_batch
    config.on_policy_n_envs_per_worker = args.num_envs
    config.on_policy_n_minibatch_iters = args.minibatch_iters
    config.on_policy_minibatch_size = args.minibatch_size
    config.evaluation = not args.disable_evaluation
    config.render = False
    config.evaluation_interval = args.evaluation_interval
    config.evaluation_episodes = args.evaluation_episodes
    config.loggers = (
        [] if args.disable_logging or args.wandb_mode == "disabled" else ["wandb"]
    )
    config.create_json = True
    config.project_name = args.wandb_project
    tuning_tags = []
    if args.cka_multiplier is not None:
        tuning_tags.extend(
            (
                f"cka-base-{args.cka_calibration_coef:.10g}",
                f"cka-multiplier-{args.cka_multiplier:.10g}",
            )
        )
    config.wandb_extra_kwargs = {
        "name": args.run_name,
        "group": f"{args.task}-nps",
        "tags": [
            "VMAS",
            "MAPPO",
            "NPS",
            args.task,
            args.condition,
            PROTOCOL_VERSION,
            args.experiment_stage,
            *tuning_tags,
        ],
        "mode": args.wandb_mode,
    }
    save_folder = args.run_root / "benchmarl_runs" / args.run_name
    save_folder.mkdir(parents=True, exist_ok=True)
    config.save_folder = str(save_folder)
    config.restore_file = None
    config.checkpoint_interval = args.checkpoint_interval
    config.checkpoint_at_end = True
    config.keep_checkpoints_num = None
    config.exclude_buffer_from_checkpoint = True

    model = AlignmentMlpConfig(hidden_sizes=(256, 256))
    critic_model = AlignmentMlpConfig(hidden_sizes=(256, 256))
    return Experiment(
        task=build_task(args.task),
        algorithm_config=algorithm,
        model_config=model,
        critic_model_config=critic_model,
        seed=args.seed,
        config=config,
        callbacks=callbacks,
    )


def parse_args():
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
    parser.add_argument("--alignment-epsilon", type=float, default=1e-8)
    parser.add_argument("--cka-calibration-coef", type=float)
    parser.add_argument("--cka-multiplier", type=float)
    parser.add_argument(
        "--experiment-stage",
        choices=("formal", "cka_tuning", "cka_validation"),
        default="formal",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-frames", type=int, default=10_000_000)
    parser.add_argument("--frames-per-batch", type=int, default=60_000)
    parser.add_argument("--num-envs", type=int, default=600)
    parser.add_argument("--minibatch-iters", type=int, default=45)
    parser.add_argument("--minibatch-size", type=int, default=4096)
    parser.add_argument("--evaluation-interval", type=int, default=120_000)
    parser.add_argument("--evaluation-episodes", type=int, default=200)
    parser.add_argument("--checkpoint-interval", type=int, default=600_000)
    parser.add_argument("--wandb-project", default="benchmarl-vmas-nps-alignment-v2")
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="online"
    )
    parser.add_argument("--disable-evaluation", action="store_true")
    parser.add_argument("--disable-logging", action="store_true")
    return parser.parse_args()


def validate_args(args) -> None:
    expected_mode = "none" if args.condition == "none" else "c_to_a"
    expected_distance = "linear_cka" if args.condition == "c_to_a_cka" else "ln_mse"
    if args.align_mode != expected_mode or args.align_distance != expected_distance:
        raise ValueError("condition, align mode, and distance are inconsistent")
    if args.condition == "none" and args.alignment_coef != 0:
        raise ValueError("isolated runs must use coefficient 0")
    if args.condition != "none" and args.alignment_coef <= 0:
        raise ValueError("aligned runs require a positive coefficient")
    tuning_values = (args.cka_calibration_coef, args.cka_multiplier)
    if any(value is not None for value in tuning_values):
        if args.condition != "c_to_a_cka" or any(
            value is None or value <= 0 for value in tuning_values
        ):
            raise ValueError(
                "CKA calibration coefficient and multiplier require a CKA run"
            )
        expected_coefficient = args.cka_calibration_coef * args.cka_multiplier
        if not math.isclose(
            args.alignment_coef, expected_coefficient, rel_tol=1e-9, abs_tol=1e-12
        ):
            raise ValueError("alignment coefficient does not match base × multiplier")
    if args.frames_per_batch % args.num_envs:
        raise ValueError("frames-per-batch must be divisible by num-envs")
    for value in (args.evaluation_interval, args.checkpoint_interval):
        if value and value % args.frames_per_batch:
            raise ValueError("evaluation/checkpoint intervals must be batch multiples")


def main() -> None:
    args = parse_args()
    args.run_root = args.run_root.expanduser().resolve()
    validate_args(args)
    status_path = args.run_root / "status" / f"{args.run_name}.json"
    metadata = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "run_name": args.run_name,
        "task": args.task,
        "seed": args.seed,
        "actor_parameterization": "nps",
        "share_policy_params": False,
        "centralized_critic": True,
        "condition": args.condition,
        "align_mode": args.align_mode,
        "align_distance": args.align_distance,
        "alignment_coef": args.alignment_coef,
        "cka_calibration_coef": args.cka_calibration_coef,
        "cka_multiplier": args.cka_multiplier,
        "experiment_stage": args.experiment_stage,
        "git_commit": git_commit(),
        "package_versions": package_versions(),
        "max_frames": args.max_frames,
    }
    atomic_json(status_path, {**metadata, "status": "initializing", "env_steps": 0})

    try:
        experiment = build_experiment(
            args, callbacks=[StatusCallback(status_path, metadata)]
        )
        output = experiment.folder_name
        atomic_json(output / "protocol_metadata.json", metadata)
        experiment.run()
        atomic_json(
            output / "completed.json",
            {**metadata, "env_steps": experiment.total_frames},
        )
        atomic_json(
            status_path,
            {
                **metadata,
                "status": "completed",
                "env_steps": experiment.total_frames,
                "benchmarl_output": str(output),
            },
        )
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
