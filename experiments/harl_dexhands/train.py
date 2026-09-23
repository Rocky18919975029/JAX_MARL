#!/usr/bin/env python3
"""Train one matched HAPPO, MAPPO, or MADPO ShadowHandOver run."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HARL_ROOT = REPO_ROOT / "third_party" / "HARL"
DEFAULT_CONFIG = Path("tuned_configs/dexhands/ShadowHandOver/happo/config.json")
MADPO_REFERENCE_COMMIT = "0596056312f52a3183676c471ffdd4638b59d63c"
sys.path.insert(0, str(REPO_ROOT))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _commit(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harl-root", type=Path, default=DEFAULT_HARL_ROOT)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-name")
    parser.add_argument(
        "--algorithm", choices=("happo", "mappo", "madpo"), required=True
    )
    parser.add_argument("--condition", choices=("none", "arec"), default="none")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-env-steps", type=int)
    parser.add_argument("--n-rollout-threads", type=int)
    parser.add_argument("--div-coef", type=float, default=1000.0)
    parser.add_argument("--div-weight", type=float, default=0.05)
    parser.add_argument("--div-sigma", type=float, default=1.0)
    parser.add_argument("--div-max-samples", type=int, default=1024)
    parser.add_argument("--arec-coef", type=float, default=0.0001)
    parser.add_argument("--arec-q-steps", type=int, default=4)
    parser.add_argument("--arec-q-lr", type=float, default=0.001)
    parser.add_argument("--arec-fisher-ridge", type=float, default=0.001)
    parser.add_argument("--wandb-project", default="harl-dexhands-shadowhandover")
    parser.add_argument("--wandb-entity")
    parser.add_argument(
        "--wandb-mode", choices=("online", "disabled"), default="online"
    )
    args = parser.parse_args()

    args.harl_root = args.harl_root.expanduser().resolve()
    args.run_root = args.run_root.expanduser().resolve()
    config_path = (
        args.config.expanduser().resolve()
        if args.config is not None
        else args.harl_root / DEFAULT_CONFIG
    )
    if not (args.harl_root / "harl").is_dir() or not config_path.is_file():
        raise FileNotFoundError(
            "Initialize third_party/HARL and its DexHands dependency"
        )
    if args.seed < 0 or args.div_max_samples < 2:
        raise ValueError("Invalid seed or divergence sample count")
    if args.run_name is None:
        from experiments.harl_dexhands.protocol import Task

        args.run_name = Task(
            args.algorithm,
            args.seed,
            args.div_coef,
            args.div_weight,
            args.div_sigma,
            args.div_max_samples,
            args.condition,
            args.arec_coef,
            args.arec_q_steps,
            args.arec_q_lr,
            args.arec_fisher_ridge,
        ).name
    args.run_root.mkdir(parents=True, exist_ok=True)
    status_path = args.run_root / "status" / f"{args.run_name}.json"
    metrics_path = args.run_root / "metrics" / f"{args.run_name}.jsonl"

    # Isaac Gym requires this import before any module imports torch.
    import isaacgym  # noqa: F401

    sys.path.insert(0, str(args.harl_root))
    from experiments.harl_dexhands.protocol import (
        PROTOCOL_VERSION,
        AREC_PROTOCOL_VERSION,
        load_matched_config,
    )

    main_args, algo_args, env_args = load_matched_config(
        config_path,
        args.algorithm,
        args.seed,
        args.run_root / "harl_results",
        num_env_steps=args.num_env_steps,
        n_rollout_threads=args.n_rollout_threads,
        div_coef=args.div_coef,
        div_weight=args.div_weight,
        div_sigma=args.div_sigma,
        div_max_samples=args.div_max_samples,
        condition=args.condition,
        arec_coef=args.arec_coef,
        arec_q_steps=args.arec_q_steps,
        arec_q_lr=args.arec_q_lr,
        arec_fisher_ridge=args.arec_fisher_ridge,
    )
    main_args["exp_name"] = args.run_name
    main_args["load_config"] = str(config_path)
    total_steps = int(algo_args["train"]["num_env_steps"])
    base_status = {
        "protocol_version": (
            AREC_PROTOCOL_VERSION if args.condition == "arec" else PROTOCOL_VERSION
        ),
        "algorithm": args.algorithm,
        "condition": args.condition,
        "task": "ShadowHandOver",
        "actor_parameterization": "nps",
        "seed": args.seed,
        "source_config": str(config_path),
        "source_config_algorithm": "happo",
        "madpo_reference_repository": "https://github.com/hwdou6677/MADPO",
        "madpo_reference_commit": MADPO_REFERENCE_COMMIT,
        "jaxmarl_git_commit": _commit(REPO_ROOT),
        "harl_git_commit": _commit(args.harl_root),
        "algo_args": algo_args,
        "env_args": env_args,
        "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    _write_json(
        status_path,
        {
            **base_status,
            "status": "running",
            "run_name": args.run_name,
            "pid": os.getpid(),
            "env_steps": 0,
            "total_env_steps": total_steps,
        },
    )

    wandb_run = None
    runner = None
    try:
        if args.wandb_mode == "online":
            import wandb

            wandb_dir = args.run_root / "wandb"
            wandb_dir.mkdir(parents=True, exist_ok=True)
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.run_name,
                group="ShadowHandOver-nps",
                job_type=f"{args.algorithm}-{args.condition}",
                tags=[
                    "HARL",
                    "Bi-DexHands",
                    "ShadowHandOver",
                    "nps",
                    args.algorithm,
                    args.condition,
                ],
                config=base_status,
                dir=str(wandb_dir),
                mode="online",
            )
            wandb_run.define_metric("env_step")
            wandb_run.define_metric("episode_return_*", step_metric="env_step")
            wandb_run.define_metric("actor/*", step_metric="env_step")
            wandb_run.define_metric("critic/*", step_metric="env_step")

        from harl.algorithms.actors import ALGO_REGISTRY
        from experiments.harl_dexhands.arec import ARecHAPPO, ARecMADPO, ARecMAPPO
        from experiments.harl_dexhands.madpo import MADPO
        from experiments.harl_dexhands.runner import (
            TrackedARecHAPPORunner,
            TrackedARecMADPORunner,
            TrackedARecMAPPORunner,
            TrackedHAPPORunner,
            TrackedMADPORunner,
            TrackedMAPPORunner,
        )

        if args.condition == "arec":
            ALGO_REGISTRY.update(
                {"happo": ARecHAPPO, "mappo": ARecMAPPO, "madpo": ARecMADPO}
            )
            runner_class = {
                "happo": TrackedARecHAPPORunner,
                "mappo": TrackedARecMAPPORunner,
                "madpo": TrackedARecMADPORunner,
            }[args.algorithm]
        else:
            if args.algorithm == "madpo":
                ALGO_REGISTRY["madpo"] = MADPO
            runner_class = {
                "happo": TrackedHAPPORunner,
                "mappo": TrackedMAPPORunner,
                "madpo": TrackedMADPORunner,
            }[args.algorithm]
        runner = runner_class(main_args, algo_args, env_args)
        runner.configure_tracking(
            status_path=status_path,
            metrics_path=metrics_path,
            run_name=args.run_name,
            total_env_steps=total_steps,
            base_status=base_status,
            wandb_run=wandb_run,
        )
        runner.run()
    except BaseException as error:
        _write_json(
            status_path,
            {
                **base_status,
                "status": "failed",
                "run_name": args.run_name,
                "pid": os.getpid(),
                "env_steps": 0,
                "total_env_steps": total_steps,
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
