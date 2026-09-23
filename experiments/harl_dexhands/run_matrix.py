#!/usr/bin/env python3
"""Launch the matched ShadowHandOver HAPPO/MAPPO/MADPO matrix."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = REPO_ROOT / "experiments" / "harl_dexhands" / "train.py"
OFFICIAL_CONFIG = Path("tuned_configs/dexhands/ShadowHandOver/happo/config.json")


def training_environment(gpu: str) -> dict[str, str]:
    """Build an Isaac Gym child environment from the active Python runtime."""

    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    conda_prefix = Path(environment.get("CONDA_PREFIX", sys.prefix)).expanduser()
    runtime_lib = str(conda_prefix / "lib")
    current = environment.get("LD_LIBRARY_PATH", "")
    entries = [entry for entry in current.split(os.pathsep) if entry]
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(
        [runtime_lib, *(entry for entry in entries if entry != runtime_lib)]
    )
    return environment


def verify_wandb() -> None:
    """Reject an incomplete or shadowed W&B import before launching workers."""
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError(
            "W&B online mode requires the wandb SDK in the active Python "
            "environment; install it or use --wandb-mode disabled"
        ) from error
    if not callable(getattr(wandb, "init", None)):
        raise RuntimeError(
            "The imported wandb module has no callable init; "
            f"loaded from {getattr(wandb, '__file__', None)!r}. "
            "Check for a shadowing local module or an incomplete SDK installation."
        )


def freeze_manifest(root: Path, study_spec: dict, tasks: list) -> dict:
    """Record the exact grid and reject accidental changes on resume."""
    path = root / "experiment_manifest.json"
    payload = {
        "schema_version": 1,
        "study_spec": study_spec,
        "runs": [
            {
                "run_name": task.name,
                "algorithm": task.algorithm,
                "condition": task.condition,
                "seed": task.seed,
                "arec_coef": task.arec_coef if task.condition == "arec" else 0.0,
            }
            for task in tasks
        ],
    }
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(
                f"Run-root manifest differs from this launch: {path}. "
                "Use the original grid/settings or a new run root."
            )
        return existing
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--harl-root", type=Path, default=REPO_ROOT / "third_party" / "HARL"
    )
    parser.add_argument("--algorithms", default="happo,mappo,madpo")
    parser.add_argument("--conditions", default="none")
    parser.add_argument("--seeds", default="1-4")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--max-runs-per-gpu", type=int, default=1)
    parser.add_argument("--num-env-steps", type=int)
    parser.add_argument("--n-rollout-threads", type=int)
    parser.add_argument("--div-coef", type=float, default=1000.0)
    parser.add_argument("--div-weight", type=float, default=0.05)
    parser.add_argument("--div-sigma", type=float, default=1.0)
    parser.add_argument("--div-max-samples", type=int, default=1024)
    coefficient_group = parser.add_mutually_exclusive_group()
    coefficient_group.add_argument("--arec-coef", type=float, default=0.0001)
    coefficient_group.add_argument("--arec-coefs")
    parser.add_argument("--arec-q-steps", type=int, default=4)
    parser.add_argument("--arec-q-lr", type=float, default=0.001)
    parser.add_argument("--arec-fisher-ridge", type=float, default=0.001)
    parser.add_argument("--wandb-project", default="harl-dexhands-shadowhandover")
    parser.add_argument(
        "--wandb-mode", choices=("online", "disabled"), default="online"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT))
    from experiments.harl_dexhands.protocol import (
        ALGORITHMS,
        AREC_PROTOCOL_VERSION,
        CONDITIONS,
        PROTOCOL_VERSION,
        parse_csv,
        parse_positive_floats,
        parse_seeds,
        task_matrix,
    )

    if args.max_runs_per_gpu < 1:
        raise ValueError("max-runs-per-gpu must be positive")
    if args.max_runs_per_gpu > 2:
        raise ValueError("ShadowHandOver is limited to two concurrent runs per GPU")
    if args.num_env_steps is not None and args.num_env_steps <= 0:
        raise ValueError("num-env-steps must be positive")
    if args.n_rollout_threads is not None and args.n_rollout_threads <= 0:
        raise ValueError("n-rollout-threads must be positive")
    if args.arec_q_steps < 1 or args.arec_q_lr <= 0 or args.arec_fisher_ridge <= 0:
        raise ValueError("ARec q steps/LR and Fisher ridge must be positive")
    algorithms = parse_csv(args.algorithms, ALGORITHMS)
    conditions = parse_csv(args.conditions, CONDITIONS)
    seeds = parse_seeds(args.seeds)
    arec_coefs = (
        (args.arec_coef,)
        if args.arec_coefs is None
        else parse_positive_floats(args.arec_coefs)
    )
    gpus = tuple(piece.strip() for piece in args.gpus.split(",") if piece.strip())
    if not gpus:
        raise ValueError("Select at least one GPU")
    if args.wandb_mode == "online" and not args.dry_run:
        verify_wandb()
    root = args.run_root.expanduser().resolve()
    harl_root = args.harl_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    tasks = task_matrix(
        algorithms,
        seeds,
        div_coef=args.div_coef,
        div_weight=args.div_weight,
        div_sigma=args.div_sigma,
        div_max_samples=args.div_max_samples,
        conditions=conditions,
        arec_coef=args.arec_coef,
        arec_coefs=arec_coefs,
        arec_q_steps=args.arec_q_steps,
        arec_q_lr=args.arec_q_lr,
        arec_fisher_ridge=args.arec_fisher_ridge,
    )
    config_path = harl_root / OFFICIAL_CONFIG
    config_hash = (
        hashlib.sha256(config_path.read_bytes()).hexdigest()
        if config_path.is_file()
        else None
    )
    if not args.dry_run and config_hash is None:
        raise FileNotFoundError(f"Missing official HARL config: {config_path}")
    official_budget = (
        int(json.loads(config_path.read_text())["algo_args"]["train"]["num_env_steps"])
        if config_hash is not None
        else 0
    )
    protocol_version = (
        AREC_PROTOCOL_VERSION if "arec" in conditions else PROTOCOL_VERSION
    )
    freeze_manifest(
        root,
        {
            "protocol_version": protocol_version,
            "algorithms": list(algorithms),
            "conditions": list(conditions),
            "seeds": list(seeds),
            "arec_coefs": list(arec_coefs),
            "arec_q_steps": args.arec_q_steps,
            "arec_q_lr": args.arec_q_lr,
            "arec_fisher_ridge": args.arec_fisher_ridge,
            "div_coef": args.div_coef,
            "div_weight": args.div_weight,
            "div_sigma": args.div_sigma,
            "div_max_samples": args.div_max_samples,
            "num_env_steps": args.num_env_steps or official_budget,
            "n_rollout_threads": args.n_rollout_threads,
            "harl_root": str(harl_root),
            "official_config_sha256": config_hash,
        },
        tasks,
    )

    def is_completed(task) -> bool:
        path = root / "status" / f"{task.name}.json"
        if not path.is_file():
            return False
        try:
            return (
                json.loads(path.read_text(encoding="utf-8")).get("status")
                == "completed"
            )
        except (OSError, ValueError):
            return False

    pending = [task for task in tasks if not is_completed(task)]
    print(
        f"Protocol={protocol_version} "
        f"total={len(tasks)} pending={len(pending)} "
        f"algorithms={','.join(algorithms)} conditions={','.join(conditions)} "
        f"seeds={','.join(map(str, seeds))} "
        f"arec_coefs={','.join(map(str, arec_coefs))}",
        flush=True,
    )
    slots = [gpu for gpu in gpus for _ in range(args.max_runs_per_gpu)]
    queues = [[] for _ in slots]
    for index, task in enumerate(pending):
        queues[index % len(slots)].append(task)
    lock = threading.Lock()
    stop_launching = threading.Event()
    launcher_log = root / "launcher.log"

    def event(message: str) -> None:
        line = f"[{dt.datetime.now().astimezone():%Y-%m-%d %H:%M:%S}] {message}"
        with lock:
            print(line, flush=True)
            with launcher_log.open("a", encoding="utf-8") as file:
                file.write(line + "\n")

    def run_queue(slot: int) -> None:
        gpu = slots[slot]
        for task in queues[slot]:
            if stop_launching.is_set():
                return
            command = [
                sys.executable,
                str(TRAIN_SCRIPT),
                "--harl-root",
                str(harl_root),
                "--run-root",
                str(root),
                "--run-name",
                task.name,
                "--algorithm",
                task.algorithm,
                "--condition",
                task.condition,
                "--seed",
                str(task.seed),
                "--div-coef",
                str(args.div_coef),
                "--div-weight",
                str(args.div_weight),
                "--div-sigma",
                str(args.div_sigma),
                "--div-max-samples",
                str(args.div_max_samples),
                "--arec-coef",
                str(task.arec_coef),
                "--arec-q-steps",
                str(args.arec_q_steps),
                "--arec-q-lr",
                str(args.arec_q_lr),
                "--arec-fisher-ridge",
                str(args.arec_fisher_ridge),
                "--wandb-project",
                args.wandb_project,
                "--wandb-mode",
                args.wandb_mode,
            ]
            if args.num_env_steps is not None:
                command.extend(["--num-env-steps", str(args.num_env_steps)])
            if args.n_rollout_threads is not None:
                command.extend(["--n-rollout-threads", str(args.n_rollout_threads)])
            event(f"GPU {gpu} START {task.name}")
            if args.dry_run:
                event("COMMAND " + " ".join(command))
                continue
            environment = training_environment(gpu)
            output = root / "logs" / f"{task.name}.log"
            with output.open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            event(f"GPU {gpu} END   {task.name} status={result.returncode}")
            if result.returncode != 0:
                stop_launching.set()
                event(
                    "Stopping new launches after a failed run; completed runs are resumable"
                )
                return

    with ThreadPoolExecutor(max_workers=len(slots)) as executor:
        futures = [executor.submit(run_queue, index) for index in range(len(slots))]
        for future in futures:
            future.result()
    failures = 0
    for task in tasks:
        path = root / "status" / f"{task.name}.json"
        if not path.is_file():
            failures += int(not args.dry_run)
            continue
        try:
            failures += json.loads(path.read_text()).get("status") != "completed"
        except (OSError, ValueError):
            failures += 1
    event(f"matrix finished; failures={failures}")
    if failures:
        raise RuntimeError(f"{failures} runs failed; inspect {root / 'logs'}")


if __name__ == "__main__":
    main()
