#!/usr/bin/env python3
"""Launch the focused 20M-step NPS matrix on 3s5z_vs_3s6z.

The matrix contains four matched seeds for exactly three conditions: isolated,
C-to-A LN-MSE, and C-to-A Linear CKA.  The launcher is restart-safe: completed
and currently running tasks are skipped, so it can repair a partially failed
manual launch without duplicating healthy runs.
"""

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


MAP_NAME = "3s5z_vs_3s6z"
TOTAL_TIMESTEPS = 20_000_000
PROTOCOL_VERSION = "3s5z-vs-3s6z-nps-20m-v1"
DEFAULT_PROJECT = "jaxmarl-smax-3s5z-vs-3s6z-nps-20m"
DEFAULT_CKA_COEF = 0.3515769798


@dataclass(frozen=True)
class Task:
    seed: int
    display_condition: str
    align_mode: str
    align_distance: str
    alignment_coef: float
    experiment_condition: str

    @property
    def run_name(self):
        return (
            f"SMAX20M-{MAP_NAME}-nps-{self.display_condition}-seed{self.seed}"
        )


def task_matrix(seeds=(1, 2, 3, 4), cka_coef=DEFAULT_CKA_COEF):
    specifications = (
        ("none", "none", "ln_mse", 0.1, "none"),
        ("c_to_a_mse", "c_to_a", "ln_mse", 0.1, "c_to_a"),
        ("c_to_a_cka", "c_to_a", "linear_cka", cka_coef, "c_to_a_cka"),
    )
    return [
        Task(seed, display, mode, distance, coefficient, experiment)
        for seed in seeds
        for display, mode, distance, coefficient, experiment in specifications
    ]


def parse_csv(value):
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("Expected a non-empty unique CSV")
    return values


def append_log(path, message):
    timestamp = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(line + "\n")


def process_listing():
    return subprocess.run(
        ("pgrep", "-af", "baselines/MAPPO/mappo_rnn_smax.py"),
        text=True,
        capture_output=True,
    ).stdout


def is_active(task, listing):
    return task.run_name in listing


def completed_checkpoint(run_root, task):
    patterns = (
        f"**/{task.run_name}-*/final/model.safetensors",
        f"**/{task.run_name}/final/model.safetensors",
    )
    return next(
        (
            path
            for pattern in patterns
            for path in (run_root / "checkpoints").glob(pattern)
        ),
        None,
    )


def build_command(repo, run_root, project, task):
    return [
        sys.executable,
        str(repo / "baselines" / "MAPPO" / "mappo_rnn_smax.py"),
        f"MAP_NAME={MAP_NAME}",
        f"SEED={task.seed}",
        "ACTOR_PARAMETER_SHARING=false",
        "MATCHED_COMPARISON=true",
        f"ALIGN_MODE={task.align_mode}",
        f"ALIGN_DISTANCE={task.align_distance}",
        f"ALIGNMENT_COEF={task.alignment_coef:.10g}",
        "ALIGN_GRADIENT_CALIBRATION=false",
        "ALIGN_TARGET_SHUFFLE=false",
        f"TOTAL_TIMESTEPS={TOTAL_TIMESTEPS}",
        "SAVE_CHECKPOINTS=true",
        "CHECKPOINT_INTERVAL_TIMESTEPS=500000",
        f"CHECKPOINT_DIR={run_root / 'checkpoints'}",
        "WANDB_UPLOAD_CHECKPOINTS=false",
        "WANDB_MODE=online",
        f"PROJECT={project}",
        f"EXPERIMENT_CONDITION={task.experiment_condition}",
        f"MATRIX_PROFILE={PROTOCOL_VERSION}",
        f"PROTOCOL_VERSION={PROTOCOL_VERSION}",
        f"hydra.run.dir={run_root / 'hydra' / task.run_name}",
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--gpus", type=parse_csv, default=("0", "1", "2", "3"))
    parser.add_argument("--max-runs-per-gpu", type=int, default=1)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--cka-coef", type=float, default=DEFAULT_CKA_COEF)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_runs_per_gpu < 1 or args.cka_coef <= 0:
        parser.error("Concurrency and CKA coefficient must be positive")

    repo = Path(__file__).resolve().parents[1]
    run_root = args.run_root.expanduser().resolve()
    directories = {
        name: run_root / name
        for name in ("logs", "status", "checkpoints", "hydra", "wandb")
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    tasks = task_matrix(cka_coef=args.cka_coef)
    listing = process_listing()
    completed = {
        task.run_name: completed_checkpoint(run_root, task) for task in tasks
    }
    active = {task.run_name for task in tasks if is_active(task, listing)}
    pending = [
        task for task in tasks if completed[task.run_name] is None and task.run_name not in active
    ]

    manifest = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "map_name": MAP_NAME,
        "actor_parameterization": "nps",
        "total_timesteps": TOTAL_TIMESTEPS,
        "seeds": [1, 2, 3, 4],
        "conditions": ["none", "c_to_a_mse", "c_to_a_cka"],
        "cka_alignment_coef": args.cka_coef,
        "project": args.project,
    }
    manifest_path = run_root / "experiment_manifest.json"
    if manifest_path.is_file():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior != manifest:
            raise RuntimeError(f"Experiment settings changed: {manifest_path}")
    else:
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    launcher_log = run_root / "launcher.log"
    append_log(
        launcher_log,
        f"selected=12 completed={sum(value is not None for value in completed.values())} "
        f"active_external={len(active)} pending={len(pending)}",
    )
    for task in tasks:
        state = (
            "COMPLETE"
            if completed[task.run_name] is not None
            else "ACTIVE"
            if task.run_name in active
            else "PENDING"
        )
        print(f"{state:8s} {task.run_name}")
    if args.dry_run or not pending:
        return

    queues = {gpu: collections.deque() for gpu in args.gpus}
    for index, task in enumerate(pending):
        queues[args.gpus[index % len(args.gpus)]].append(task)

    running = {}
    stop_requested = False

    def request_stop(signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        append_log(launcher_log, f"received signal {signum}; stopping children")

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    failures = 0
    while any(queues.values()) or running:
        if stop_requested:
            for process, _, _, handle in running.values():
                process.terminate()
                handle.close()
            raise SystemExit(130)
        for gpu in args.gpus:
            gpu_active = sum(item[1] == gpu for item in running.values())
            while queues[gpu] and gpu_active < args.max_runs_per_gpu:
                task = queues[gpu].popleft()
                log_path = directories["logs"] / f"{task.run_name}.log"
                handle = log_path.open("a", encoding="utf-8")
                handle.write("\n===== restart-safe launcher attempt =====\n")
                handle.flush()
                environment = os.environ.copy()
                environment.pop("LD_LIBRARY_PATH", None)
                environment.update(
                    {
                        "CUDA_VISIBLE_DEVICES": gpu,
                        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                        "HYDRA_FULL_ERROR": "1",
                        "WANDB_DIR": str(directories["wandb"]),
                        "WANDB_NAME": task.run_name,
                        "WANDB_RUN_GROUP": (
                            f"SMAX20M-{MAP_NAME}-nps-{task.display_condition}"
                        ),
                        "WANDB_TAGS": ",".join(
                            (
                                "smax",
                                MAP_NAME,
                                "nps",
                                "20m",
                                task.display_condition,
                                f"seed-{task.seed}",
                            )
                        ),
                    }
                )
                process = subprocess.Popen(
                    build_command(repo, run_root, args.project, task),
                    cwd=repo,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
                running[process.pid] = (process, gpu, task, handle)
                append_log(
                    launcher_log,
                    f"GPU {gpu} START {task.run_name} pid={process.pid}",
                )
                gpu_active += 1

        finished = [pid for pid, item in running.items() if item[0].poll() is not None]
        for pid in finished:
            process, gpu, task, handle = running.pop(pid)
            handle.close()
            status = "completed" if process.returncode == 0 else "failed"
            if process.returncode != 0:
                failures += 1
            (directories["status"] / f"{task.run_name}.json").write_text(
                json.dumps(
                    {
                        "run_name": task.run_name,
                        "status": status,
                        "return_code": process.returncode,
                        "gpu": gpu,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            append_log(
                launcher_log,
                f"GPU {gpu} END   {task.run_name} status={process.returncode}",
            )
        if running:
            time.sleep(2)
    append_log(launcher_log, f"launcher finished; failures={failures}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
