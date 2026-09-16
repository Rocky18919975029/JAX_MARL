#!/usr/bin/env python3
"""Run the matched four-seed MABrax MAPPO alignment matrix."""

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


PROTOCOL_VERSION = "mabrax-halfcheetah6x1-alignment-4seed-v1.0"
ACTOR_VARIANTS = {"ps": True, "nps": False}
DISTANCES = ("ln_mse", "linear_cka")
AVAILABLE_DISTANCES = (*DISTANCES, "containment")
MODES = ("none", "c_to_a", "a_to_c", "joint")


def repository_root():
    return Path(__file__).resolve().parents[1]


def parse_csv(value, allowed=None):
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(values) - set(allowed or values))
    if not values or unknown or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("invalid empty, duplicate, or unknown value")
    return values


def parse_seeds(value):
    if "-" in value and "," not in value:
        start, end = map(int, value.split("-", 1))
        values = tuple(range(start, end + 1))
    else:
        values = tuple(int(item) for item in value.split(",") if item.strip())
    if (
        not values
        or any(seed < 0 for seed in values)
        or len(values) != len(set(values))
    ):
        raise argparse.ArgumentTypeError("invalid seed selection")
    return values


def coefficient_label(value):
    return f"{value:.10g}".replace("-", "m").replace(".", "p")


@dataclass(frozen=True)
class Task:
    actor_label: str
    sharing: bool
    distance: str
    mode: str
    seed: int
    coefficient: float

    @property
    def condition(self):
        if self.mode == "none":
            return "none"
        suffix = {"ln_mse": "", "linear_cka": "_cka", "containment": "_dsc"}[
            self.distance
        ]
        return f"{self.mode}{suffix}"

    @property
    def run_name(self):
        return (
            f"MABRAX-halfcheetah_6x1-{self.actor_label}-{self.condition}-"
            f"lam{coefficient_label(self.coefficient)}-seed{self.seed}"
        )


def task_matrix(
    actor_variants,
    seeds,
    cka_coefficient,
    distances=DISTANCES,
    containment_coefficient=None,
):
    tasks = []
    for actor_label in actor_variants:
        sharing = ACTOR_VARIANTS[actor_label]
        for seed in seeds:
            tasks.append(Task(actor_label, sharing, "ln_mse", "none", seed, 0.1))
            for distance in distances:
                coefficients = {
                    "ln_mse": 0.1,
                    "linear_cka": cka_coefficient,
                    "containment": containment_coefficient,
                }
                coefficient = coefficients[distance]
                if coefficient is None:
                    raise ValueError(f"Missing calibrated coefficient for {distance}")
                for mode in MODES[1:]:
                    tasks.append(
                        Task(actor_label, sharing, distance, mode, seed, coefficient)
                    )
    if len(tasks) != len({task.run_name for task in tasks}):
        raise AssertionError("Run names are not unique")
    return tasks


def load_calibration(path, expected_distance):
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if payload.get("environment") != "halfcheetah_6x1":
        raise ValueError("Alignment calibration is not for halfcheetah_6x1")
    if payload.get("target_distance") != expected_distance:
        raise ValueError(
            f"Expected {expected_distance} calibration, got "
            f"{payload.get('target_distance')!r}"
        )
    coefficient = float(payload["global_alignment_coef"])
    if not coefficient > 0:
        raise ValueError("Alignment coefficient must be positive")
    return coefficient


def append_log(path, message):
    line = f"[{dt.datetime.now().astimezone():%Y-%m-%d %H:%M:%S}] {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(line + "\n")


def append_jsonl(path, payload):
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, sort_keys=True) + "\n")
        file.flush()


def build_command(args, task, commit):
    return [
        sys.executable,
        str(repository_root() / "baselines/MAPPO/mappo_ff_mabrax.py"),
        "ENV_NAME=halfcheetah_6x1",
        f"SEED={task.seed}",
        f"ACTOR_PARAMETER_SHARING={str(task.sharing).lower()}",
        "MATCHED_COMPARISON=true",
        f"ALIGN_MODE={task.mode}",
        f"ALIGN_DISTANCE={task.distance}",
        "ALIGN_DISTANCE_EPS=1e-8",
        "ALIGN_CONTAINMENT_RIDGE_RATIO=1e-3",
        "ALIGN_CONTAINMENT_EPS=1e-6",
        f"ALIGNMENT_COEF={task.coefficient:.12g}",
        "ALIGN_GRADIENT_CALIBRATION=false",
        f"EXPERIMENT_CONDITION={task.condition}",
        f"MATRIX_PROFILE={PROTOCOL_VERSION}",
        f"PROTOCOL_VERSION={PROTOCOL_VERSION}",
        f"GIT_COMMIT={commit}",
        "SAVE_CHECKPOINTS=true",
        f"CHECKPOINT_INTERVAL_TIMESTEPS={args.checkpoint_interval}",
        f"CHECKPOINT_DIR={args.run_root / 'checkpoints'}",
        f"WANDB_UPLOAD_CHECKPOINTS={str(args.upload_checkpoints).lower()}",
        "WANDB_MODE=online",
        f"PROJECT={args.project}",
        f"hydra.run.dir={args.run_root / 'hydra' / task.run_name}",
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--cka-calibration", type=Path)
    parser.add_argument("--containment-calibration", type=Path)
    parser.add_argument(
        "--distances",
        type=lambda value: parse_csv(value, AVAILABLE_DISTANCES),
        default=DISTANCES,
    )
    parser.add_argument(
        "--actor-variants",
        type=lambda value: parse_csv(value, ACTOR_VARIANTS),
        default=("ps", "nps"),
    )
    parser.add_argument("--seeds", type=parse_seeds, default=(1, 2, 3, 4))
    parser.add_argument("--gpus", type=parse_csv, default=("0", "1", "2", "3"))
    parser.add_argument("--max-runs-per-gpu", type=int, default=4)
    parser.add_argument("--checkpoint-interval", type=int, default=5_000_000)
    parser.add_argument("--project", default="jaxmarl-mabrax-halfcheetah6x1-alignment")
    parser.add_argument("--upload-checkpoints", action="store_true")
    parser.add_argument("--rerun-successful", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_runs_per_gpu < 1 or args.checkpoint_interval < 1:
        parser.error("concurrency and checkpoint interval must be positive")
    repo = repository_root()
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if subprocess.run(
        ("git", "status", "--porcelain=v1"),
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip():
        raise RuntimeError("Worktree is dirty; commit the protocol before launching")
    cka_coefficient = None
    containment_coefficient = None
    if "linear_cka" in args.distances:
        if args.cka_calibration is None:
            parser.error("--cka-calibration is required for linear_cka")
        cka_coefficient = load_calibration(args.cka_calibration, "linear_cka")
    if "containment" in args.distances:
        if args.containment_calibration is None:
            parser.error("--containment-calibration is required for containment")
        containment_coefficient = load_calibration(
            args.containment_calibration, "containment"
        )
    tasks = task_matrix(
        args.actor_variants,
        args.seeds,
        cka_coefficient,
        args.distances,
        containment_coefficient,
    )
    args.run_root = args.run_root.expanduser().resolve()
    expected = len(args.actor_variants) * len(args.seeds) * (
        1 + 3 * len(args.distances)
    )
    if len(tasks) != expected:
        raise AssertionError(f"Expected {expected} tasks, found {len(tasks)}")
    if args.dry_run:
        print(
            f"protocol={PROTOCOL_VERSION} runs={len(tasks)} "
            f"cka_lambda={cka_coefficient} dsc_lambda={containment_coefficient}"
        )
        for index, task in enumerate(tasks):
            print(f"GPU {args.gpus[index % len(args.gpus)]}: {task.run_name}")
        return

    directories = {
        name: args.run_root / name
        for name in (
            "logs",
            "status",
            "checkpoints",
            "hydra",
            "wandb",
            "wandb_cache",
            "wandb_staging",
            "wandb_artifacts",
        )
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": commit,
        "environment": "halfcheetah_6x1",
        "actor_variants": args.actor_variants,
        "distances": args.distances,
        "modes": MODES,
        "seeds": args.seeds,
        "ln_mse_alignment_coef": 0.1,
        "linear_cka_alignment_coef": cka_coefficient,
        "cka_calibration": (
            str(args.cka_calibration.expanduser().resolve())
            if args.cka_calibration is not None
            else None
        ),
        "unique_runs": len(tasks),
        "checkpoint_interval": args.checkpoint_interval,
    }
    if "containment" in args.distances:
        manifest.update(
            {
                "containment_alignment_coef": containment_coefficient,
                "containment_ridge_ratio": 1e-3,
                "containment_epsilon": 1e-6,
                "containment_calibration": str(
                    args.containment_calibration.expanduser().resolve()
                ),
            }
        )
    manifest_path = args.run_root / "experiment_manifest.json"
    if manifest_path.is_file():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise RuntimeError(f"Experiment settings changed: {manifest_path}")
    else:
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    launcher_log = args.run_root / "launcher.log"
    completion_log = args.run_root / "completion_manifest.jsonl"
    failure_log = args.run_root / "failure_registry.jsonl"
    queues = {gpu: collections.deque() for gpu in args.gpus}
    skipped = 0
    for index, task in enumerate(tasks):
        marker = directories["status"] / f"{task.run_name}.json"
        if marker.is_file() and not args.rerun_successful:
            if (
                json.loads(marker.read_text(encoding="utf-8")).get("status")
                == "completed"
            ):
                skipped += 1
                continue
        queues[args.gpus[index % len(args.gpus)]].append(task)
    append_log(
        launcher_log,
        f"selected={len(tasks)} pending={sum(map(len, queues.values()))} skipped={skipped}",
    )

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
            for process, _, _, handle, _ in running.values():
                process.terminate()
                handle.close()
            raise SystemExit(130)
        for gpu in args.gpus:
            active = sum(item[1] == gpu for item in running.values())
            while queues[gpu] and active < args.max_runs_per_gpu:
                task = queues[gpu].popleft()
                log_path = directories["logs"] / f"{task.run_name}.log"
                handle = log_path.open("w", encoding="utf-8")
                environment = os.environ.copy()
                environment.pop("LD_LIBRARY_PATH", None)
                environment.update(
                    {
                        "CUDA_VISIBLE_DEVICES": gpu,
                        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                        "HYDRA_FULL_ERROR": "1",
                        "WANDB_DIR": str(directories["wandb"]),
                        "WANDB_CACHE_DIR": str(directories["wandb_cache"]),
                        "WANDB_DATA_DIR": str(directories["wandb_staging"]),
                        "WANDB_ARTIFACT_DIR": str(directories["wandb_artifacts"]),
                        "WANDB_NAME": task.run_name,
                        "WANDB_RUN_GROUP": (
                            f"MABRAX-halfcheetah_6x1-{task.actor_label}-"
                            f"{task.condition}-lam{coefficient_label(task.coefficient)}"
                        ),
                        "WANDB_TAGS": ",".join(
                            (
                                PROTOCOL_VERSION,
                                "mabrax",
                                "halfcheetah_6x1",
                                task.actor_label,
                                task.condition,
                                f"distance-{task.distance}",
                            )
                        ),
                    }
                )
                started = dt.datetime.now(dt.timezone.utc).isoformat()
                process = subprocess.Popen(
                    build_command(args, task, commit),
                    cwd=repo,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
                running[process.pid] = (process, gpu, task, handle, started)
                append_log(
                    launcher_log, f"GPU {gpu} START {task.run_name} pid={process.pid}"
                )
                active += 1

        completed_pids = [
            pid for pid, item in running.items() if item[0].poll() is not None
        ]
        for pid in completed_pids:
            process, gpu, task, handle, started = running.pop(pid)
            handle.close()
            status = "completed" if process.returncode == 0 else "failed"
            record = {
                "schema_version": 1,
                "protocol_version": PROTOCOL_VERSION,
                "git_commit": commit,
                "run_name": task.run_name,
                "environment": "halfcheetah_6x1",
                "actor_parameterization": task.actor_label,
                "actor_parameter_sharing": task.sharing,
                "condition": task.condition,
                "align_mode": task.mode,
                "align_distance": task.distance,
                "alignment_coef": task.coefficient,
                "seed": task.seed,
                "gpu": gpu,
                "started_at_utc": started,
                "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "status": status,
                "exit_code": process.returncode,
                "log": str(directories["logs"] / f"{task.run_name}.log"),
            }
            append_jsonl(completion_log, record)
            if process.returncode:
                failures += 1
                append_jsonl(failure_log, record)
            (directories["status"] / f"{task.run_name}.json").write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            append_log(
                launcher_log,
                f"GPU {gpu} END   {task.run_name} status={process.returncode}",
            )
        if not completed_pids:
            time.sleep(1)
    append_log(launcher_log, f"all selected runs finished; failures={failures}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
