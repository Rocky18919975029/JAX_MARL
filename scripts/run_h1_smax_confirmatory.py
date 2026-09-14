#!/usr/bin/env python3
"""Launch the frozen H1 SMAX matrix with bounded per-GPU concurrency."""

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

from h1_protocol import FROZEN_KEYS, PROTOCOL_VERSION, repository_root


MAPS = ("10m_vs_11m", "smacv2_10_units")
ACTOR_VARIANTS = (("ps", True), ("nps", False))
CONDITIONS = (
    "none",
    "a_to_c",
    "c_to_a",
    "reciprocal",
    "joint",
    "a_to_c_shuffled",
    "c_to_a_shuffled",
)
DEFAULT_SEEDS = tuple(range(101, 111))


@dataclass(frozen=True)
class Task:
    map_name: str
    actor_label: str
    sharing: bool
    condition: str
    seed: int

    @property
    def align_mode(self):
        return self.condition.removesuffix("_shuffled")

    @property
    def shuffled(self):
        return self.condition.endswith("_shuffled")

    @property
    def run_name(self):
        return (
            f"H1-{self.map_name}-{self.actor_label}-{self.condition}-"
            f"lam0p1-seed{self.seed}"
        )


def parse_csv(value, allowed=None):
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if allowed is not None:
        unknown = sorted(set(values) - set(allowed))
        if unknown:
            raise argparse.ArgumentTypeError(f"unknown values: {', '.join(unknown)}")
    return values


def parse_seeds(value):
    seeds = []
    for piece in value.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            start, end = (int(item) for item in piece.split("-", 1))
            seeds.extend(range(start, end + 1))
        else:
            seeds.append(int(piece))
    if not seeds or len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("seeds must be a non-empty unique list/range")
    invalid = sorted(set(seeds) - set(DEFAULT_SEEDS))
    if invalid:
        raise argparse.ArgumentTypeError(
            f"H1 confirmatory seeds must be 101-110; got {invalid}"
        )
    return tuple(seeds)


def hydra_value(value):
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def git_state(repo):
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return commit, status


def append_jsonl(path, payload):
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, sort_keys=True) + "\n")
        file.flush()


def append_log(path, message):
    timestamp = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(line + "\n")


def build_command(args, frozen, task, commit):
    script = repository_root() / "baselines" / "MAPPO" / "mappo_rnn_smax.py"
    command = [sys.executable, str(script)]
    for key in FROZEN_KEYS:
        if key == "ENV_KWARGS":
            for nested_key, value in frozen[key].items():
                command.append(f"ENV_KWARGS.{nested_key}={hydra_value(value)}")
        else:
            command.append(f"{key}={hydra_value(frozen[key])}")
    command.extend(
        (
            f"MAP_NAME={task.map_name}",
            f"SEED={task.seed}",
            f"ACTOR_PARAMETER_SHARING={hydra_value(task.sharing)}",
            f"ALIGN_MODE={task.align_mode}",
            f"ALIGN_TARGET_SHUFFLE={hydra_value(task.shuffled)}",
            "ALIGN_TARGET_SHUFFLE_SCOPE=same_agent_env_time",
            f"ALIGN_SHUFFLE_SEED_OFFSET={args.shuffle_seed_offset}",
            f"EXPERIMENT_CONDITION={task.condition}",
            f"PROTOCOL_VERSION={PROTOCOL_VERSION}",
            f"GIT_COMMIT={commit}",
            "SAVE_CHECKPOINTS=true",
            "CHECKPOINT_INTERVAL_TIMESTEPS=500000",
            f"CHECKPOINT_DIR={args.run_root / 'checkpoints'}",
            f"WANDB_UPLOAD_CHECKPOINTS={hydra_value(args.upload_checkpoints)}",
            "WANDB_MODE=online",
            f"PROJECT={args.project}",
            f"hydra.run.dir={args.run_root / 'hydra' / task.run_name}",
        )
    )
    return command


def task_matrix(args):
    actor_lookup = dict(ACTOR_VARIANTS)
    return [
        Task(map_name, actor_label, actor_lookup[actor_label], condition, seed)
        for map_name in args.maps
        for actor_label in args.actor_variants
        for condition in args.conditions
        for seed in args.seeds
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--frozen-config",
        type=Path,
        help="defaults to RUN_ROOT/protocol/frozen_training_config.json",
    )
    parser.add_argument("--project", default="h1-smax-confirmatory")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--max-runs-per-gpu", type=int, default=5)
    parser.add_argument("--seeds", type=parse_seeds, default=(101, 102))
    parser.add_argument(
        "--maps",
        type=lambda value: parse_csv(value, MAPS),
        default=MAPS,
    )
    parser.add_argument(
        "--actor-variants",
        type=lambda value: parse_csv(value, dict(ACTOR_VARIANTS)),
        default=tuple(dict(ACTOR_VARIANTS)),
    )
    parser.add_argument(
        "--conditions",
        type=lambda value: parse_csv(value, CONDITIONS),
        default=CONDITIONS,
    )
    parser.add_argument("--shuffle-seed-offset", type=int, default=700000)
    parser.add_argument("--upload-checkpoints", action="store_true")
    parser.add_argument("--allow-git-mismatch", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--rerun-successful", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.run_root = args.run_root.expanduser().resolve()
    frozen_path = args.frozen_config or (
        args.run_root / "protocol" / "frozen_training_config.json"
    )
    payload = json.loads(frozen_path.expanduser().resolve().read_text(encoding="utf-8"))
    frozen = payload.get("training_config", payload)
    missing = [key for key in FROZEN_KEYS if key not in frozen]
    if missing:
        raise ValueError(f"Frozen config is missing: {', '.join(missing)}")
    if payload.get("protocol_version", PROTOCOL_VERSION) != PROTOCOL_VERSION:
        raise ValueError("Frozen config protocol version does not match this launcher")
    if float(frozen["ALIGNMENT_COEF"]) != 0.1:
        raise ValueError("H1 v1.0 locks ALIGNMENT_COEF=0.1")
    if not bool(frozen["MATCHED_COMPARISON"]):
        raise ValueError("H1 v1.0 locks MATCHED_COMPARISON=true")
    if args.max_runs_per_gpu <= 0:
        raise ValueError("--max-runs-per-gpu must be positive")

    repo = repository_root()
    commit, status = git_state(repo)
    frozen_commit = payload.get("git_commit_at_freeze")
    if frozen_commit and frozen_commit != "unknown" and commit != frozen_commit:
        if not args.allow_git_mismatch:
            raise RuntimeError(
                f"Git commit {commit} differs from frozen commit {frozen_commit}. "
                "Freeze again for a new protocol version, or explicitly pass "
                "--allow-git-mismatch after documenting the deviation."
            )
    if status and not args.allow_dirty:
        raise RuntimeError(
            "Worktree is dirty. Commit the protocol implementation first, or pass "
            "--allow-dirty only after saving the manifest/diff."
        )

    gpu_ids = tuple(parse_csv(args.gpus))
    if not gpu_ids:
        raise ValueError("--gpus must select at least one GPU")
    tasks = task_matrix(args)
    args.run_root.mkdir(parents=True, exist_ok=True)
    logs_dir = args.run_root / "logs"
    status_dir = args.run_root / "status"
    wandb_dir = args.run_root / "wandb"
    wandb_cache_dir = args.run_root / "wandb_cache"
    wandb_data_dir = args.run_root / "wandb_staging"
    wandb_artifact_dir = args.run_root / "wandb_artifacts"
    hydra_dir = args.run_root / "hydra"
    logs_dir.mkdir(parents=True, exist_ok=True)
    status_dir.mkdir(parents=True, exist_ok=True)
    wandb_dir.mkdir(parents=True, exist_ok=True)
    wandb_cache_dir.mkdir(parents=True, exist_ok=True)
    wandb_data_dir.mkdir(parents=True, exist_ok=True)
    wandb_artifact_dir.mkdir(parents=True, exist_ok=True)
    hydra_dir.mkdir(parents=True, exist_ok=True)
    launcher_log = args.run_root / "launcher.log"
    manifest_path = args.run_root / "completion_manifest.jsonl"
    failure_registry = args.run_root / "failure_registry.jsonl"

    pending_by_gpu = {gpu: collections.deque() for gpu in gpu_ids}
    skipped = 0
    for index, task in enumerate(tasks):
        marker = status_dir / f"{task.run_name}.json"
        if marker.is_file() and not args.rerun_successful:
            prior = json.loads(marker.read_text(encoding="utf-8"))
            if prior.get("status") == "completed":
                skipped += 1
                continue
        gpu = gpu_ids[index % len(gpu_ids)]
        pending_by_gpu[gpu].append(task)

    append_log(
        launcher_log,
        f"protocol={PROTOCOL_VERSION} commit={commit} selected={len(tasks)} "
        f"pending={sum(map(len, pending_by_gpu.values()))} skipped={skipped}",
    )
    if args.dry_run:
        for gpu, pending in pending_by_gpu.items():
            for task in pending:
                print(f"GPU {gpu}: {task.run_name}")
        return

    running = {}
    stop_requested = False

    def request_stop(signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        append_log(launcher_log, f"received signal {signum}; stopping children")

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    failures = 0
    while any(pending_by_gpu.values()) or running:
        if stop_requested:
            for process, _, _, handle, _ in running.values():
                process.terminate()
                handle.close()
            raise SystemExit(130)

        for gpu in gpu_ids:
            active_on_gpu = sum(item[1] == gpu for item in running.values())
            while pending_by_gpu[gpu] and active_on_gpu < args.max_runs_per_gpu:
                task = pending_by_gpu[gpu].popleft()
                log_path = logs_dir / f"{task.run_name}.log"
                log_handle = log_path.open("w", encoding="utf-8")
                environment = os.environ.copy()
                environment.pop("LD_LIBRARY_PATH", None)
                environment.update(
                    {
                        "CUDA_VISIBLE_DEVICES": gpu,
                        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                        "HYDRA_FULL_ERROR": "1",
                        "WANDB_DIR": str(wandb_dir),
                        "WANDB_CACHE_DIR": str(wandb_cache_dir),
                        "WANDB_DATA_DIR": str(wandb_data_dir),
                        "WANDB_ARTIFACT_DIR": str(wandb_artifact_dir),
                        "WANDB_NAME": task.run_name,
                        "WANDB_RUN_GROUP": (
                            f"H1-{task.map_name}-{task.actor_label}-lam0p1"
                        ),
                        "WANDB_TAGS": ",".join(
                            (
                                "h1-confirmatory",
                                "smax",
                                task.map_name,
                                task.actor_label,
                                task.condition,
                                "lambda-0.1",
                                PROTOCOL_VERSION,
                            )
                        ),
                    }
                )
                command = build_command(args, frozen, task, commit)
                started = dt.datetime.now(dt.timezone.utc).isoformat()
                process = subprocess.Popen(
                    command,
                    cwd=repo,
                    env=environment,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                )
                running[process.pid] = (
                    process,
                    gpu,
                    task,
                    log_handle,
                    started,
                )
                append_log(
                    launcher_log,
                    f"GPU {gpu} START {task.run_name} pid={process.pid}",
                )
                active_on_gpu += 1

        completed_pids = [
            pid for pid, item in running.items() if item[0].poll() is not None
        ]
        for pid in completed_pids:
            process, gpu, task, log_handle, started = running.pop(pid)
            log_handle.close()
            status = "completed" if process.returncode == 0 else "failed"
            record = {
                "schema_version": 1,
                "protocol_version": PROTOCOL_VERSION,
                "git_commit": commit,
                "run_name": task.run_name,
                "map_name": task.map_name,
                "actor_parameterization": task.actor_label,
                "actor_parameter_sharing": task.sharing,
                "condition": task.condition,
                "align_mode": task.align_mode,
                "shuffled": task.shuffled,
                "alignment_coef": 0.1,
                "seed": task.seed,
                "gpu": gpu,
                "pid": pid,
                "started_at_utc": started,
                "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "status": status,
                "exit_code": process.returncode,
                "log": str(logs_dir / f"{task.run_name}.log"),
            }
            append_jsonl(manifest_path, record)
            if process.returncode != 0:
                append_jsonl(failure_registry, record)
            (status_dir / f"{task.run_name}.json").write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if process.returncode != 0:
                failures += 1
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
