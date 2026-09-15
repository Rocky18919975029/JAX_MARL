#!/usr/bin/env python3
"""Extend the matched SMAX PS/NPS LN-MSE/CKA matrix from 4 to 10 seeds."""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

try:
    from h1_protocol import FROZEN_KEYS, repository_root
    from run_h1_smax_confirmatory import load_cka_calibration, parse_seeds
except ModuleNotFoundError:  # Imported as scripts.run_smax_alignment_10seed.
    from scripts.h1_protocol import FROZEN_KEYS, repository_root
    from scripts.run_h1_smax_confirmatory import load_cka_calibration, parse_seeds


PROTOCOL_VERSION = "smax-alignment-10seed-v1.0"
MAPS = ("10m_vs_11m", "smacv2_10_units")
ACTOR_VARIANTS = {"ps": True, "nps": False}
DISTANCES = ("ln_mse", "linear_cka")
ALIGN_MODES = ("none", "c_to_a", "a_to_c", "joint")
ALIGNED_MODES = ALIGN_MODES[1:]


@dataclass(frozen=True)
class Task:
    map_name: str
    actor_label: str
    sharing: bool
    align_distance: str
    align_mode: str
    seed: int
    alignment_coef: float

    @property
    def distance_label(self):
        return "distance_free" if self.align_mode == "none" else self.align_distance

    @property
    def condition(self):
        if self.align_mode == "none":
            return "none"
        return (
            f"{self.align_mode}_cka"
            if self.align_distance == "linear_cka"
            else self.align_mode
        )

    @property
    def lambda_label(self):
        value = f"{self.alignment_coef:.10g}".replace("-", "m").replace(".", "p")
        return f"lam{value}"

    @property
    def run_name(self):
        return (
            f"SMAX10-{self.map_name}-{self.actor_label}-{self.condition}-"
            f"{self.lambda_label}-seed{self.seed}"
        )

    @property
    def key(self):
        return (
            self.map_name,
            self.actor_label,
            self.distance_label,
            self.align_mode,
            self.seed,
        )


def parse_csv(value, allowed):
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(values) - set(allowed))
    if not values or unknown or len(values) != len(set(values)):
        message = f"unknown values: {', '.join(unknown)}" if unknown else "empty/duplicate list"
        raise argparse.ArgumentTypeError(message)
    return values


def hydra_value(value):
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def append_log(path, message):
    timestamp = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(line + "\n")


def append_jsonl(path, payload):
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, sort_keys=True) + "\n")
        file.flush()


def git_state(repo):
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ("git", "status", "--porcelain=v1"),
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return commit, status


def load_frozen_config(path):
    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    frozen = payload.get("training_config", payload)
    missing = [key for key in FROZEN_KEYS if key not in frozen]
    if missing:
        raise ValueError(f"Frozen config is missing: {', '.join(missing)}")
    if frozen["MATCHED_COMPARISON"] is not True:
        raise ValueError("The extension requires MATCHED_COMPARISON=true")
    if float(frozen["ALIGNMENT_COEF"]) != 0.1:
        raise ValueError("The LN-MSE reference coefficient must be 0.1")
    digest = hashlib.sha256(
        json.dumps(frozen, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return frozen, resolved, digest


def task_matrix(maps, actors, seeds, cka_alignment_coef):
    tasks = []
    for map_name in maps:
        for actor_label in actors:
            sharing = ACTOR_VARIANTS[actor_label]
            for seed in seeds:
                # ALIGN_MODE=none is distance-free and is trained exactly once.
                tasks.append(
                    Task(map_name, actor_label, sharing, "ln_mse", "none", seed, 0.1)
                )
                for distance in DISTANCES:
                    coefficient = 0.1 if distance == "ln_mse" else cka_alignment_coef
                    for align_mode in ALIGNED_MODES:
                        tasks.append(
                            Task(
                                map_name,
                                actor_label,
                                sharing,
                                distance,
                                align_mode,
                                seed,
                                coefficient,
                            )
                        )
    if len(tasks) != len(set(task.run_name for task in tasks)):
        raise AssertionError("Run names are not unique")
    return tasks


def matching_frozen_config(config, frozen):
    return all(
        key == "ALIGNMENT_COEF" or config.get(key) == value
        for key, value in frozen.items()
        if key in FROZEN_KEYS
    )


def reusable_tasks(roots, tasks, frozen, exclude_root=None):
    targets = {task.key: task for task in tasks}
    matches = {}
    for root in roots:
        for metadata_path in sorted(
            root.expanduser().resolve().rglob("final/metadata.json")
        ):
            directory = metadata_path.parent
            if exclude_root is not None:
                try:
                    directory.relative_to(exclude_root)
                except ValueError:
                    pass
                else:
                    continue
            config_path = directory / "config.json"
            model_path = directory / "model.safetensors"
            if not config_path.is_file() or not model_path.is_file():
                continue
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if metadata.get("matched_comparison") is not True:
                continue
            if not matching_frozen_config(config, frozen):
                continue
            sharing = bool(metadata["actor_parameter_sharing"])
            actor_label = "ps" if sharing else "nps"
            align_mode = metadata["align_mode"]
            distance = (
                "distance_free" if align_mode == "none" else metadata["align_distance"]
            )
            key = (
                metadata["map_name"],
                actor_label,
                distance,
                align_mode,
                int(metadata["seed"]),
            )
            target = targets.get(key)
            if target is None:
                continue
            if not abs(float(metadata["alignment_coef"]) - target.alignment_coef) < 1e-12:
                continue
            matches.setdefault(key, directory)
    return matches


def build_command(args, frozen, task, commit):
    script = repository_root() / "baselines" / "MAPPO" / "mappo_rnn_smax.py"
    command = [sys.executable, str(script)]
    for key in FROZEN_KEYS:
        if key == "ENV_KWARGS":
            for nested_key, value in frozen[key].items():
                command.append(f"ENV_KWARGS.{nested_key}={hydra_value(value)}")
        else:
            value = task.alignment_coef if key == "ALIGNMENT_COEF" else frozen[key]
            command.append(f"{key}={hydra_value(value)}")
    command.extend(
        (
            f"MAP_NAME={task.map_name}",
            f"SEED={task.seed}",
            f"ACTOR_PARAMETER_SHARING={hydra_value(task.sharing)}",
            f"ALIGN_MODE={task.align_mode}",
            f"ALIGN_DISTANCE={task.align_distance}",
            "ALIGN_DISTANCE_EPS=1e-8",
            "ALIGN_GRADIENT_CALIBRATION=false",
            "ALIGN_TARGET_SHUFFLE=false",
            "ALIGN_TARGET_SHUFFLE_SCOPE=same_agent_env_time",
            "ALIGN_SHUFFLE_SEED_OFFSET=700000",
            f"EXPERIMENT_CONDITION={task.condition}",
            f"MATRIX_PROFILE={PROTOCOL_VERSION}",
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--frozen-config", type=Path, required=True)
    parser.add_argument("--cka-calibration", type=Path, required=True)
    parser.add_argument("--maps", type=lambda value: parse_csv(value, MAPS), default=MAPS)
    parser.add_argument(
        "--actor-variants",
        type=lambda value: parse_csv(value, ACTOR_VARIANTS),
        default=tuple(ACTOR_VARIANTS),
    )
    parser.add_argument(
        "--seeds",
        type=parse_seeds,
        default=tuple(range(1, 11)),
        help="Defaults to 1-10; exact completed checkpoints can be reused.",
    )
    parser.add_argument(
        "--reuse-root",
        type=Path,
        action="append",
        default=[],
        help=(
            "Recursively reuse exact completed final checkpoints under this root. "
            "May be repeated."
        ),
    )
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--max-runs-per-gpu", type=int, default=5)
    parser.add_argument("--project", default="jaxmarl-smax-alignment-10seed")
    parser.add_argument("--upload-checkpoints", action="store_true")
    parser.add_argument("--rerun-successful", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    gpu_ids = parse_csv(args.gpus, tuple(str(index) for index in range(64)))
    if args.max_runs_per_gpu < 1:
        parser.error("--max-runs-per-gpu must be positive")
    repo = repository_root()
    commit, dirty = git_state(repo)
    if dirty:
        raise RuntimeError("Worktree is dirty; commit the experiment protocol first")
    frozen, frozen_path, frozen_digest = load_frozen_config(args.frozen_config)
    cka_alignment_coef = load_cka_calibration(args.cka_calibration)
    tasks = task_matrix(args.maps, args.actor_variants, args.seeds, cka_alignment_coef)
    args.run_root = args.run_root.expanduser().resolve()
    reusable = reusable_tasks(args.reuse_root, tasks, frozen, args.run_root)
    expected = len(args.maps) * len(args.actor_variants) * len(args.seeds) * 7
    if len(tasks) != expected:
        raise AssertionError(f"Expected {expected} unique runs, found {len(tasks)}")

    if args.dry_run:
        print(
            f"Protocol={PROTOCOL_VERSION} unique_runs={len(tasks)} "
            f"reused={len(reusable)} pending={len(tasks) - len(reusable)} "
            f"seeds={','.join(map(str, args.seeds))} "
            f"cka_lambda={cka_alignment_coef:.10g}"
        )
        for index, task in enumerate(tasks):
            if task.key in reusable:
                print(f"REUSE: {task.run_name} <- {reusable[task.key]}")
                continue
            print(f"GPU {gpu_ids[index % len(gpu_ids)]}: {task.run_name}")
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
        "frozen_config": str(frozen_path),
        "frozen_config_sha256": frozen_digest,
        "cka_calibration": str(args.cka_calibration.expanduser().resolve()),
        "cka_alignment_coef": cka_alignment_coef,
        "maps": args.maps,
        "actor_variants": args.actor_variants,
        "align_modes": ALIGN_MODES,
        "align_distances": DISTANCES,
        "seeds": args.seeds,
        "unique_runs": len(tasks),
        "reuse_roots": [str(path.expanduser().resolve()) for path in args.reuse_root],
        "reused_runs": len(reusable),
        "none_reuse": "one distance-free run per task/actor/seed",
    }
    manifest_path = args.run_root / "experiment_manifest.json"
    if manifest_path.is_file():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise RuntimeError(f"Experiment settings changed: {manifest_path}")
    else:
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    (args.run_root / "reused_runs.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "runs": {
                    "|".join(map(str, key)): str(path)
                    for key, path in sorted(reusable.items())
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    launcher_log = args.run_root / "launcher.log"
    completion_manifest = args.run_root / "completion_manifest.jsonl"
    failure_registry = args.run_root / "failure_registry.jsonl"
    pending_by_gpu = {gpu: collections.deque() for gpu in gpu_ids}
    skipped = 0
    reused = 0
    for index, task in enumerate(tasks):
        if task.key in reusable:
            reused += 1
            continue
        marker = directories["status"] / f"{task.run_name}.json"
        if marker.is_file() and not args.rerun_successful:
            prior = json.loads(marker.read_text(encoding="utf-8"))
            if prior.get("status") == "completed":
                skipped += 1
                continue
        pending_by_gpu[gpu_ids[index % len(gpu_ids)]].append(task)
    append_log(
        launcher_log,
        f"protocol={PROTOCOL_VERSION} selected={len(tasks)} "
        f"pending={sum(map(len, pending_by_gpu.values()))} "
        f"reused={reused} skipped_local={skipped}",
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
    while any(pending_by_gpu.values()) or running:
        if stop_requested:
            for process, _, _, handle, _ in running.values():
                process.terminate()
                handle.close()
            raise SystemExit(130)
        for gpu in gpu_ids:
            active = sum(item[1] == gpu for item in running.values())
            while pending_by_gpu[gpu] and active < args.max_runs_per_gpu:
                task = pending_by_gpu[gpu].popleft()
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
                            f"SMAX10-{task.map_name}-{task.actor_label}-"
                            f"{task.condition}-{task.lambda_label}"
                        ),
                        "WANDB_TAGS": ",".join(
                            (
                                PROTOCOL_VERSION,
                                "smax",
                                task.map_name,
                                task.actor_label,
                                task.condition,
                                f"distance-{task.distance_label}",
                                f"lambda-{task.alignment_coef:.10g}",
                            )
                        ),
                    }
                )
                started = dt.datetime.now(dt.timezone.utc).isoformat()
                process = subprocess.Popen(
                    build_command(args, frozen, task, commit),
                    cwd=repo,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
                running[process.pid] = (process, gpu, task, handle, started)
                append_log(
                    launcher_log,
                    f"GPU {gpu} START {task.run_name} pid={process.pid}",
                )
                active += 1

        completed = [pid for pid, item in running.items() if item[0].poll() is not None]
        for pid in completed:
            process, gpu, task, handle, started = running.pop(pid)
            handle.close()
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
                "align_distance": task.distance_label,
                "alignment_coef": task.alignment_coef,
                "seed": task.seed,
                "gpu": gpu,
                "started_at_utc": started,
                "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "status": status,
                "exit_code": process.returncode,
                "log": str(directories["logs"] / f"{task.run_name}.log"),
            }
            append_jsonl(completion_manifest, record)
            if process.returncode:
                failures += 1
                append_jsonl(failure_registry, record)
            (directories["status"] / f"{task.run_name}.json").write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            append_log(
                launcher_log,
                f"GPU {gpu} END   {task.run_name} status={process.returncode}",
            )
        if not completed:
            time.sleep(1)
    append_log(launcher_log, f"all selected runs finished; failures={failures}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
