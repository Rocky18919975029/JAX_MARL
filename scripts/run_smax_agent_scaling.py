#!/usr/bin/env python3
"""Run the matched NPS SMAX agent-count scaling experiment.

Both the homogeneous and heterogeneous families use the same ally counts.
Every enemy team has exactly one additional unit.  The only trained conditions
are isolated, C-to-A LN-MSE, and C-to-A Linear CKA.  All runs use four matched
seeds and a 20M-step budget by default.
"""

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
    from run_h1_smax_confirmatory import parse_seeds
    from run_smax_alignment_benchmark import hydra_value, load_alignment_calibration
except ModuleNotFoundError:
    from scripts.h1_protocol import FROZEN_KEYS, repository_root
    from scripts.run_h1_smax_confirmatory import parse_seeds
    from scripts.run_smax_alignment_benchmark import (
        hydra_value,
        load_alignment_calibration,
    )


PROTOCOL_VERSION = "smax-agent-scaling-nps-20m-v1.0"
TOTAL_TIMESTEPS = 20_000_000
DEFAULT_PROJECT = "jaxmarl-smax-agent-scaling-nps-20m"
AGENT_COUNTS = (3, 5, 8, 10, 15)
FAMILY_MAPS = {
    "homogeneous": {
        3: "3m_vs_4m",
        5: "5m_vs_6m",
        8: "8m_vs_9m",
        10: "10m_vs_11m",
        15: "15m_vs_16m",
    },
    "heterogeneous": {
        3: "1s2z_vs_1s3z",
        5: "2s3z_vs_2s4z",
        8: "3s5z_vs_3s6z",
        10: "4s6z_vs_4s7z",
        15: "6s9z_vs_6s10z",
    },
}
FAMILY_LABELS = {"homogeneous": "hom", "heterogeneous": "het"}
CONDITIONS = (
    ("none", "none", "ln_mse", 0.1, "none"),
    ("c_to_a_mse", "c_to_a", "ln_mse", 0.1, "c_to_a"),
    ("c_to_a_cka", "c_to_a", "linear_cka", None, "c_to_a_cka"),
)


@dataclass(frozen=True)
class Task:
    family: str
    agent_count: int
    map_name: str
    seed: int
    display_condition: str
    align_mode: str
    align_distance: str
    alignment_coef: float
    experiment_condition: str

    @property
    def distance_label(self):
        return "distance_free" if self.align_mode == "none" else self.align_distance

    @property
    def key(self):
        return self.map_name, self.distance_label, self.align_mode, self.seed

    @property
    def run_name(self):
        return (
            f"SMAXSCALE-{FAMILY_LABELS[self.family]}-n{self.agent_count}-"
            f"{self.map_name}-nps-{self.display_condition}-seed{self.seed}"
        )


def parse_csv(value, allowed):
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(values) - set(allowed))
    if not values or unknown or len(values) != len(set(values)):
        detail = (
            f"unknown values: {', '.join(unknown)}"
            if unknown
            else "empty/duplicate list"
        )
        raise argparse.ArgumentTypeError(detail)
    return values


def parse_counts(value):
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("Agent counts must be integers") from error
    unknown = sorted(set(values) - set(AGENT_COUNTS))
    if not values or unknown or len(values) != len(set(values)):
        detail = f"unsupported counts: {unknown}" if unknown else "empty/duplicate list"
        raise argparse.ArgumentTypeError(detail)
    return values


def task_matrix(
    families=tuple(FAMILY_MAPS),
    agent_counts=AGENT_COUNTS,
    seeds=(1, 2, 3, 4),
    cka_coef=0.3515769798,
):
    if not (cka_coef > 0):
        raise ValueError("CKA coefficient must be positive")
    tasks = []
    for agent_count in agent_counts:
        for family in families:
            map_name = FAMILY_MAPS[family][agent_count]
            for seed in seeds:
                for display, mode, distance, coefficient, experiment in CONDITIONS:
                    tasks.append(
                        Task(
                            family=family,
                            agent_count=agent_count,
                            map_name=map_name,
                            seed=seed,
                            display_condition=display,
                            align_mode=mode,
                            align_distance=distance,
                            alignment_coef=(
                                cka_coef if distance == "linear_cka" else coefficient
                            ),
                            experiment_condition=experiment,
                        )
                    )
    if len(tasks) != len({task.run_name for task in tasks}):
        raise AssertionError("Scaling run names are not unique")
    return tasks


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
    dirty = subprocess.run(
        ("git", "status", "--porcelain=v1"),
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return commit, dirty


def load_effective_frozen_config(path):
    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    frozen = dict(payload.get("training_config", payload))
    missing = [key for key in FROZEN_KEYS if key not in frozen]
    if missing:
        raise ValueError(f"Frozen config is missing: {', '.join(missing)}")
    if frozen["MATCHED_COMPARISON"] is not True:
        raise ValueError("Agent scaling requires MATCHED_COMPARISON=true")
    if float(frozen["ALIGNMENT_COEF"]) != 0.1:
        raise ValueError("LN-MSE reference coefficient must be 0.1")
    frozen["TOTAL_TIMESTEPS"] = TOTAL_TIMESTEPS
    digest = hashlib.sha256(
        json.dumps(frozen, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return frozen, resolved, digest


def configs_match(config, frozen, task):
    for key in FROZEN_KEYS:
        expected = task.alignment_coef if key == "ALIGNMENT_COEF" else frozen[key]
        actual = config.get(key)
        if key == "ALIGNMENT_COEF":
            try:
                if abs(float(actual) - float(expected)) >= 1e-12:
                    return False
            except (TypeError, ValueError):
                return False
        elif actual != expected:
            return False
    return True


def reusable_tasks(roots, tasks, frozen, exclude_root=None):
    targets = {task.key: task for task in tasks}
    matches = {}
    for root in roots:
        metadata_paths = root.expanduser().resolve().rglob("final/metadata.json")
        for metadata_path in sorted(metadata_paths):
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
            if metadata.get("actor_parameter_sharing") is not False:
                continue
            mode = metadata.get("align_mode", config.get("ALIGN_MODE"))
            distance = metadata.get("align_distance", config.get("ALIGN_DISTANCE"))
            map_name = metadata.get("map_name", config.get("MAP_NAME"))
            seed = metadata.get("seed", config.get("SEED"))
            if None in (mode, distance, map_name, seed):
                continue
            distance = "distance_free" if mode == "none" else distance
            key = map_name, distance, mode, int(seed)
            task = targets.get(key)
            if task is None or not configs_match(config, frozen, task):
                continue
            matches.setdefault(key, directory.resolve())
    return matches


def local_completed_checkpoint(run_root, task, frozen):
    patterns = (
        f"**/{task.run_name}-*/final/model.safetensors",
        f"**/{task.run_name}/final/model.safetensors",
    )
    for pattern in patterns:
        for model_path in (run_root / "checkpoints").glob(pattern):
            config_path = model_path.parent / "config.json"
            if not config_path.is_file():
                continue
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if configs_match(config, frozen, task):
                return model_path.parent.resolve()
    return None


def build_command(args, frozen, task, commit):
    command = [
        sys.executable,
        str(repository_root() / "baselines" / "MAPPO" / "mappo_rnn_smax.py"),
    ]
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
            "ACTOR_PARAMETER_SHARING=false",
            f"ALIGN_MODE={task.align_mode}",
            f"ALIGN_DISTANCE={task.align_distance}",
            "ALIGN_DISTANCE_EPS=1e-8",
            "ALIGN_GRADIENT_CALIBRATION=false",
            "ALIGN_TARGET_SHUFFLE=false",
            f"EXPERIMENT_CONDITION={task.experiment_condition}",
            f"MATRIX_PROFILE={PROTOCOL_VERSION}",
            f"PROTOCOL_VERSION={PROTOCOL_VERSION}",
            f"GIT_COMMIT={commit}",
            "SAVE_CHECKPOINTS=true",
            f"CHECKPOINT_INTERVAL_TIMESTEPS={args.checkpoint_interval}",
            f"CHECKPOINT_DIR={args.run_root / 'checkpoints'}",
            "WANDB_UPLOAD_CHECKPOINTS=false",
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
    parser.add_argument(
        "--families",
        type=lambda value: parse_csv(value, FAMILY_MAPS),
        default=tuple(FAMILY_MAPS),
    )
    parser.add_argument("--agent-counts", type=parse_counts, default=AGENT_COUNTS)
    parser.add_argument("--seeds", type=parse_seeds, default=(1, 2, 3, 4))
    parser.add_argument("--reuse-root", type=Path, action="append", default=[])
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--max-runs-per-gpu", type=int, default=2)
    parser.add_argument("--checkpoint-interval", type=int, default=500_000)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--rerun-successful", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    gpu_ids = parse_csv(args.gpus, tuple(str(index) for index in range(64)))
    if args.max_runs_per_gpu < 1 or args.checkpoint_interval < 1:
        parser.error("Concurrency and checkpoint interval must be positive")
    repo = repository_root()
    commit, dirty = git_state(repo)
    if dirty:
        raise RuntimeError("Worktree is dirty; commit the scaling protocol first")
    frozen, frozen_path, frozen_digest = load_effective_frozen_config(
        args.frozen_config
    )
    cka_coef = load_alignment_calibration(args.cka_calibration, "linear_cka")
    tasks = task_matrix(args.families, args.agent_counts, args.seeds, cka_coef)
    args.run_root = args.run_root.expanduser().resolve()
    reusable = reusable_tasks(args.reuse_root, tasks, frozen, args.run_root)
    local_completed = {
        task.key: local_completed_checkpoint(args.run_root, task, frozen)
        for task in tasks
    }

    if args.dry_run:
        completed_count = sum(
            path is not None for path in local_completed.values()
        )
        pending_count = sum(
            task.key not in reusable and local_completed[task.key] is None
            for task in tasks
        )
        print(
            f"Protocol={PROTOCOL_VERSION} selected={len(tasks)} "
            f"reused={len(reusable)} "
            f"completed_local={completed_count} pending={pending_count} "
            f"total_timesteps={TOTAL_TIMESTEPS} cka_lambda={cka_coef}"
        )
        pending_index = 0
        for task in tasks:
            if task.key in reusable:
                print(f"REUSE: {task.run_name} <- {reusable[task.key]}")
            elif local_completed[task.key] is not None:
                print(f"DONE:  {task.run_name}")
            else:
                print(f"GPU {gpu_ids[pending_index % len(gpu_ids)]}: {task.run_name}")
                pending_index += 1
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
        "effective_frozen_config_sha256": frozen_digest,
        "total_timesteps": TOTAL_TIMESTEPS,
        "checkpoint_interval": args.checkpoint_interval,
        "families": list(args.families),
        "agent_counts": list(args.agent_counts),
        "maps": {
            family: {
                str(count): FAMILY_MAPS[family][count] for count in args.agent_counts
            }
            for family in args.families
        },
        "actor_parameterization": "nps",
        "conditions": [item[0] for item in CONDITIONS],
        "seeds": list(args.seeds),
        "cka_calibration": str(args.cka_calibration.expanduser().resolve()),
        "cka_alignment_coef": cka_coef,
        "ln_mse_alignment_coef": 0.1,
        "unique_runs": len(tasks),
        "reuse_roots": [str(path.expanduser().resolve()) for path in args.reuse_root],
        "reused_runs": len(reusable),
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
    queues = {gpu: collections.deque() for gpu in gpu_ids}
    skipped_local = 0
    pending_index = 0
    for task in tasks:
        if task.key in reusable:
            continue
        if local_completed[task.key] is not None and not args.rerun_successful:
            skipped_local += 1
            continue
        marker = directories["status"] / f"{task.run_name}.json"
        if marker.is_file() and not args.rerun_successful:
            prior = json.loads(marker.read_text(encoding="utf-8"))
            if prior.get("status") == "completed":
                skipped_local += 1
                continue
        queues[gpu_ids[pending_index % len(gpu_ids)]].append(task)
        pending_index += 1
    append_log(
        launcher_log,
        f"protocol={PROTOCOL_VERSION} selected={len(tasks)} "
        f"pending={sum(map(len, queues.values()))} reused={len(reusable)} "
        f"skipped_local={skipped_local}",
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
        for gpu in gpu_ids:
            active = sum(item[1] == gpu for item in running.values())
            while queues[gpu] and active < args.max_runs_per_gpu:
                task = queues[gpu].popleft()
                log_path = directories["logs"] / f"{task.run_name}.log"
                handle = log_path.open("a", encoding="utf-8")
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
                            f"SMAXSCALE-{FAMILY_LABELS[task.family]}-"
                            f"n{task.agent_count}-{task.display_condition}"
                        ),
                        "WANDB_TAGS": ",".join(
                            (
                                PROTOCOL_VERSION,
                                "smax",
                                "agent-scaling",
                                task.family,
                                f"agents-{task.agent_count}",
                                task.map_name,
                                "nps",
                                task.display_condition,
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
            log_path = directories["logs"] / f"{task.run_name}.log"
            status = "completed" if process.returncode == 0 else "failed"
            record = {
                "schema_version": 1,
                "protocol_version": PROTOCOL_VERSION,
                "git_commit": commit,
                "run_name": task.run_name,
                "family": task.family,
                "agent_count": task.agent_count,
                "map_name": task.map_name,
                "actor_parameterization": "nps",
                "condition": task.display_condition,
                "align_mode": task.align_mode,
                "align_distance": task.distance_label,
                "alignment_coef": task.alignment_coef,
                "seed": task.seed,
                "gpu": gpu,
                "started_at_utc": started,
                "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "status": status,
                "exit_code": process.returncode,
                "log": str(log_path),
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
