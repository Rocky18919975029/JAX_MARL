#!/usr/bin/env python3
"""Run the matched four-seed alignment matrix on benchmark SMAX tasks."""

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
except ModuleNotFoundError:  # Imported as scripts.run_smax_alignment_benchmark.
    from scripts.h1_protocol import FROZEN_KEYS, repository_root
    from scripts.run_h1_smax_confirmatory import parse_seeds


PROTOCOL_VERSION = "smax-alignment-benchmark-4seed-v1.0"
BENCHMARK_MAPS = ("2s3z", "3s5z_vs_3s6z", "smacv2_10_units", "6h_vs_8z")
# smacv2_10_units already belongs to the completed H1 matrix. This launcher
# defaults to the other maps explicitly listed by run_minimal_baseline_set.yaml.
DEFAULT_MAPS = tuple(name for name in BENCHMARK_MAPS if name != "smacv2_10_units")
ACTOR_VARIANTS = {"ps": True, "nps": False}
DISTANCES = ("ln_mse", "linear_cka")
AVAILABLE_DISTANCES = (*DISTANCES, "containment")
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
        suffix = {
            "ln_mse": "",
            "linear_cka": "_cka",
            "containment": "_dsc",
        }[self.align_distance]
        return f"{self.align_mode}{suffix}"

    @property
    def lambda_label(self):
        value = f"{self.alignment_coef:.10g}".replace("-", "m").replace(".", "p")
        return f"lam{value}"

    @property
    def run_name(self):
        return (
            f"SMAXB4-{self.map_name}-{self.actor_label}-{self.condition}-"
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
        raise ValueError("The benchmark matrix requires MATCHED_COMPARISON=true")
    if float(frozen["ALIGNMENT_COEF"]) != 0.1:
        raise ValueError("The LN-MSE reference coefficient must be 0.1")
    digest = hashlib.sha256(
        json.dumps(frozen, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return frozen, resolved, digest


def task_matrix(
    maps,
    actors,
    seeds,
    cka_alignment_coef,
    distances=DISTANCES,
    containment_alignment_coef=None,
):
    tasks = []
    for map_name in maps:
        for actor_label in actors:
            sharing = ACTOR_VARIANTS[actor_label]
            for seed in seeds:
                # ALIGN_MODE=none is distance-free and is trained exactly once.
                tasks.append(
                    Task(map_name, actor_label, sharing, "ln_mse", "none", seed, 0.1)
                )
                for distance in distances:
                    coefficient = {
                        "ln_mse": 0.1,
                        "linear_cka": cka_alignment_coef,
                        "containment": containment_alignment_coef,
                    }[distance]
                    if coefficient is None:
                        raise ValueError(
                            f"Missing calibrated coefficient for {distance}"
                        )
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


def load_alignment_calibration(path, target_distance):
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if payload.get("selection_uses_return") is not False:
        raise ValueError("Calibration must explicitly record no return selection")
    if payload.get("reference_distance") != "ln_mse":
        raise ValueError("Calibration reference must be ln_mse")
    if float(payload.get("reference_alignment_coef", -1)) != 0.1:
        raise ValueError("Calibration must reference LN-MSE lambda=0.1")
    if payload.get("target_distance") != target_distance:
        raise ValueError(
            f"Expected {target_distance} calibration, got "
            f"{payload.get('target_distance')!r}"
        )
    coefficient = float(payload["global_alignment_coef"])
    if not (coefficient > 0 and coefficient < float("inf")):
        raise ValueError("Calibrated coefficient must be finite and positive")
    return coefficient


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
            matched = metadata.get(
                "matched_comparison", config.get("MATCHED_COMPARISON")
            )
            if matched is not True:
                continue
            if not matching_frozen_config(config, frozen):
                continue
            sharing = metadata.get(
                "actor_parameter_sharing", config.get("ACTOR_PARAMETER_SHARING")
            )
            align_mode = metadata.get("align_mode", config.get("ALIGN_MODE"))
            map_name = metadata.get("map_name", config.get("MAP_NAME"))
            seed = metadata.get("seed", config.get("SEED"))
            coefficient = metadata.get(
                "alignment_coef", config.get("ALIGNMENT_COEF")
            )
            if None in (sharing, align_mode, map_name, seed, coefficient):
                continue
            actor_label = "ps" if sharing else "nps"
            raw_distance = metadata.get(
                "align_distance", config.get("ALIGN_DISTANCE")
            )
            if raw_distance is None:
                condition = metadata.get(
                    "condition", config.get("EXPERIMENT_CONDITION", "")
                )
                # ALIGN_DISTANCE did not exist before Linear CKA support. Such
                # legacy aligned checkpoints are necessarily LN-MSE unless the
                # explicitly recorded condition carries the later CKA suffix.
                raw_distance = (
                    "linear_cka" if str(condition).endswith("_cka") else "ln_mse"
                )
            distance = "distance_free" if align_mode == "none" else raw_distance
            key = (
                map_name,
                actor_label,
                distance,
                align_mode,
                int(seed),
            )
            target = targets.get(key)
            if target is None:
                continue
            if not abs(float(coefficient) - target.alignment_coef) < 1e-12:
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
            "ALIGN_CONTAINMENT_RIDGE_RATIO=1e-3",
            "ALIGN_CONTAINMENT_EPS=1e-6",
            "ALIGN_CONTAINMENT_GROUP_BY_UNIT_TYPE=true",
            "ALIGN_CONTAINMENT_MIN_GROUP_SAMPLES=64",
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
    parser.add_argument("--cka-calibration", type=Path)
    parser.add_argument("--containment-calibration", type=Path)
    parser.add_argument(
        "--distances",
        type=lambda value: parse_csv(value, AVAILABLE_DISTANCES),
        default=DISTANCES,
    )
    parser.add_argument(
        "--maps",
        type=lambda value: parse_csv(value, BENCHMARK_MAPS),
        default=DEFAULT_MAPS,
        help=(
            "Defaults to the three benchmark maps not already in the H1 matrix: "
            "2s3z,3s5z_vs_3s6z,6h_vs_8z."
        ),
    )
    parser.add_argument(
        "--actor-variants",
        type=lambda value: parse_csv(value, ACTOR_VARIANTS),
        default=tuple(ACTOR_VARIANTS),
    )
    parser.add_argument(
        "--seeds",
        type=parse_seeds,
        default=tuple(range(1, 5)),
        help="Defaults to the matched training seeds 1-4.",
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
    parser.add_argument("--project", default="jaxmarl-smax-alignment-benchmark-4seed")
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
    cka_alignment_coef = None
    containment_alignment_coef = None
    if "linear_cka" in args.distances:
        if args.cka_calibration is None:
            parser.error("--cka-calibration is required for linear_cka")
        cka_alignment_coef = load_alignment_calibration(
            args.cka_calibration, "linear_cka"
        )
    if "containment" in args.distances:
        if args.containment_calibration is None:
            parser.error("--containment-calibration is required for containment")
        containment_alignment_coef = load_alignment_calibration(
            args.containment_calibration, "containment"
        )
    tasks = task_matrix(
        args.maps,
        args.actor_variants,
        args.seeds,
        cka_alignment_coef,
        args.distances,
        containment_alignment_coef,
    )
    args.run_root = args.run_root.expanduser().resolve()
    reusable = reusable_tasks(args.reuse_root, tasks, frozen, args.run_root)
    expected = len(args.maps) * len(args.actor_variants) * len(args.seeds) * (
        1 + 3 * len(args.distances)
    )
    if len(tasks) != expected:
        raise AssertionError(f"Expected {expected} unique runs, found {len(tasks)}")

    if args.dry_run:
        print(
            f"Protocol={PROTOCOL_VERSION} unique_runs={len(tasks)} "
            f"reused={len(reusable)} pending={len(tasks) - len(reusable)} "
            f"seeds={','.join(map(str, args.seeds))} "
            f"cka_lambda={cka_alignment_coef} dsc_lambda={containment_alignment_coef}"
        )
        pending_index = 0
        for task in tasks:
            if task.key in reusable:
                print(f"REUSE: {task.run_name} <- {reusable[task.key]}")
                continue
            print(
                f"GPU {gpu_ids[pending_index % len(gpu_ids)]}: {task.run_name}"
            )
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
        "benchmark_suite_config": "baselines/run_minimal_baseline_set.yaml",
        "git_commit": commit,
        "frozen_config": str(frozen_path),
        "frozen_config_sha256": frozen_digest,
        "cka_calibration": (
            str(args.cka_calibration.expanduser().resolve())
            if args.cka_calibration is not None
            else None
        ),
        "cka_alignment_coef": cka_alignment_coef,
        "maps": args.maps,
        "actor_variants": args.actor_variants,
        "align_modes": ALIGN_MODES,
        "align_distances": args.distances,
        "seeds": args.seeds,
        "unique_runs": len(tasks),
        "reuse_roots": [str(path.expanduser().resolve()) for path in args.reuse_root],
        "reused_runs": len(reusable),
        "none_reuse": "one distance-free run per task/actor/seed",
    }
    if "containment" in args.distances:
        manifest.update(
            {
                "containment_calibration": str(
                    args.containment_calibration.expanduser().resolve()
                ),
                "containment_alignment_coef": containment_alignment_coef,
                "containment_ridge_ratio": 1e-3,
                "containment_epsilon": 1e-6,
                "containment_group_by_unit_type": True,
                "containment_min_group_samples": 64,
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
    pending_index = 0
    for task in tasks:
        if task.key in reusable:
            reused += 1
            continue
        marker = directories["status"] / f"{task.run_name}.json"
        if marker.is_file() and not args.rerun_successful:
            prior = json.loads(marker.read_text(encoding="utf-8"))
            if prior.get("status") == "completed":
                skipped += 1
                continue
        pending_by_gpu[gpu_ids[pending_index % len(gpu_ids)]].append(task)
        pending_index += 1
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
                            f"SMAXB4-{task.map_name}-{task.actor_label}-"
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
