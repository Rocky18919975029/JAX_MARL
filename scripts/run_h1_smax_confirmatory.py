#!/usr/bin/env python3
"""Launch a frozen H1 SMAX matrix with bounded per-GPU concurrency."""

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

try:
    from h1_protocol import FROZEN_KEYS, PROTOCOL_VERSION, repository_root
except ModuleNotFoundError:  # Imported as scripts.run_h1_smax_confirmatory in tests.
    from scripts.h1_protocol import FROZEN_KEYS, PROTOCOL_VERSION, repository_root


MAPS = ("10m_vs_11m", "smacv2_10_units")
ACTOR_VARIANTS = (("nps", False),)
LN_MSE_CONDITIONS = ("none", "a_to_c", "c_to_a")
CKA_CONDITIONS = ("a_to_c_cka", "c_to_a_cka")
CONDITIONS = LN_MSE_CONDITIONS + CKA_CONDITIONS
SEEDS = (1, 2, 3, 4)
MATRIX_PROFILES = ("nps-ln-mse", "nps-linear-cka")


@dataclass(frozen=True)
class Task:
    matrix_profile: str
    map_name: str
    actor_label: str
    sharing: bool
    condition: str
    seed: int
    alignment_coef: float = 0.1

    @property
    def align_mode(self):
        return self.condition.removesuffix("_cka")

    @property
    def align_distance(self):
        return "linear_cka" if self.condition.endswith("_cka") else "ln_mse"

    @property
    def lambda_label(self):
        value = f"{self.alignment_coef:.10g}".replace("-", "m").replace(".", "p")
        return f"lam{value}"

    @property
    def run_name(self):
        return (
            f"H1-nps-{self.map_name}-{self.condition}-"
            f"{self.lambda_label}-seed{self.seed}"
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
    if any(seed < 0 for seed in seeds):
        raise argparse.ArgumentTypeError("seeds must be non-negative")
    return tuple(seeds)


def validate_matrix_profile(args):
    conditions = (
        CKA_CONDITIONS if args.matrix_profile == "nps-linear-cka" else LN_MSE_CONDITIONS
    )
    run_count = 16 if args.matrix_profile == "nps-linear-cka" else 24
    expected = {
        "maps": MAPS,
        "actor variants": ("nps",),
        "conditions": conditions,
        "seeds": SEEDS,
    }
    actual = {
        "maps": args.maps,
        "actor variants": args.actor_variants,
        "conditions": args.conditions,
        "seeds": args.seeds,
    }
    mismatches = [
        f"{name}: expected {expected[name]}, got {actual[name]}"
        for name in expected
        if tuple(actual[name]) != tuple(expected[name])
    ]
    if mismatches:
        raise ValueError(
            f"The {args.matrix_profile} profile is locked to exactly "
            f"{run_count} runs:\n" + "\n".join(mismatches)
        )


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
            f"MATRIX_PROFILE={args.matrix_profile}",
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
        Task(
            args.matrix_profile,
            map_name,
            actor_label,
            actor_lookup[actor_label],
            condition,
            seed,
            (args.cka_alignment_coef if condition.endswith("_cka") else 0.1),
        )
        for map_name in args.maps
        for actor_label in args.actor_variants
        for condition in args.conditions
        for seed in args.seeds
    ]


def load_cka_calibration(path):
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if payload.get("selection_uses_return") is not False:
        raise ValueError("CKA calibration must explicitly record no return selection")
    if payload.get("performance_fields_persisted") is not False:
        raise ValueError("CKA calibration artifact must exclude performance fields")
    if payload.get("reference_distance") != "ln_mse":
        raise ValueError("CKA calibration reference must be ln_mse")
    if float(payload.get("reference_alignment_coef", -1)) != 0.1:
        raise ValueError("CKA calibration must reference LN-MSE lambda=0.1")
    if payload.get("target_distance") != "linear_cka":
        raise ValueError("Calibration target must be linear_cka")
    coefficient = float(payload["global_alignment_coef"])
    if not (coefficient > 0 and coefficient < float("inf")):
        raise ValueError("Calibrated CKA coefficient must be finite and positive")
    return coefficient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--frozen-config",
        type=Path,
        help="defaults to RUN_ROOT/protocol/frozen_training_config.json",
    )
    parser.add_argument(
        "--matrix-profile", choices=MATRIX_PROFILES, default="nps-ln-mse"
    )
    parser.add_argument("--project", default=None)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--max-runs-per-gpu", type=int, default=5)
    parser.add_argument("--seeds", type=parse_seeds, default=SEEDS)
    parser.add_argument(
        "--maps",
        type=lambda value: parse_csv(value, MAPS),
        default=MAPS,
    )
    parser.add_argument(
        "--actor-variants",
        type=lambda value: parse_csv(value, dict(ACTOR_VARIANTS)),
        default=("nps",),
    )
    parser.add_argument(
        "--conditions",
        type=lambda value: parse_csv(value, CONDITIONS),
        default=None,
    )
    parser.add_argument(
        "--cka-calibration",
        type=Path,
        help=(
            "JSON produced by calibrate_h1_cka.py; required when a linear-CKA "
            "condition is selected"
        ),
    )
    parser.add_argument("--upload-checkpoints", action="store_true")
    parser.add_argument("--allow-git-mismatch", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--rerun-successful", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.conditions is None:
        args.conditions = (
            CKA_CONDITIONS
            if args.matrix_profile == "nps-linear-cka"
            else LN_MSE_CONDITIONS
        )
    validate_matrix_profile(args)
    if args.project is None:
        args.project = f"h1-smax-{args.matrix_profile}"

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
    uses_cka = any(condition.endswith("_cka") for condition in args.conditions)
    if uses_cka:
        if args.cka_calibration is None:
            raise ValueError(
                "Linear-CKA conditions require --cka-calibration; do not select "
                "lambda from returns or copy the LN-MSE coefficient."
            )
        args.cka_alignment_coef = load_cka_calibration(args.cka_calibration)
    else:
        args.cka_alignment_coef = None

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
    expected_profile_sizes = {
        "nps-ln-mse": 24,
        "nps-linear-cka": 16,
    }
    expected_size = expected_profile_sizes.get(args.matrix_profile)
    if expected_size is not None and len(tasks) != expected_size:
        raise AssertionError(
            f"{args.matrix_profile} must contain {expected_size} runs, "
            f"got {len(tasks)}"
        )
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
        f"profile={args.matrix_profile} protocol={PROTOCOL_VERSION} "
        f"commit={commit} selected={len(tasks)} "
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
                            f"H1-nps-{task.map_name}-{task.condition}-{task.lambda_label}"
                        ),
                        "WANDB_TAGS": ",".join(
                            (
                                args.matrix_profile,
                                "smax",
                                task.map_name,
                                task.actor_label,
                                task.condition,
                                f"distance-{task.align_distance}",
                                f"lambda-{task.alignment_coef:.10g}",
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
                "matrix_profile": args.matrix_profile,
                "protocol_version": PROTOCOL_VERSION,
                "git_commit": commit,
                "run_name": task.run_name,
                "map_name": task.map_name,
                "actor_parameterization": task.actor_label,
                "actor_parameter_sharing": task.sharing,
                "condition": task.condition,
                "align_mode": task.align_mode,
                "align_distance": task.align_distance,
                "alignment_coef": task.alignment_coef,
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
