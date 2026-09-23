#!/usr/bin/env python3
"""Frozen, seed-paired SMAX NPS sweeps: none, C→A MSE/CKA, and ARec.

This launcher never resumes an optimizer state. Completed runs are verified and
skipped; interrupted/failed runs require a fresh run root to avoid mixing data.
"""

from __future__ import annotations

import argparse
import ast
import collections
import csv
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path


PROTOCOL = "smax-nps-four-method-v1"
METHODS = ("none", "mse", "cka", "arec")
CONDITIONS = {"none": "none", "mse": "c_to_a", "cka": "c_to_a_cka", "arec": "arec"}
REPO = Path(__file__).resolve().parents[1]


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def comma_strings(raw: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in raw.split(","))
    if not values or any(not part for part in values) or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("Expected distinct, nonempty CSV values")
    return values


def positive_floats(raw: str) -> tuple[float, ...]:
    values = tuple(float(value) for value in comma_strings(raw))
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise argparse.ArgumentTypeError("Grid values must be finite and positive")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("Duplicate grid values")
    return values


def positive_ints(raw: str) -> tuple[int, ...]:
    values = tuple(int(value) for value in comma_strings(raw))
    if any(value <= 0 for value in values) or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("Grid values must be distinct positive integers")
    return values


def map_names() -> set[str]:
    """Validate against the checked-out SMAX source without importing JAX."""
    source = REPO / "jaxmarl/environments/smax/smax_env.py"
    for node in ast.parse(source.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "MAP_NAME_TO_SCENARIO"
            for target in node.targets
        ):
            return {key.value for key in node.value.keys if isinstance(key, ast.Constant)}
    raise RuntimeError("Cannot locate the SMAX map registry")


@dataclass(frozen=True)
class Run:
    map_name: str
    seed: int
    method: str
    timesteps: int
    lr: float
    epochs: int
    num_envs: int
    num_minibatches: int
    num_steps: int
    coef: float = 0.0
    q_steps: int = 0
    q_lr: float = 0.0
    fisher_ridge: float = 0.0

    @property
    def ident(self) -> str:
        return hashlib.sha256(canonical(asdict(self)).encode()).hexdigest()[:12]

    @property
    def name(self) -> str:
        return f"SMAX4-{self.map_name}-nps-{self.method}-seed{self.seed}-{self.ident}"

    @property
    def condition(self) -> str:
        return CONDITIONS[self.method]

    @property
    def group(self) -> str:
        # Every training seed with this exact grid cell is grouped together.
        cell = dict(asdict(self), seed=0)
        tag = hashlib.sha256(canonical(cell).encode()).hexdigest()[:10]
        return f"SMAX4-{self.map_name}-{self.method}-{tag}"


def suite_id(root: Path) -> str:
    return hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:10]


def wandb_id(root: Path, run: Run) -> str:
    # Fresh run roots must not reuse W&B IDs from failed/earlier sweeps.
    return hashlib.sha256(f"{suite_id(root)}:{run.ident}".encode()).hexdigest()[:12]


def group_for(root: Path, run: Run) -> str:
    return f"{run.group}-{suite_id(root)}"


def make_grid(args: argparse.Namespace) -> list[Run]:
    combos = tuple(itertools.product(
        args.ppo_lrs, args.ppo_epochs, args.num_envs_grid, args.num_minibatches_grid
    ))
    if any(envs % minibatches for _, _, envs, minibatches in combos):
        raise ValueError("Every NUM_ENVS value must be divisible by every NUM_MINIBATCHES value")
    if args.seed_start + args.seed_count >= 2**32:
        raise ValueError("JAX PRNGKey seeds must remain below 2**32")
    update_sizes = [args.num_steps * envs for _, _, envs, _ in combos]
    common_multiple = math.lcm(*update_sizes)
    timesteps = args.total_timesteps // common_multiple * common_multiple
    if timesteps <= 0:
        raise ValueError("Requested budget is shorter than one common update block")
    methods = tuple(method for method in METHODS if method in args.methods)
    runs: list[Run] = []
    for seed in range(args.seed_start, args.seed_start + args.seed_count):
        for lr, epochs, envs, minibatches in combos:
            base = dict(
                map_name=args.map_name, seed=seed, timesteps=timesteps,
                lr=lr, epochs=epochs, num_envs=envs,
                num_minibatches=minibatches, num_steps=args.num_steps,
            )
            for method in methods:
                if method == "none":
                    runs.append(Run(**base, method=method))
                elif method in {"mse", "cka"}:
                    coefficients = args.mse_coefs if method == "mse" else args.cka_coefs
                    runs.extend(Run(**base, method=method, coef=coef) for coef in coefficients)
                else:
                    for coef, q_steps, q_lr, ridge in itertools.product(
                        args.arec_coefs, args.arec_q_steps,
                        args.arec_q_lrs, args.arec_fisher_ridges,
                    ):
                        runs.append(Run(
                            **base, method=method, coef=coef,
                            q_steps=q_steps, q_lr=q_lr, fisher_ridge=ridge,
                        ))
    if len({run.name for run in runs}) != len(runs):
        raise ValueError("Grid produced duplicate run names")
    return runs


def hydra_float(value: float) -> str:
    return format(Decimal(str(value)), "f")


def train_command(root: Path, args: argparse.Namespace, run: Run) -> list[str]:
    align = run.method in {"mse", "cka"}
    arec = run.method == "arec"
    return [
        sys.executable, str(REPO / "baselines/MAPPO/mappo_rnn_smax.py"),
        f"MAP_NAME={run.map_name}", f"SEED={run.seed}",
        "ACTOR_PARAMETER_SHARING=false", "MATCHED_COMPARISON=true",
        f"ALIGN_MODE={'c_to_a' if align else 'none'}",
        f"ALIGN_DISTANCE={'linear_cka' if run.method == 'cka' else 'ln_mse'}",
        f"ALIGNMENT_COEF={hydra_float(run.coef) if align else '0'}",
        f"ACTOR_SCORE_RECOVERY={'true' if arec else 'false'}",
        f"ACTOR_SCORE_RECOVERY_COEF={hydra_float(run.coef) if arec else '0'}",
        f"ACTOR_SCORE_RECOVERY_Q_STEPS={run.q_steps if arec else 8}",
        f"ACTOR_SCORE_RECOVERY_Q_LR={hydra_float(run.q_lr) if arec else '0.001'}",
        f"ACTOR_SCORE_RECOVERY_FISHER_RIDGE={hydra_float(run.fisher_ridge) if arec else '0.001'}",
        f"TOTAL_TIMESTEPS={run.timesteps}", f"NUM_STEPS={run.num_steps}",
        f"UPDATE_EPOCHS={run.epochs}", f"LR={hydra_float(run.lr)}",
        f"NUM_ENVS={run.num_envs}", f"NUM_MINIBATCHES={run.num_minibatches}",
        "SAVE_CHECKPOINTS=true",
        f"CHECKPOINT_INTERVAL_TIMESTEPS={args.checkpoint_interval}",
        f"CHECKPOINT_DIR={root / 'checkpoints'}",
        "WANDB_UPLOAD_CHECKPOINTS=false", f"WANDB_MODE={args.wandb_mode}",
        f"PROJECT={args.project}", f"WANDB_RUN_ID={wandb_id(root, run)}",
        f"EXPERIMENT_CONDITION={run.condition}",
        f"MATRIX_PROFILE={PROTOCOL}", f"PROTOCOL_VERSION={PROTOCOL}",
        f"METRICS_JSONL={root / 'metrics' / (run.name + '.jsonl')}",
        f"hydra.run.dir={root / 'hydra' / run.name}",
        "hydra.job.chdir=false",
    ]


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO,
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def require_clean_checkout() -> None:
    changes = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO,
        capture_output=True, text=True, check=True,
    ).stdout
    if changes:
        raise RuntimeError("Checkout is dirty; commit/pull the exact experiment code first")


def environment_fingerprint() -> dict:
    versions = {}
    for package in ("jax", "jaxlib", "flax", "optax", "distrax", "wandb", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    gpu = (
        subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=False,
        )
        if shutil.which("nvidia-smi") else None
    )
    pip_freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True, text=True, check=False,
    )
    conda_explicit = (
        subprocess.run(
            ["conda", "list", "--explicit"],
            capture_output=True, text=True, check=False,
        )
        if os.environ.get("CONDA_PREFIX") and shutil.which("conda") else None
    )
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
        "gpu_inventory": gpu.stdout.strip().splitlines() if gpu and gpu.returncode == 0 else [],
        "pip_freeze": pip_freeze.stdout.strip().splitlines() if pip_freeze.returncode == 0 else [],
        "conda_explicit": (
            conda_explicit.stdout.strip().splitlines()
            if conda_explicit and conda_explicit.returncode == 0 else []
        ),
    }


def manifest_for(root: Path, args: argparse.Namespace, runs: list[Run]) -> dict:
    sources = (
        "baselines/MAPPO/mappo_rnn_smax.py",
        "baselines/MAPPO/score_recovery_target.py",
        "baselines/MAPPO/config/mappo_homogenous_rnn_smax.yaml",
        "baselines/MAPPO/eval_mappo_rnn_smax.py",
        "jaxmarl/environments/smax/smax_env.py",
        "scripts/smax_four_method.py",
    )
    return {
        "protocol": PROTOCOL, "schema_version": 1,
        "git_commit": git_commit(),
        "source_sha256": {name: sha256(REPO / name) for name in sources},
        "environment": environment_fingerprint(),
        "map_name": args.map_name,
        "seed_start": args.seed_start, "seed_count": args.seed_count,
        "requested_timesteps": args.total_timesteps,
        "effective_timesteps": runs[0].timesteps,
        "methods": list(args.methods),
        "checkpoint_interval": args.checkpoint_interval,
        "gpus": list(args.gpus), "max_runs_per_gpu": args.max_runs_per_gpu,
        "project": args.project, "wandb_mode": args.wandb_mode,
        "suite_id": suite_id(root),
        "runs": [dict(asdict(run), name=run.name, ident=run.ident,
                      wandb_id=wandb_id(root, run), group=group_for(root, run),
                      command=train_command(root, args, run))
                 for run in runs],
    }


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def log(path: Path, message: str) -> None:
    stamped = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(stamped, flush=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(stamped + "\n")


def checkpoint_dir(root: Path, project: str, run: Run) -> Path:
    safe_project = re.sub(r"[^A-Za-z0-9._-]+", "-", project).strip("-_") or "local"
    return root / "checkpoints" / safe_project / f"{run.name}-{wandb_id(root, run)}" / "final"


def validate_artifacts(root: Path, project: str, run: Run, commit: str) -> str | None:
    directory = checkpoint_dir(root, project, run)
    if not (directory / "model.safetensors").is_file():
        return "missing final checkpoint"
    try:
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        metrics = root / "metrics" / f"{run.name}.jsonl"
        last = json.loads(metrics.read_text(encoding="utf-8").splitlines()[-1])
    except (OSError, ValueError, IndexError) as exc:
        return f"invalid metadata or metrics: {exc}"
    if not metadata.get("is_final") or int(metadata.get("env_step", -1)) != run.timesteps:
        return "final checkpoint step mismatch"
    if metadata.get("map_name") != run.map_name or metadata.get("seed") != run.seed:
        return "checkpoint task/seed mismatch"
    if metadata.get("wandb_run_id") != wandb_id(root, run) or metadata.get("wandb_run_name") != run.name:
        return "checkpoint identity mismatch"
    if metadata.get("git_commit") != commit or config.get("PROTOCOL_VERSION") != PROTOCOL:
        return "checkpoint protocol/code mismatch"
    if config.get("EXPERIMENT_CONDITION") != run.condition or config.get("WANDB_RUN_ID") != wandb_id(root, run):
        return "checkpoint method/run-id mismatch"
    expected = {
        "TOTAL_TIMESTEPS": run.timesteps, "NUM_STEPS": run.num_steps,
        "NUM_ENVS": run.num_envs, "NUM_MINIBATCHES": run.num_minibatches,
        "UPDATE_EPOCHS": run.epochs, "LR": run.lr,
        "ALIGNMENT_COEF": run.coef if run.method in {"mse", "cka"} else 0.0,
        "ACTOR_SCORE_RECOVERY_COEF": run.coef if run.method == "arec" else 0.0,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        return "checkpoint hyperparameter grid mismatch"
    if int(last.get("env_step", -1)) != run.timesteps:
        return "training metrics are incomplete"
    return None


def frozen_manifest(root: Path) -> dict:
    path = root / "experiment_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; launcher has not initialized this run root")
    return json.loads(path.read_text(encoding="utf-8"))


def parse_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--map-name", required=True)
    parser.add_argument("--methods", type=comma_strings, default=METHODS)
    parser.add_argument("--seed-count", type=int, default=4, metavar="K")
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--total-timesteps", type=int, required=True)
    parser.add_argument("--mse-coefs", type=positive_floats, default=(0.1,))
    parser.add_argument("--cka-coefs", type=positive_floats, default=(0.3,))
    parser.add_argument("--arec-coefs", type=positive_floats, default=(3e-5,))
    parser.add_argument("--arec-q-steps", type=positive_ints, default=(4,))
    parser.add_argument("--arec-q-lrs", type=positive_floats, default=(1e-3,))
    parser.add_argument("--arec-fisher-ridges", type=positive_floats, default=(1e-3,))
    parser.add_argument("--ppo-lrs", type=positive_floats, default=(0.002,))
    parser.add_argument("--ppo-epochs", type=positive_ints, default=(4,))
    parser.add_argument("--num-envs-grid", type=positive_ints, default=(128,))
    parser.add_argument("--num-minibatches-grid", type=positive_ints, default=(4,))
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--checkpoint-interval", type=int, default=1_000_000)
    parser.add_argument("--gpus", type=comma_strings, default=("0", "1", "2", "3"))
    parser.add_argument("--max-runs-per-gpu", type=int, default=2)
    parser.add_argument("--project", default="jaxmarl-smax-four-method")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--dry-run", action="store_true")


def launch(args: argparse.Namespace) -> None:
    aliases = {"linear_cka": "cka", "ln_mse": "mse", "actor_score_recovery": "arec"}
    args.methods = tuple(aliases.get(method, method) for method in args.methods)
    if len(set(args.methods)) != len(args.methods):
        raise ValueError("Duplicate methods after alias normalization")
    if args.map_name not in map_names():
        raise ValueError(f"Unknown SMAX map: {args.map_name}")
    if not args.methods or any(method not in METHODS for method in args.methods):
        raise ValueError(f"Methods must be drawn from {METHODS}")
    if args.seed_count <= 0 or args.seed_start < 0:
        raise ValueError("Seed count must be positive and seed start nonnegative")
    if min(args.total_timesteps, args.num_steps, args.checkpoint_interval,
           args.max_runs_per_gpu) <= 0:
        raise ValueError("Budget, rollout length, interval, and concurrency must be positive")
    if not args.gpus or any(not re.fullmatch(r"\d+", gpu) for gpu in args.gpus):
        raise ValueError("--gpus must be distinct nonnegative device indices")
    if not args.project or not re.fullmatch(r"[A-Za-z0-9._-]+", args.project):
        raise ValueError("--project must be a simple W&B project name")
    root = args.run_root.expanduser().resolve()
    if root == REPO or REPO in root.parents:
        raise ValueError("Run root must be outside the Git checkout")
    runs = make_grid(args)
    if not args.dry_run:
        require_clean_checkout()
    manifest = manifest_for(root, args, runs)
    print(
        f"protocol={PROTOCOL} task={args.map_name} seeds={args.seed_count} "
        f"runs={len(runs)} requested={args.total_timesteps:,} "
        f"effective={runs[0].timesteps:,}", flush=True,
    )
    if args.dry_run:
        for run in runs:
            print(run.name, " ".join(train_command(root, args, run)), flush=True)
        return
    manifest_path = root / "experiment_manifest.json"
    if root.exists() and any(root.iterdir()) and not manifest_path.is_file():
        raise RuntimeError(
            "Run root is nonempty but has no manifest; use a fresh directory"
        )
    for subdir in ("logs", "status", "metrics", "checkpoints", "hydra", "wandb"):
        (root / subdir).mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        if frozen_manifest(root) != manifest:
            raise RuntimeError("Existing run root has a different immutable manifest")
    else:
        atomic_json(manifest_path, manifest)
    pending: list[Run] = []
    for run in runs:
        status_path = root / "status" / f"{run.name}.json"
        if status_path.exists():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("status") == "completed" and validate_artifacts(
                root, args.project, run, manifest["git_commit"]
            ) is None:
                continue
            raise RuntimeError(
                f"Run {run.name} has prior non-complete or invalid state; "
                "use a new run root rather than silently mixing attempts"
            )
        if (root / "metrics" / f"{run.name}.jsonl").exists() or checkpoint_dir(
            root, args.project, run
        ).exists():
            raise RuntimeError(f"Untracked prior output for {run.name}")
        pending.append(run)
    launcher_log = root / "launcher.log"
    log(launcher_log, f"selected={len(runs)} pending={len(pending)}")
    if not pending:
        return
    queues = {gpu: collections.deque() for gpu in args.gpus}
    for index, run in enumerate(pending):
        queues[args.gpus[index % len(args.gpus)]].append(run)
    active: dict[int, tuple[subprocess.Popen, str, Run, object]] = {}
    stopping = False

    def stop(_signal, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    failures = 0
    while any(queues.values()) or active:
        if stopping:
            for process, gpu, run, output in active.values():
                process.terminate()
                log(launcher_log, f"GPU {gpu} INTERRUPT {run.name}")
                atomic_json(root / "status" / f"{run.name}.json", {
                    "status": "interrupted", "run_name": run.name,
                    "seed": run.seed, "method": run.method, "gpu": gpu,
                })
                output.close()
            raise SystemExit(130)
        for gpu in args.gpus:
            while queues[gpu] and sum(item[1] == gpu for item in active.values()) < args.max_runs_per_gpu:
                run = queues[gpu].popleft()
                output = (root / "logs" / f"{run.name}.log").open("x", encoding="utf-8")
                environment = dict(os.environ)
                environment.pop("LD_LIBRARY_PATH", None)
                environment.update({
                    "CUDA_VISIBLE_DEVICES": gpu,
                    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                    "PYTHONHASHSEED": "0",
                    "HYDRA_FULL_ERROR": "1",
                    "WANDB_DIR": str(root / "wandb"),
                    "WANDB_NAME": run.name,
                    "WANDB_RUN_GROUP": group_for(root, run),
                    "WANDB_TAGS": f"smax,nps,{run.map_name},{run.method},seed-{run.seed},{PROTOCOL}",
                })
                process = subprocess.Popen(
                    train_command(root, args, run), cwd=REPO, env=environment,
                    stdout=output, stderr=subprocess.STDOUT,
                )
                active[process.pid] = (process, gpu, run, output)
                atomic_json(root / "status" / f"{run.name}.json", {
                    "status": "running", "pid": process.pid, "gpu": gpu,
                    "run_name": run.name, "seed": run.seed, "method": run.method,
                    "budget": run.timesteps,
                })
                log(launcher_log, f"GPU {gpu} START {run.name} pid={process.pid}")
        time.sleep(2)
        for pid, (process, gpu, run, output) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            output.close()
            issue = validate_artifacts(root, args.project, run, manifest["git_commit"]) if code == 0 else None
            failed = code != 0 or issue is not None
            failures += int(failed)
            atomic_json(root / "status" / f"{run.name}.json", {
                "status": "failed" if failed else "completed", "exit_code": code,
                "artifact_error": issue, "gpu": gpu, "run_name": run.name,
                "seed": run.seed, "method": run.method, "budget": run.timesteps,
            })
            log(launcher_log, f"GPU {gpu} END {run.name} exit={code} artifact={issue}")
            del active[pid]
    log(launcher_log, f"matrix finished; failures={failures}")
    if failures:
        raise SystemExit(1)


def show_status(root: Path) -> None:
    manifest = frozen_manifest(root)
    counts: collections.Counter[str] = collections.Counter()
    rows = []
    for entry in manifest["runs"]:
        name = entry["name"]
        status_path = root / "status" / f"{name}.json"
        state = json.loads(status_path.read_text())["status"] if status_path.exists() else "pending"
        counts[state] += 1
        metrics_path = root / "metrics" / f"{name}.jsonl"
        step = 0
        if metrics_path.is_file():
            lines = metrics_path.read_text(encoding="utf-8").splitlines()
            if lines:
                try:
                    step = int(json.loads(lines[-1]).get("env_step", 0))
                except ValueError:
                    pass
        rows.append((state, step, entry["timesteps"], name))
    print(" ".join(f"{key.upper()}={counts[key]}" for key in
                   ("completed", "running", "failed", "interrupted", "pending"))
          + f" TOTAL={len(rows)}")
    for state, step, budget, name in rows:
        print(f"{state.upper():11s} {100*step/budget:6.2f}% {step:>11,}/{budget:,} {name}")


def final_five_checkpoints(root: Path, project: str, entry: dict) -> list[Path]:
    run = Run(**{key: entry[key] for key in Run.__dataclass_fields__})
    parent = checkpoint_dir(root, project, run).parent
    candidates = []
    for directory in parent.iterdir():
        if not directory.is_dir() or not (directory / "model.safetensors").is_file():
            continue
        if directory.name != "final" and not directory.name.startswith("step_"):
            continue
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        candidates.append((int(metadata["nominal_env_step"]), directory))
    by_step = {step: directory for step, directory in sorted(candidates)}
    # A final checkpoint can share a nominal step with an interval snapshot;
    # always evaluate the explicitly final weights for that last step.
    for step, directory in candidates:
        if directory.name == "final":
            by_step[step] = directory
    if len(by_step) < 5:
        raise RuntimeError(f"Need five distinct checkpoints for {run.name}; found {len(by_step)}")
    return list(by_step.values())[-5:]


def evaluate(root: Path, episodes: int, num_envs: int, seed_base: int,
             policy: str, gpus: tuple[str, ...], max_runs_per_gpu: int) -> None:
    if min(episodes, num_envs, max_runs_per_gpu) <= 0:
        raise ValueError("Evaluation episodes, environments and concurrency must be positive")
    manifest = frozen_manifest(root)
    source = REPO / "baselines/MAPPO/eval_mappo_rnn_smax.py"
    if sha256(source) != manifest["source_sha256"]["baselines/MAPPO/eval_mappo_rnn_smax.py"]:
        raise RuntimeError("Evaluator source changed since training manifest")
    if git_commit() != manifest["git_commit"]:
        raise RuntimeError("Checkout commit differs from the frozen training manifest")
    require_clean_checkout()
    protocol = {
        "schema_version": 1, "training_git_commit": manifest["git_commit"],
        "evaluator_sha256": sha256(source), "episodes": episodes,
        "num_envs": num_envs, "seed_base": seed_base, "policy": policy,
        "gpus": list(gpus), "max_runs_per_gpu": max_runs_per_gpu,
    }
    path = root / "evaluation_manifest.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise RuntimeError("Existing evaluation manifest differs; use a new run root")
    jobs = []
    for entry in manifest["runs"]:
        run = Run(**{key: entry[key] for key in Run.__dataclass_fields__})
        status_path = root / "status" / f"{run.name}.json"
        if not status_path.is_file() or json.loads(status_path.read_text())["status"] != "completed":
            raise RuntimeError(f"Training run is not complete: {run.name}")
        issue = validate_artifacts(root, manifest["project"], run, manifest["git_commit"])
        if issue:
            raise RuntimeError(f"Invalid training artifact for {run.name}: {issue}")
        for checkpoint in final_five_checkpoints(root, manifest["project"], entry):
            output = root / "evaluation" / run.name / f"{checkpoint.name}.json"
            if output.exists():
                record = json.loads(output.read_text())
                if (record.get("checkpoint") != str(checkpoint)
                    or record.get("episodes") != episodes
                    or record.get("eval_seed") != seed_base + run.seed
                    or record.get("policy") != policy):
                    raise RuntimeError(f"Incompatible prior evaluation: {output}")
                continue
            jobs.append((run, checkpoint, output))
    if not path.exists():
        atomic_json(path, protocol)
    (root / "evaluation_logs").mkdir(exist_ok=True)
    log_path = root / "evaluation.log"
    log(log_path, f"selected={5 * len(manifest['runs'])} pending={len(jobs)}")
    queues = {gpu: collections.deque() for gpu in gpus}
    for index, job in enumerate(jobs):
        queues[gpus[index % len(gpus)]].append(job)
    active = {}
    failures = 0
    while any(queues.values()) or active:
        for gpu in gpus:
            while queues[gpu] and sum(item[1] == gpu for item in active.values()) < max_runs_per_gpu:
                run, checkpoint, output = queues[gpu].popleft()
                output.parent.mkdir(parents=True, exist_ok=True)
                logfile = (root / "evaluation_logs" / f"{run.name}-{checkpoint.name}.log").open("x")
                environment = dict(os.environ)
                environment.pop("LD_LIBRARY_PATH", None)
                environment.update({"CUDA_VISIBLE_DEVICES": gpu,
                                    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                                    "PYTHONHASHSEED": "0"})
                command = [sys.executable, str(source), "--checkpoint", str(checkpoint),
                           "--episodes", str(episodes), "--num-envs", str(num_envs),
                           "--seed", str(seed_base + run.seed), "--policy", policy,
                           "--wandb-mode", "disabled", "--output", str(output)]
                process = subprocess.Popen(command, cwd=REPO, env=environment,
                                           stdout=logfile, stderr=subprocess.STDOUT)
                active[process.pid] = (process, gpu, run, checkpoint, output, logfile)
                log(log_path, f"GPU {gpu} START {run.name}/{checkpoint.name} pid={process.pid}")
        time.sleep(2)
        for pid, (process, gpu, run, checkpoint, output, logfile) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            logfile.close()
            failures += int(code != 0 or not output.is_file())
            log(log_path, f"GPU {gpu} END {run.name}/{checkpoint.name} status={code}")
            del active[pid]
    if failures:
        raise RuntimeError(f"{failures} evaluations failed; inspect {root / 'evaluation_logs'}")


def summarize(root: Path, bootstrap: int, bootstrap_seed: int) -> None:
    import numpy as np

    manifest = frozen_manifest(root)
    if bootstrap <= 0:
        raise ValueError("--bootstrap must be positive")
    groups: dict[str, list[tuple[dict, list[dict]]]] = collections.defaultdict(list)
    for entry in manifest["runs"]:
        name = entry["name"]
        status_path = root / "status" / f"{name}.json"
        if not status_path.is_file() or json.loads(status_path.read_text())["status"] != "completed":
            raise RuntimeError(f"Cannot summarize incomplete run {name}")
        rows = [json.loads(line) for line in (root / "metrics" / f"{name}.jsonl").read_text().splitlines() if line.strip()]
        groups[entry["group"]].append((entry, rows))
    output_root = root / "summary"
    output_root.mkdir(exist_ok=True)
    output = []
    per_seed = []
    curves = []
    for group, members in sorted(groups.items()):
        members.sort(key=lambda item: item[0]["seed"])
        if len(members) != manifest["seed_count"]:
            raise RuntimeError(f"Incomplete seed cohort: {group}")
        steps = np.asarray([row["env_step"] for row in members[0][1]], dtype=np.int64)
        if len(steps) < 5 or len(np.unique(steps)) != len(steps):
            raise RuntimeError(f"Invalid metric steps: {group}")
        for entry, rows in members[1:]:
            if not np.array_equal(steps, [row["env_step"] for row in rows]):
                raise RuntimeError(f"Unmatched metric steps in {group}, seed {entry['seed']}")
        for metric in ("returns", "win_rate"):
            matrix = np.asarray([[row[metric] for row in rows] for _, rows in members], dtype=float)
            imputed = 0
            for row in matrix:
                finite = np.flatnonzero(np.isfinite(row))
                if not len(finite):
                    raise RuntimeError(f"No finite {metric} observations in {group}")
                imputed += int(len(row) - len(finite))
                row[:finite[0]] = row[finite[0]]
                for index in range(finite[0] + 1, len(row)):
                    if not np.isfinite(row[index]):
                        row[index] = row[index - 1]
            auc = np.trapz(matrix, steps, axis=1) / (steps[-1] - steps[0])
            checkpoint_steps = list(range(
                manifest["checkpoint_interval"], int(steps[-1]),
                manifest["checkpoint_interval"],
            )) + [int(steps[-1])]
            if len(checkpoint_steps) < 5:
                raise RuntimeError(f"Need at least five checkpoints for final-five: {group}")
            selected = np.searchsorted(steps, checkpoint_steps[-5:], side="right") - 1
            final5 = matrix[:, selected].mean(axis=1)
            final_source = "training"
            if (root / "evaluation_manifest.json").exists():
                eval_key = "return_mean" if metric == "returns" else "win_rate"
                eval_values = []
                for entry, _ in members:
                    paths = final_five_checkpoints(root, manifest["project"], entry)
                    values = []
                    for checkpoint in paths:
                        path = root / "evaluation" / entry["name"] / f"{checkpoint.name}.json"
                        if not path.is_file():
                            raise RuntimeError(f"Missing held-out evaluation {path}")
                        values.append(float(json.loads(path.read_text())[eval_key]))
                    eval_values.append(sum(values) / 5)
                final5 = np.asarray(eval_values)
                final_source = "heldout_eval"
            rng = np.random.default_rng(bootstrap_seed)
            picks = rng.integers(0, len(members), size=(bootstrap, len(members)))
            auc_ci = np.quantile(auc[picks].mean(axis=1), [0.025, 0.975])
            final_ci = np.quantile(final5[picks].mean(axis=1), [0.025, 0.975])
            curve_bootstrap = matrix[picks].mean(axis=1)
            curve_ci = np.quantile(curve_bootstrap, [0.025, 0.975], axis=0)
            for index, step in enumerate(steps):
                curves.append({
                    "group": group, "method": members[0][0]["method"],
                    "metric": metric, "env_step": int(step),
                    "mean": float(matrix[:, index].mean()),
                    "ci_low": float(curve_ci[0, index]),
                    "ci_high": float(curve_ci[1, index]),
                })
            for index, (entry, _) in enumerate(members):
                per_seed.append({
                    "group": group, "method": entry["method"],
                    "metric": metric, "seed": entry["seed"],
                    "normalized_auc": float(auc[index]),
                    "final_five": float(final5[index]),
                    "final_source": final_source,
                })
            item = dict(
                map_name=manifest["map_name"], group=group, method=members[0][0]["method"],
                metric=metric, seeds=manifest["seed_count"],
                imputed_metric_rows=imputed,
                final_source=final_source,
                coef=members[0][0]["coef"], lr=members[0][0]["lr"],
                epochs=members[0][0]["epochs"], num_envs=members[0][0]["num_envs"],
                num_minibatches=members[0][0]["num_minibatches"],
                q_steps=members[0][0]["q_steps"], q_lr=members[0][0]["q_lr"],
                fisher_ridge=members[0][0]["fisher_ridge"],
                normalized_auc=float(auc.mean()),
                auc_ci_low=float(auc_ci[0]), auc_ci_high=float(auc_ci[1]),
                final_five=float(final5.mean()),
                final_ci_low=float(final_ci[0]), final_ci_high=float(final_ci[1]),
            )
            output.append(item)
    for name, records in (("summary", output), ("per_seed", per_seed), ("curves", curves)):
        with (output_root / f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    print(output_root / "summary.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    run_parser = sub.add_parser("run", help="Launch or verify an immutable sweep")
    parse_run_args(run_parser)
    for name in ("status", "evaluate", "analyze"):
        part = sub.add_parser(name)
        part.add_argument("--run-root", type=Path, required=True)
        if name == "evaluate":
            part.add_argument("--episodes", type=int, default=256)
            part.add_argument("--num-envs", type=int, default=64)
            part.add_argument("--eval-seed-base", type=int, default=10000)
            part.add_argument("--policy", choices=("deterministic", "stochastic"), default="deterministic")
            part.add_argument("--gpus", type=comma_strings, default=("0", "1", "2", "3"))
            part.add_argument("--max-runs-per-gpu", type=int, default=2)
        if name == "analyze":
            part.add_argument("--bootstrap", type=int, default=5000)
            part.add_argument("--bootstrap-seed", type=int, default=20260924)
    args = parser.parse_args()
    if args.action == "run":
        launch(args)
    elif args.action == "status":
        show_status(args.run_root.expanduser().resolve())
    elif args.action == "evaluate":
        evaluate(args.run_root.expanduser().resolve(), args.episodes, args.num_envs,
                 args.eval_seed_base, args.policy, args.gpus, args.max_runs_per_gpu)
    else:
        summarize(args.run_root.expanduser().resolve(), args.bootstrap, args.bootstrap_seed)


if __name__ == "__main__":
    main()
