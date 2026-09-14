#!/usr/bin/env python3
"""Evaluate preregistered or all H1 checkpoints with paired deterministic seeds."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import signal
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_SPECS = (
    ("initial", 0),
    ("step_000000500000", 500_000),
    ("step_000001000000", 1_000_000),
    ("step_000002000000", 2_000_000),
    ("step_000004000000", 4_000_000),
    ("step_000006000000", 6_000_000),
    ("step_000008000000", 8_000_000),
    ("final", None),
)


def checkpoint_specs(run_dir, include_all=False):
    """Return the preregistered checkpoints plus any saved dense checkpoints.

    Preregistered entries keep their original indices so existing evaluation
    JSON remains reusable. Additional checkpoints receive stable indices after
    that fixed set, ordered by nominal environment step.
    """

    if not include_all:
        return CHECKPOINT_SPECS
    known_names = {name for name, _ in CHECKPOINT_SPECS}
    additional = []
    for checkpoint_dir in run_dir.glob("step_*"):
        if checkpoint_dir.name in known_names:
            continue
        try:
            nominal_step = int(checkpoint_dir.name.removeprefix("step_"))
        except ValueError:
            continue
        if (checkpoint_dir / "model.safetensors").is_file():
            additional.append((checkpoint_dir.name, nominal_step))
    additional.sort(key=lambda item: item[1])
    return CHECKPOINT_SPECS + tuple(additional)


@dataclass(frozen=True)
class EvalTask:
    run_dir: Path
    checkpoint_dir: Path
    output_path: Path
    run_name: str
    training_seed: int
    checkpoint_index: int
    nominal_step: int | None

    @property
    def eval_seed(self):
        return 100_000 + 100 * self.training_seed + self.checkpoint_index


def append_jsonl(path, payload):
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, sort_keys=True) + "\n")
        file.flush()


def discover_tasks(run_root, include_all=False):
    checkpoint_root = run_root / "checkpoints"
    run_dirs = sorted(
        path.parent.parent
        for path in checkpoint_root.rglob("initial/model.safetensors")
    )
    tasks = []
    missing = []
    for run_dir in run_dirs:
        initial_config = run_dir / "initial" / "config.json"
        if not initial_config.is_file():
            missing.append(f"{run_dir}: initial/config.json")
            continue
        config = json.loads(initial_config.read_text(encoding="utf-8"))
        if config.get("PROTOCOL_VERSION") != "h1-v1.0":
            continue
        run_name = config.get("WANDB_NAME") or run_dir.name.rsplit("-", 1)[0]
        # WANDB_NAME lives in the environment rather than Hydra config, so the
        # callback metadata is the authoritative source when available.
        metadata_path = run_dir / "initial" / "metadata.json"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            run_name = metadata.get("wandb_run_name") or run_name
        training_seed = int(config["SEED"])
        output_dir = run_root / "evaluation" / run_name
        for checkpoint_index, (directory_name, nominal_step) in enumerate(
            checkpoint_specs(run_dir, include_all)
        ):
            checkpoint_dir = run_dir / directory_name
            if not (checkpoint_dir / "model.safetensors").is_file():
                missing.append(f"{run_name}: {directory_name}")
                continue
            output_path = output_dir / f"{directory_name}.json"
            tasks.append(
                EvalTask(
                    run_dir,
                    checkpoint_dir,
                    output_path,
                    run_name,
                    training_seed,
                    checkpoint_index,
                    nominal_step,
                )
            )
    return tasks, missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--max-runs-per-gpu", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument(
        "--all-checkpoints",
        action="store_true",
        help="Evaluate every saved step_* checkpoint, not only preregistered points",
    )
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    run_root = args.run_root.expanduser().resolve()
    gpu_ids = tuple(item.strip() for item in args.gpus.split(",") if item.strip())
    if not gpu_ids or args.max_runs_per_gpu <= 0:
        raise ValueError("Select GPUs and a positive --max-runs-per-gpu")
    tasks, missing = discover_tasks(run_root, include_all=args.all_checkpoints)
    if missing:
        report = run_root / "evaluation" / "missing_checkpoints.txt"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("\n".join(missing) + "\n", encoding="utf-8")
        if not args.allow_missing:
            raise RuntimeError(
                f"{len(missing)} preregistered checkpoints are missing; see {report}"
            )

    pending = {gpu: deque() for gpu in gpu_ids}
    selected = [task for task in tasks if args.rerun or not task.output_path.is_file()]
    for index, task in enumerate(selected):
        pending[gpu_ids[index % len(gpu_ids)]].append(task)
    eval_root = run_root / "evaluation"
    eval_root.mkdir(parents=True, exist_ok=True)
    manifest_path = eval_root / "evaluation_manifest.jsonl"
    launcher_log = eval_root / "launcher.log"

    def log(message):
        timestamp = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        print(line, flush=True)
        with launcher_log.open("a", encoding="utf-8") as file:
            file.write(line + "\n")

    log(
        f"mode={'all' if args.all_checkpoints else 'preregistered'} "
        f"discovered={len(tasks)} pending={len(selected)} missing={len(missing)} "
        f"episodes={args.episodes}"
    )
    running = {}
    stop = False

    def stop_handler(_signum, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)
    failures = 0
    while any(pending.values()) or running:
        if stop:
            for process, _, _, handle, _ in running.values():
                process.terminate()
                handle.close()
            raise SystemExit(130)
        for gpu in gpu_ids:
            active = sum(item[1] == gpu for item in running.values())
            while pending[gpu] and active < args.max_runs_per_gpu:
                task = pending[gpu].popleft()
                task.output_path.parent.mkdir(parents=True, exist_ok=True)
                log_path = task.output_path.with_suffix(".log")
                handle = log_path.open("w", encoding="utf-8")
                command = [
                    sys.executable,
                    str(REPO_ROOT / "baselines/MAPPO/eval_mappo_rnn_smax.py"),
                    "--checkpoint",
                    str(task.checkpoint_dir),
                    "--episodes",
                    str(args.episodes),
                    "--num-envs",
                    str(args.num_envs),
                    "--seed",
                    str(task.eval_seed),
                    "--policy",
                    "deterministic",
                    "--output",
                    str(task.output_path),
                ]
                environment = os.environ.copy()
                environment.pop("LD_LIBRARY_PATH", None)
                environment.update(
                    {
                        "CUDA_VISIBLE_DEVICES": gpu,
                        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                    }
                )
                started = dt.datetime.now(dt.timezone.utc).isoformat()
                process = subprocess.Popen(
                    command,
                    cwd=REPO_ROOT,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
                running[process.pid] = (process, gpu, task, handle, started)
                log(
                    f"GPU {gpu} START {task.run_name}/{task.checkpoint_dir.name} "
                    f"eval_seed={task.eval_seed} pid={process.pid}"
                )
                active += 1

        finished = [pid for pid, item in running.items() if item[0].poll() is not None]
        for pid in finished:
            process, gpu, task, handle, started = running.pop(pid)
            handle.close()
            status = "completed" if process.returncode == 0 else "failed"
            record = {
                "schema_version": 1,
                "run_name": task.run_name,
                "checkpoint": str(task.checkpoint_dir),
                "checkpoint_index": task.checkpoint_index,
                "nominal_step": task.nominal_step,
                "training_seed": task.training_seed,
                "eval_seed": task.eval_seed,
                "episodes": args.episodes,
                "gpu": gpu,
                "started_at_utc": started,
                "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "status": status,
                "exit_code": process.returncode,
                "output": str(task.output_path),
            }
            append_jsonl(manifest_path, record)
            failures += process.returncode != 0
            log(
                f"GPU {gpu} END   {task.run_name}/{task.checkpoint_dir.name} "
                f"status={process.returncode}"
            )
        if not finished:
            time.sleep(1)
    log(f"evaluation finished; failures={failures}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
