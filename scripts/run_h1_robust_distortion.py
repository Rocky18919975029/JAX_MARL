#!/usr/bin/env python3
"""Run resumable robust-distortion recomputation over collected checkpoints."""

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
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MARKER = "robust_distortion_summary.json"


def parse_csv(value):
    return tuple(item.strip() for item in value.split(",") if item.strip())


def log(path, message):
    timestamp = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(line + "\n")


def discover(source_roots, run_name_glob, maps, conditions):
    selected = {}
    for source_root in source_roots:
        diagnostics_root = source_root / "diagnostics_raw"
        for metadata_path in sorted(
            diagnostics_root.glob(f"{run_name_glob}/*/metadata.json")
        ):
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("actor_parameter_sharing"):
                continue
            if maps and metadata.get("map_name") not in maps:
                continue
            condition = str(metadata.get("condition", ""))
            if conditions and condition not in conditions:
                continue
            key = (metadata["run_name"], metadata_path.parent.name)
            prior = selected.get(key)
            if prior is not None and prior != metadata_path.parent:
                raise RuntimeError(
                    f"Duplicate collected checkpoint {key}: {prior} and "
                    f"{metadata_path.parent}"
                )
            selected[key] = metadata_path.parent
    return sorted(selected.items())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-name-glob", default="*")
    parser.add_argument("--maps", type=parse_csv)
    parser.add_argument("--conditions", type=parse_csv)
    parser.add_argument("--fisher-ridges", default="0.0001,0.0003,0.001,0.003,0.01")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    source_roots = tuple(path.expanduser().resolve() for path in args.source_root)
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    tasks = discover(
        source_roots,
        args.run_name_glob,
        set(args.maps or ()),
        set(args.conditions or ()),
    )
    pending = deque()
    completed = 0
    for (run_name, checkpoint_name), diagnostics_dir in tasks:
        output_dir = output_root / "checkpoints" / run_name / checkpoint_name
        if not args.rerun and (output_dir / MARKER).is_file():
            completed += 1
        else:
            pending.append((run_name, checkpoint_name, diagnostics_dir, output_dir))
    if args.dry_run:
        print(
            f"discovered={len(tasks)} completed={completed} pending={len(pending)} "
            f"workers={args.workers}"
        )
        for run_name, checkpoint_name, diagnostics_dir, output_dir in pending:
            print(f"{run_name}/{checkpoint_name}: {diagnostics_dir} -> {output_dir}")
        return

    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    launcher_log = output_root / "launcher.log"
    manifest_path = output_root / "run_manifest.json"
    manifest = {
        "schema_version": 1,
        "source_roots": [str(path) for path in source_roots],
        "output_root": str(output_root),
        "run_name_glob": args.run_name_glob,
        "maps": args.maps,
        "conditions": args.conditions,
        "fisher_ridges": args.fisher_ridges,
        "workers": args.workers,
        "discovered": len(tasks),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    log(
        launcher_log,
        f"discovered={len(tasks)} completed={completed} pending={len(pending)} "
        f"workers={args.workers}",
    )
    environment = dict(os.environ)
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        environment[variable] = "1"
    running = {}
    stop_requested = False

    def request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    failures = 0
    while pending or running:
        if stop_requested:
            for process, handle, *_ in running.values():
                process.terminate()
                handle.close()
            raise SystemExit(130)
        while pending and len(running) < args.workers:
            run_name, checkpoint_name, diagnostics_dir, output_dir = pending.popleft()
            output_dir.mkdir(parents=True, exist_ok=True)
            worker_log = log_dir / f"{run_name}-{checkpoint_name}.log"
            handle = worker_log.open("w", encoding="utf-8")
            command = [
                sys.executable,
                str(REPO_ROOT / "scripts/h1_robust_distortion.py"),
                "--diagnostics-dir",
                str(diagnostics_dir),
                "--output-dir",
                str(output_dir),
                "--fisher-ridges",
                args.fisher_ridges,
            ]
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            running[process.pid] = (
                process,
                handle,
                run_name,
                checkpoint_name,
                worker_log,
            )
            log(
                launcher_log,
                f"START {run_name}/{checkpoint_name} pid={process.pid}",
            )
        finished = [pid for pid, item in running.items() if item[0].poll() is not None]
        for pid in finished:
            process, handle, run_name, checkpoint_name, worker_log = running.pop(pid)
            handle.close()
            if process.returncode:
                failures += 1
            log(
                launcher_log,
                f"END   {run_name}/{checkpoint_name} status={process.returncode} "
                f"log={worker_log}",
            )
        if not finished:
            time.sleep(0.5)
    log(launcher_log, f"robust distortion finished; failures={failures}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
