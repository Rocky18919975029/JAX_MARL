#!/usr/bin/env python3
"""Index the first four-panel's selected runs and their six-seed extension.

Historical sweep roots also contain unselected hyperparameter cells. This
script creates a per-run symlink view, leaving the original sweep/report paths
untouched. New seeds are trained in each task's ``extension_6seed`` directory.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

try:
    from scripts import smax_four_method as control
    from scripts.run_smax_first_four_panel_tuned import (
        FIGURE_ID, TASKS, _selected_historical_runs, launch_args, load_pair,
        verify_historical,
    )
except ModuleNotFoundError:  # Direct execution from scripts/.
    import smax_four_method as control
    from run_smax_first_four_panel_tuned import (
        FIGURE_ID, TASKS, _selected_historical_runs, launch_args, load_pair,
        verify_historical,
    )


EXTENSION_SEEDS = tuple(range(5, 11))
METHODS = ("none", "arec")


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _historical_outputs(source: Path, run: dict) -> dict[str, Path]:
    name = run["run_name"]
    outputs = {
        "source_manifest.json": source / "experiment_manifest.json",
        "status.json": source / "status" / f"{name}.json",
        "metrics.jsonl": source / "metrics" / f"{name}.jsonl",
    }
    matches = list((source / "checkpoints").glob(f"**/{name}-*/final/config.json"))
    if len(matches) != 1 or not (matches[0].parent / "model.safetensors").is_file():
        raise RuntimeError(f"Expected one complete final checkpoint for {name}")
    outputs["checkpoints"] = matches[0].parent.parent
    train_log = source / "logs" / f"{name}.log"
    if train_log.is_file():
        outputs["train.log"] = train_log
    for label, path in outputs.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing historical {label}: {path}")
    if _read(outputs["status.json"]).get("status") != "completed":
        raise RuntimeError(f"Historical run is not completed: {name}")
    return outputs


def historical_plan(matrix_root: Path) -> tuple[list[dict], dict]:
    entries: list[dict] = []
    provenance: dict = {}
    for task in TASKS:
        none, arec, _ = load_pair(task)
        verified = verify_historical(task, (none, arec), matrix_root)
        source = Path(verified["historical_source_root"])
        manifest = _read(source / "experiment_manifest.json")
        provenance[task] = verified
        for profile in (none, arec):
            method = profile["method"]
            selected = _selected_historical_runs(manifest, method, profile["config"])
            for seed in range(1, 5):
                run = selected[seed]
                outputs = _historical_outputs(source, run)
                checkpoint_config = _read(outputs["checkpoints"] / "final" / "config.json")
                entries.append({
                    "task": task, "method": method, "seed": seed,
                    "origin": "first_four_panel_v1", "run_name": run["run_name"],
                    "source_root": str(source),
                    "training_git_commit": checkpoint_config.get("GIT_COMMIT", "unknown"),
                    "outputs": {key: str(path) for key, path in outputs.items()},
                })
    return entries, provenance


def extension_root(collection_root: Path, task: str) -> Path:
    return collection_root / task / "extension_6seed"


def extension_plan(collection_root: Path) -> tuple[list[dict], dict[str, int]]:
    entries: list[dict] = []
    counts: dict[str, int] = {}
    for task in TASKS:
        root = extension_root(collection_root, task)
        manifest_path = root / "experiment_manifest.json"
        if not manifest_path.is_file():
            counts[task] = 0
            continue
        manifest = _read(manifest_path)
        if (manifest.get("protocol") != control.PROTOCOL
                or manifest.get("map_name") != task
                or manifest.get("seed_start") != 5
                or manifest.get("seed_count") != 6
                or manifest.get("methods") != list(METHODS)
                or manifest.get("tuned_selection", {}).get("figure_id") != FIGURE_ID):
            raise RuntimeError(f"Unexpected six-seed extension manifest: {manifest_path}")
        none, arec, paths = load_pair(task)
        expected_args = SimpleNamespace(
            task=task, seed_start=5, seed_count=6, run_root=root,
            gpus=tuple(manifest["gpus"]),
            max_runs_per_gpu=manifest["max_runs_per_gpu"],
            project=manifest["project"], wandb_mode=manifest["wandb_mode"],
            dry_run=False,
        )
        expected_launch = launch_args(expected_args, none, arec, paths, None)
        expected = {run.name: asdict(run) for run in control.make_grid(expected_launch)}
        observed = {
            row["name"]: {key: row[key] for key in control.Run.__dataclass_fields__}
            for row in manifest["runs"]
        }
        if observed != expected or len(manifest["runs"]) != 12:
            raise RuntimeError(f"Extension grid differs from frozen YAML: {manifest_path}")
        completed = 0
        for row in manifest["runs"]:
            run = control.Run(**observed[row["name"]])
            status_path = root / "status" / f"{run.name}.json"
            if not status_path.is_file() or _read(status_path).get("status") != "completed":
                continue
            issue = control.validate_artifacts(
                root, manifest["project"], run, manifest["git_commit"]
            )
            if issue is not None:
                raise RuntimeError(f"Invalid completed extension {run.name}: {issue}")
            completed += 1
            outputs = {
                "source_manifest.json": manifest_path,
                "status.json": status_path,
                "metrics.jsonl": root / "metrics" / f"{run.name}.jsonl",
                "checkpoints": control.checkpoint_dir(root, manifest["project"], run).parent,
                "train.log": root / "logs" / f"{run.name}.log",
            }
            if any(not path.exists() for path in outputs.values()):
                raise RuntimeError(f"Missing completed extension outputs: {run.name}")
            entries.append({
                "task": task, "method": run.method, "seed": run.seed,
                "origin": "six_seed_extension", "run_name": run.name,
                "source_root": str(root),
                "training_git_commit": manifest["git_commit"],
                "outputs": {key: str(path) for key, path in outputs.items()},
            })
        counts[task] = completed
    return entries, counts


def _link(path: Path, target: Path) -> None:
    if path.is_symlink():
        if path.resolve() == target.resolve():
            return
        raise RuntimeError(f"Conflicting existing symlink: {path}")
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite existing path: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(target, path, target_is_directory=target.is_dir())


def apply_plan(collection_root: Path, entries: list[dict], provenance: dict) -> int:
    index_path = collection_root / "collection_index.json"
    prior_entries: dict[tuple[str, str, int], dict] = {}
    if index_path.exists():
        previous = _read(index_path)
        if previous.get("figure_id") != FIGURE_ID or previous.get("historical_provenance") != provenance:
            raise RuntimeError(f"Existing collection identity differs: {index_path}")
        prior_entries = {
            (row["task"], row["method"], row["seed"]): row
            for row in previous["runs"]
        }
        for row in entries:
            key = row["task"], row["method"], row["seed"]
            if key in prior_entries and prior_entries[key] != row:
                raise RuntimeError(f"Existing collection source differs: {key}")
    for row in entries:
        run_view = (collection_root / row["task"] / "runs" / row["method"]
                    / f"seed_{row['seed']:02d}")
        for label, target in row["outputs"].items():
            _link(run_view / label, Path(target))
        prior_entries[row["task"], row["method"], row["seed"]] = row
    all_entries = sorted(
        prior_entries.values(),
        key=lambda row: (TASKS.index(row["task"]), METHODS.index(row["method"]), row["seed"]),
    )
    control.atomic_json(index_path, {
        "schema_version": 1, "figure_id": FIGURE_ID,
        "historical_provenance": provenance,
        "extension_seeds": list(EXTENSION_SEEDS),
        "storage_rule": "Original training artifacts are symlinked, not moved or copied",
        "runs": all_entries,
    })
    return len(all_entries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("prepare", "refresh"), default="prepare")
    parser.add_argument("--apply", action="store_true",
                        help="Create links and index; without it, print a read-only plan")
    parser.add_argument("--require-complete", action="store_true",
                        help="For refresh, require all twelve extension runs per task")
    args = parser.parse_args()
    matrix_root = args.matrix_root.expanduser().resolve()
    collection_root = args.collection_root.expanduser().resolve()
    if (matrix_root not in collection_root.parents
            or collection_root == control.REPO
            or control.REPO in collection_root.parents):
        parser.error("Collection root must be a dedicated child of --matrix-root outside the checkout")
    historical, provenance = historical_plan(matrix_root)
    for source in (Path(info["historical_source_root"]) for info in provenance.values()):
        if collection_root == source or source in collection_root.parents:
            parser.error(f"Collection root cannot be inside a historical sweep: {source}")
    entries = historical
    if args.phase == "refresh":
        extension, counts = extension_plan(collection_root)
        entries = historical + extension
        for task in TASKS:
            print(f"{task}: original=8 extension={counts[task]}/12", flush=True)
        if args.require_complete and any(counts[task] != 12 for task in TASKS):
            raise RuntimeError("The six-seed extension has not completed on all four tasks")
    else:
        for task in TASKS:
            print(f"{task}: original=8 extension_run_root={extension_root(collection_root, task)}",
                  flush=True)
    if args.apply:
        linked = apply_plan(collection_root, entries, provenance)
        print(f"Indexed {linked} selected runs: {collection_root}", flush=True)
    else:
        print(f"Read-only plan: {len(entries)} selected runs; add --apply to link", flush=True)


if __name__ == "__main__":
    main()
