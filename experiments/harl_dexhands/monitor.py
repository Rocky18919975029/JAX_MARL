#!/usr/bin/env python3
"""Display all ShadowHandOver HAPPO/MAPPO/MADPO run progress."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from experiments.harl_dexhands.protocol import (  # noqa: E402
    ALGORITHMS,
    CONDITIONS,
    parse_csv,
    parse_positive_floats,
    parse_seeds,
    task_matrix,
)


def bar(progress: float, width: int = 28) -> str:
    filled = min(width, max(0, round(width * progress)))
    return "█" * filled + "░" * (width - filled)


def load_rows(
    root: Path,
    algorithms: tuple[str, ...],
    seeds: tuple[int, ...],
    *,
    div_coef: float = 1000.0,
    div_weight: float = 0.05,
    div_sigma: float = 1.0,
    div_max_samples: int = 1024,
    conditions: tuple[str, ...] = ("none",),
    arec_coef: float = 0.0001,
    arec_coefs: tuple[float, ...] | None = None,
    arec_q_steps: int = 4,
    arec_q_lr: float = 0.001,
    arec_fisher_ridge: float = 0.001,
) -> list[dict]:
    rows = []
    for task in task_matrix(
        algorithms,
        seeds,
        div_coef=div_coef,
        div_weight=div_weight,
        div_sigma=div_sigma,
        div_max_samples=div_max_samples,
        conditions=conditions,
        arec_coef=arec_coef,
        arec_coefs=arec_coefs,
        arec_q_steps=arec_q_steps,
        arec_q_lr=arec_q_lr,
        arec_fisher_ridge=arec_fisher_ridge,
    ):
        path = root / "status" / f"{task.name}.json"
        if not path.is_file():
            rows.append(
                {"name": task.name, "status": "pending", "steps": 0, "total": 0}
            )
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = {}
        state = str(payload.get("status", "unknown"))
        total = int(payload.get("total_env_steps", 0))
        steps = int(payload.get("env_steps", 0))
        if state == "completed":
            steps = total
        rows.append(
            {"name": task.name, "status": state, "steps": steps, "total": total}
        )
    return rows


def load_manifest_rows(root: Path) -> list[dict]:
    """Use the launcher's frozen grid so a simple --run-root shows every run."""
    manifest = json.loads((root / "experiment_manifest.json").read_text())
    budget = int(manifest["study_spec"]["num_env_steps"])
    rows = []
    for run in manifest["runs"]:
        name = run["run_name"]
        path = root / "status" / f"{name}.json"
        if not path.is_file():
            rows.append(
                {"name": name, "status": "pending", "steps": 0, "total": budget}
            )
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = {}
        state = str(payload.get("status", "unknown"))
        total = int(payload.get("total_env_steps", budget))
        steps = total if state == "completed" else int(payload.get("env_steps", 0))
        rows.append({"name": name, "status": state, "steps": steps, "total": total})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--algorithms", default="happo,mappo,madpo")
    parser.add_argument("--conditions", default="none")
    parser.add_argument("--seeds", default="1-4")
    parser.add_argument("--div-coef", type=float, default=1000.0)
    parser.add_argument("--div-weight", type=float, default=0.05)
    parser.add_argument("--div-sigma", type=float, default=1.0)
    parser.add_argument("--div-max-samples", type=int, default=1024)
    coefficient_group = parser.add_mutually_exclusive_group()
    coefficient_group.add_argument("--arec-coef", type=float, default=0.0001)
    coefficient_group.add_argument("--arec-coefs")
    parser.add_argument("--arec-q-steps", type=int, default=4)
    parser.add_argument("--arec-q-lr", type=float, default=0.001)
    parser.add_argument("--arec-fisher-ridge", type=float, default=0.001)
    parser.add_argument("--ignore-manifest", action="store_true")
    args = parser.parse_args()
    root = args.run_root.expanduser().resolve()
    if (root / "experiment_manifest.json").is_file() and not args.ignore_manifest:
        rows = load_manifest_rows(root)
    else:
        algorithms = parse_csv(args.algorithms, ALGORITHMS)
        conditions = parse_csv(args.conditions, CONDITIONS)
        seeds = parse_seeds(args.seeds)
        arec_coefs = (
            (args.arec_coef,)
            if args.arec_coefs is None
            else parse_positive_floats(args.arec_coefs)
        )
        rows = load_rows(
            root,
            algorithms,
            seeds,
            div_coef=args.div_coef,
            div_weight=args.div_weight,
            div_sigma=args.div_sigma,
            div_max_samples=args.div_max_samples,
            conditions=conditions,
            arec_coef=args.arec_coef,
            arec_coefs=arec_coefs,
            arec_q_steps=args.arec_q_steps,
            arec_q_lr=args.arec_q_lr,
            arec_fisher_ridge=args.arec_fisher_ridge,
        )
    counts = {
        state: sum(row["status"] == state for row in rows)
        for state in ("completed", "running", "failed", "pending")
    }
    print(
        f"DONE={counts['completed']} RUNNING={counts['running']} "
        f"FAILED={counts['failed']} PENDING={counts['pending']} TOTAL={len(rows)}\n"
    )
    for row in rows:
        progress = row["steps"] / row["total"] if row["total"] else 0.0
        print(
            f"{row['status'].upper():9s} [{bar(progress)}] {progress:7.2%} "
            f"{row['steps']:>11,}/{row['total']:,}  {row['name']}"
        )
    launcher = root / "launcher.log"
    if launcher.is_file():
        print("\nLatest launcher events:")
        print("\n".join(launcher.read_text(encoding="utf-8").splitlines()[-12:]))


if __name__ == "__main__":
    main()
