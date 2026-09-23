#!/usr/bin/env python3
"""Show all isolated and actor-score-recovery sweep progress bars."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def latest_step(path: Path) -> int:
    if not path.is_file():
        return 0
    step = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                step = max(step, int(json.loads(line).get("env_step", 0)))
            except (ValueError, TypeError):
                continue
    return step


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.expanduser().resolve()
    manifest = json.loads((root / "experiment_manifest.json").read_text())
    rows = []
    for run in manifest["runs"]:
        name = run["run_name"]
        status_path = root / "status" / f"{name}.json"
        status = (
            json.loads(status_path.read_text()).get("status", "unknown")
            if status_path.is_file()
            else "pending"
        )
        rows.append(
            (
                status,
                latest_step(root / "metrics" / f"{name}.jsonl"),
                int(run["steps"]),
                name,
            )
        )
    counts = {
        status: sum(row[0] == status for row in rows)
        for status in ("completed", "running", "failed", "pending")
    }
    print(
        " ".join(f"{status.upper()}={count}" for status, count in counts.items())
        + f" TOTAL={len(rows)}"
    )
    for status, step, budget, name in rows:
        progress = min(1.0, step / budget)
        filled = round(28 * progress)
        bar = "█" * filled + "░" * (28 - filled)
        print(
            f"{status.upper():9s} [{bar}] {100*progress:6.2f}% "
            f"{step:>11,}/{budget:,}  {name}"
        )
    launcher = root / "launcher.log"
    if launcher.is_file():
        print("\nLatest launcher events:")
        print("\n".join(launcher.read_text(encoding="utf-8").splitlines()[-12:]))


if __name__ == "__main__":
    main()
