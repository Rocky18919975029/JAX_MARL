#!/usr/bin/env python3
"""Show all SMAX score-recoverability collection/probe jobs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.expanduser().resolve()
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    rows = []
    for task in protocol["tasks"]:
        for condition in protocol["conditions"]:
            for seed in protocol["seeds"]:
                name = f"{task}--{condition}--seed{seed}"
                output = root / "runs" / task / condition / f"seed_{seed}"
                status_path = root / "status" / f"{name}.json"
                progress_path = output / "progress.json"
                if (output / "summary.json").is_file():
                    status, stage = "COMPLETED", "completed"
                elif status_path.is_file():
                    payload = json.loads(status_path.read_text(encoding="utf-8"))
                    status = str(payload.get("status", "unknown")).upper()
                    stage = "starting"
                    if progress_path.is_file():
                        stage = json.loads(
                            progress_path.read_text(encoding="utf-8")
                        ).get("stage", stage)
                else:
                    status, stage = "PENDING", "waiting"
                rows.append((status, stage, task, condition, seed))
    counts = {
        state: sum(row[0] == state for row in rows)
        for state in ("COMPLETED", "RUNNING", "FAILED", "PENDING")
    }
    print(
        " ".join(f"{key}={value}" for key, value in counts.items())
        + f" TOTAL={len(rows)}"
    )
    for status, stage, task, condition, seed in rows:
        print(f"{status:9s} {stage:16s} {task:20s} " f"{condition:12s} seed={seed}")
    launcher = root / "launcher.log"
    if launcher.is_file():
        print("\nLatest launcher events:")
        print("\n".join(launcher.read_text(encoding="utf-8").splitlines()[-12:]))


if __name__ == "__main__":
    main()
