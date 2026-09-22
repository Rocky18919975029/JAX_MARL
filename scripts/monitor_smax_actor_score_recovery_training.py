#!/usr/bin/env python3
"""Display all actor-side score-recovery SMAX training runs."""

import argparse
import json
from pathlib import Path

try:
    from scripts.run_smax_actor_score_recovery_training import Run
except ModuleNotFoundError:  # Direct execution from scripts/.
    from run_smax_actor_score_recovery_training import Run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.expanduser().resolve()
    manifest = json.loads((root / "experiment_manifest.json").read_text())
    rows = []
    for task in manifest["maps"]:
        for seed in manifest["seeds"]:
            run = Run(
                task,
                seed,
                manifest["budgets"][task],
                manifest["actor_score_recovery_coef"],
            )
            status_path = root / "status" / f"{run.name}.json"
            status = (
                json.loads(status_path.read_text()).get("status", "unknown")
                if status_path.is_file()
                else "pending"
            )
            step = 0
            metrics_path = root / "metrics" / f"{run.name}.jsonl"
            if metrics_path.is_file():
                with metrics_path.open(encoding="utf-8") as file:
                    for line in file:
                        if line.strip():
                            try:
                                step = int(json.loads(line).get("env_step", step))
                            except json.JSONDecodeError:
                                continue
            rows.append((status, step, run))
    counts = {
        state: sum(status == state for status, _, _ in rows)
        for state in ("completed", "running", "failed", "pending")
    }
    print(
        " ".join(f"{state.upper()}={count}" for state, count in counts.items())
        + f" TOTAL={len(rows)}"
    )
    for status, step, run in rows:
        progress = min(1.0, step / max(run.steps, 1))
        width = 28
        fill = round(width * progress)
        bar = "█" * fill + "░" * (width - fill)
        print(
            f"{status.upper():9s} [{bar}] {100*progress:6.2f}% "
            f"{step:>11,}/{run.steps:,}  {run.name}"
        )
    launcher = root / "launcher.log"
    if launcher.is_file():
        print("\nLatest launcher events:")
        print("\n".join(launcher.read_text(encoding="utf-8").splitlines()[-12:]))


if __name__ == "__main__":
    main()
