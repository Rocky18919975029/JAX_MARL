#!/usr/bin/env python3
"""Confirm each map's pilot-selected actor-score-recovery setting on new seeds."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

try:
    from scripts.analyze_smax_actor_score_recovery_sweep import summarize
    from scripts.run_smax_actor_score_recovery_sweep import PROTOCOL
    from scripts.run_smax_actor_score_recovery_training import DEFAULT_BUDGETS
    from scripts.run_smax_score_recovery_training import (
        atomic_json,
        csv_items,
        seed_items,
    )
except ModuleNotFoundError:  # Direct execution from scripts/.
    from analyze_smax_actor_score_recovery_sweep import summarize
    from run_smax_actor_score_recovery_sweep import PROTOCOL
    from run_smax_actor_score_recovery_training import DEFAULT_BUDGETS
    from run_smax_score_recovery_training import atomic_json, csv_items, seed_items


MAPS = ("10m_vs_11m", "3s5z_vs_3s6z")


def confirmation_plan(
    sweep_root: Path,
    run_root: Path,
    maps: tuple[str, ...],
    seeds: tuple[int, ...],
    selection: dict,
) -> dict:
    pilot = json.loads((sweep_root / "experiment_manifest.json").read_text())
    if pilot.get("protocol") != PROTOCOL:
        raise ValueError(f"Not an actor-score-recovery sweep: {sweep_root}")
    if set(maps) - set(pilot["maps"]):
        raise ValueError("Requested maps are missing from the pilot sweep")
    if set(seeds) & set(pilot["seeds"]):
        raise ValueError("Confirmation seeds must not overlap pilot seeds")

    tasks = {}
    for task in maps:
        choice = selection[task]
        candidate = choice["top_candidate"]
        if candidate["condition"] != "actor_score_recovery":
            raise ValueError(f"No actor-score-recovery candidate for {task}")
        if int(candidate["budget"]) != int(pilot["budgets"][task]):
            raise ValueError(f"Pilot selection budget is inconsistent for {task}")
        tasks[task] = {
            "run_root": str(run_root / task),
            "training_budget": DEFAULT_BUDGETS[task],
            "pilot_budget": int(pilot["budgets"][task]),
            "pilot_seeds": list(pilot["seeds"]),
            "confirmatory_seeds": list(seeds),
            "selection_metric": choice["selection_metric"],
            "paired_delta_win_rate_auc_vs_none": candidate[
                "paired_delta_win_rate_auc_vs_none"
            ],
            "promote_to_confirmatory": choice["promote_to_confirmatory"],
            "coef": candidate["coef"],
            "q_steps": candidate["q_steps"],
            "q_learning_rate": candidate["q_learning_rate"],
            "fisher_ridge": candidate["fisher_ridge"],
        }
    return {
        "schema_version": 1,
        "pilot_root": str(sweep_root),
        "run_root": str(run_root),
        "protocol": "smax-nps-actor-score-recovery-confirmation-v1.0",
        "pilot_protocol": PROTOCOL,
        "pilot_ppo": {
            key: pilot[key]
            for key in (
                "update_epochs",
                "learning_rate",
                "num_envs",
                "num_minibatches",
                "checkpoint_interval",
            )
        },
        "tasks": tasks,
    }


def launcher_command(
    repo: Path,
    task: str,
    task_plan: dict,
    ppo: dict,
    gpus: tuple[str, ...],
    max_runs_per_gpu: int,
    project: str,
    wandb_mode: str,
) -> list[str]:
    return [
        sys.executable,
        str(repo / "scripts/run_smax_actor_score_recovery_sweep.py"),
        "--run-root",
        task_plan["run_root"],
        "--maps",
        task,
        "--seeds",
        ",".join(map(str, task_plan["confirmatory_seeds"])),
        "--conditions",
        "none,actor_score_recovery",
        "--coefs",
        str(task_plan["coef"]),
        "--q-steps-grid",
        str(task_plan["q_steps"]),
        "--q-learning-rates",
        str(task_plan["q_learning_rate"]),
        "--fisher-ridges",
        str(task_plan["fisher_ridge"]),
        "--update-epochs",
        str(ppo["update_epochs"]),
        "--learning-rate",
        str(ppo["learning_rate"]),
        "--num-envs",
        str(ppo["num_envs"]),
        "--num-minibatches",
        str(ppo["num_minibatches"]),
        "--checkpoint-interval",
        str(ppo["checkpoint_interval"]),
        "--gpus",
        ",".join(gpus),
        "--max-runs-per-gpu",
        str(max_runs_per_gpu),
        "--project",
        project,
        "--wandb-mode",
        wandb_mode,
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--maps", type=csv_items, default=MAPS)
    parser.add_argument("--seeds", type=seed_items, default=(1, 2, 3, 4))
    parser.add_argument("--gpus", type=csv_items, default=("0", "1", "2", "3"))
    parser.add_argument("--max-runs-per-gpu", type=int, default=1)
    parser.add_argument("--project", default="jaxmarl-smax-actor-score-recovery")
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="online"
    )
    parser.add_argument("--allow-nonpositive-gain", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if set(args.maps) - set(MAPS):
        parser.error(f"--maps must be drawn from {MAPS}")
    if not args.gpus or args.max_runs_per_gpu <= 0:
        parser.error("Provide GPUs and a positive concurrency limit")

    sweep_root = args.sweep_root.expanduser().resolve()
    run_root = args.run_root.expanduser().resolve()
    if run_root == sweep_root or sweep_root in run_root.parents:
        parser.error("Confirmation root must be separate from the pilot sweep root")

    # Recompute from completed pilot metrics; never trust a stale selection.json.
    selection = summarize(sweep_root)
    plan = confirmation_plan(sweep_root, run_root, args.maps, args.seeds, selection)
    for task, cell in plan["tasks"].items():
        print(
            f"{task}: lambda={cell['coef']:.10g} q_steps={cell['q_steps']} "
            f"q_lr={cell['q_learning_rate']:.10g} "
            f"ridge={cell['fisher_ridge']:.10g} "
            f"paired_win_AUC_gain={cell['paired_delta_win_rate_auc_vs_none']:.6g} "
            f"promote={cell['promote_to_confirmatory']} "
            f"budget={cell['training_budget']:,} runs={2 * len(args.seeds)}",
            flush=True,
        )
        if not cell["promote_to_confirmatory"] and not args.allow_nonpositive_gain:
            parser.error(
                f"{task} has no positive pilot win-rate AUC gain; "
                "pass --allow-nonpositive-gain only if this comparison is intentional"
            )

    repo = Path(__file__).resolve().parents[1]
    commands = [
        launcher_command(
            repo,
            task,
            cell,
            plan["pilot_ppo"],
            args.gpus,
            args.max_runs_per_gpu,
            args.project,
            args.wandb_mode,
        )
        for task, cell in plan["tasks"].items()
    ]
    if args.dry_run:
        for command in commands:
            print(shlex.join(command), flush=True)
        return

    run_root.mkdir(parents=True, exist_ok=True)
    plan_path = run_root / "confirmation_plan.json"
    if plan_path.is_file():
        if json.loads(plan_path.read_text()) != plan:
            raise RuntimeError(f"Confirmation plan changed: {plan_path}")
    else:
        atomic_json(plan_path, plan)
    for task, command in zip(plan["tasks"], commands):
        print(f"START {task}", flush=True)
        subprocess.run(command, cwd=repo, check=True)
        print(f"COMPLETE {task}", flush=True)
    print("All confirmation runs finished", flush=True)


if __name__ == "__main__":
    main()
