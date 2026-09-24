#!/usr/bin/env python3
"""Report the frozen first four-panel none/ARec cells over ten paired seeds.

Seeds 1--4 are the historical selections; seeds 5--10 are the extension. This
script never reselects a hyperparameter cell from the ten-seed outcomes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts import report_smax_arec_best_returns as previous
    from scripts.run_smax_first_four_panel_tuned import (
        CONFIG_KEYS, FIGURE_ID, TASKS, load_pair, same_value,
    )
except ModuleNotFoundError:  # Direct execution from scripts/.
    import report_smax_arec_best_returns as previous
    from run_smax_first_four_panel_tuned import (
        CONFIG_KEYS, FIGURE_ID, TASKS, load_pair, same_value,
    )


METHODS = ("none", "arec")
SEEDS = tuple(range(1, 11))
CONDITION = {"none": "none", "arec": "actor_score_recovery"}
ORIGIN = {"first_four_panel_v1": range(1, 5), "six_seed_extension": range(5, 11)}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_collection(root: Path) -> tuple[dict, dict]:
    """Require exactly the fixed four-task, paired ten-seed collection."""
    index_path = root / "collection_index.json"
    index = read_json(index_path)
    if index.get("figure_id") != FIGURE_ID or index.get("extension_seeds") != list(range(5, 11)):
        raise RuntimeError(f"Not the frozen first-four-panel collection: {index_path}")
    expected = {(task, method, seed) for task in TASKS for method in METHODS for seed in SEEDS}
    entries = {}
    for row in index.get("runs", []):
        key = row["task"], row["method"], int(row["seed"])
        if key not in expected or key in entries:
            raise RuntimeError(f"Unexpected or duplicated collection entry: {key}")
        if row.get("origin") not in ORIGIN or key[2] not in ORIGIN[row["origin"]]:
            raise RuntimeError(f"Wrong provenance for {key}: {row.get('origin')}")
        view = root / key[0] / "runs" / key[1] / f"seed_{key[2]:02d}"
        for label in ("source_manifest.json", "status.json", "metrics.jsonl", "checkpoints"):
            target = Path(row["outputs"][label])
            link = view / label
            if not target.exists() or not link.is_symlink() or link.resolve() != target.resolve():
                raise RuntimeError(f"Missing or mismatched collection link: {link}")
        status = read_json(Path(row["outputs"]["status.json"]))
        if status.get("status") != "completed":
            raise RuntimeError(f"Training is incomplete for {key}: {row['run_name']}")
        manifest = read_json(Path(row["outputs"]["source_manifest.json"]))
        source_runs = [
            item for item in manifest["runs"]
            if item.get("run_name", item.get("name")) == row["run_name"]
        ]
        if len(source_runs) != 1 or int(source_runs[0]["seed"]) != key[2]:
            raise RuntimeError(f"Source manifest does not identify {row['run_name']}")
        source = source_runs[0]
        if row["origin"] == "first_four_panel_v1":
            if source.get("condition") != CONDITION[key[1]] or source.get("map_name") != key[0]:
                raise RuntimeError(f"Wrong historical method/task for {key}")
        elif source.get("method") != key[1] or manifest.get("map_name") != key[0]:
            raise RuntimeError(f"Wrong extension method/task for {key}")
        entries[key] = row
    if set(entries) != expected:
        missing = sorted(expected - set(entries))
        raise RuntimeError(f"Ten-seed collection is incomplete: {len(entries)}/80; missing {missing[:5]}")
    return index, entries


def frozen_profiles() -> dict:
    profiles = {}
    for task in TASKS:
        none, arec, _ = load_pair(task)
        profiles[task] = {"none": none["config"], "arec": arec["config"]}
    return profiles


def last_five(row: dict, profile: dict) -> list[tuple[int, Path]]:
    parent = Path(row["outputs"]["checkpoints"])
    task, method, seed = row["task"], row["method"], int(row["seed"])
    source_manifest = read_json(Path(row["outputs"]["source_manifest.json"]))
    nominal_budget = int(profile["TOTAL_TIMESTEPS"])
    checkpoint_budget = (
        nominal_budget if row["origin"] == "first_four_panel_v1"
        else int(source_manifest["effective_timesteps"])
    )
    rollout_size = int(profile["NUM_ENVS"] * profile["NUM_STEPS"])
    if not nominal_budget - rollout_size < checkpoint_budget <= nominal_budget:
        raise RuntimeError(f"Checkpoint budget is not the matched whole-rollout budget: {parent}")
    expected_condition = (
        CONDITION[method] if row["origin"] == "first_four_panel_v1" else method
    )
    by_step: dict[int, Path] = {}
    for checkpoint in sorted(parent.iterdir()):
        if checkpoint.name != "final" and not checkpoint.name.startswith("step_"):
            continue
        if not (checkpoint / "model.safetensors").is_file():
            continue
        config = read_json(checkpoint / "config.json")
        metadata = read_json(checkpoint / "metadata.json")
        if int(config.get("SEED", -1)) != seed:
            raise RuntimeError(f"Checkpoint training seed differs: {checkpoint}")
        for key in CONFIG_KEYS:
            expected = (
                expected_condition if key == "EXPERIMENT_CONDITION"
                else checkpoint_budget if key == "TOTAL_TIMESTEPS"
                else profile[key]
            )
            if not same_value(key, config.get(key), expected):
                raise RuntimeError(
                    f"Checkpoint {key} differs from frozen method config: {checkpoint}; "
                    f"observed={config.get(key)!r}, expected={expected!r}"
                )
        if row.get("training_git_commit") not in (None, "unknown", config.get("GIT_COMMIT")):
            raise RuntimeError(f"Training commit differs from collection index: {checkpoint}")
        step = int(metadata["nominal_env_step"])
        if step < 1 or step > checkpoint_budget:
            raise RuntimeError(f"Invalid nominal checkpoint step: {checkpoint}")
        if step not in by_step or checkpoint.name == "final":
            by_step[step] = checkpoint
    if (checkpoint_budget not in by_step or by_step[checkpoint_budget].name != "final"
            or len(by_step) < 5):
        raise RuntimeError(f"Need final and five distinct checkpoints: {parent}")
    return [(step, by_step[step]) for step in sorted(by_step)[-5:]]


def prepare(root: Path, output: Path) -> tuple[dict, dict, list[previous.EvalJob], dict]:
    index, entries = load_collection(root)
    profiles = frozen_profiles()
    selections, checkpoint_steps, jobs = {}, {}, []
    for task in TASKS:
        budget = int(profiles[task]["none"]["TOTAL_TIMESTEPS"])
        if int(profiles[task]["arec"]["TOTAL_TIMESTEPS"]) != budget:
            raise RuntimeError(f"Unmatched method budgets for {task}")
        groups, histories, aucs, step_sets = {}, {}, {}, []
        for method in METHODS:
            group = {}
            profile = profiles[task][method]
            for seed in SEEDS:
                row = entries[task, method, seed]
                name = row["run_name"]
                points = previous.clean_history(Path(row["outputs"]["metrics.jsonl"]), budget)
                if int(points[-1]["env_step"]) < budget - int(profile["NUM_ENVS"] * profile["NUM_STEPS"]):
                    raise RuntimeError(f"Training history ends too early: {name}")
                histories[name] = points
                aucs[name] = previous.metric_summary(points, "returns", budget)[0]
                step_sets.append({int(point["env_step"]) for point in points})
                group[seed] = {
                    "run_name": name, "map_name": task, "seed": seed,
                    "condition": CONDITION[method],
                    "coef": float(profile["ACTOR_SCORE_RECOVERY_COEF"]),
                    "q_steps": int(profile["ACTOR_SCORE_RECOVERY_Q_STEPS"]),
                    "q_learning_rate": float(profile["ACTOR_SCORE_RECOVERY_Q_LR"]),
                    "fisher_ridge": float(profile["ACTOR_SCORE_RECOVERY_FISHER_RIDGE"]),
                    "learning_rate": float(profile["LR"]),
                    "update_epochs": int(profile["UPDATE_EPOCHS"]),
                }
                checkpoints = last_five(row, profile)
                checkpoint_steps[task, CONDITION[method], seed] = tuple(
                    step for step, _ in checkpoints
                )
                for checkpoint_index, (step, checkpoint) in enumerate(checkpoints):
                    jobs.append(previous.EvalJob(
                        task, CONDITION[method], seed, step, checkpoint,
                        output / "evaluation" / task / CONDITION[method]
                        / f"seed_{seed}" / f"step_{step:012d}.json",
                        100_000 + 100 * seed + checkpoint_index,
                    ))
            groups[method] = group
        for seed in SEEDS:
            if checkpoint_steps[task, "none", seed] != checkpoint_steps[task, "actor_score_recovery", seed]:
                raise RuntimeError(f"Paired last-five checkpoint steps differ in {task}/seed{seed}")
        shared_steps = set.intersection(*step_sets)
        if (len(shared_steps) < max(3, (max(map(len, step_sets)) + 1) // 2)
                or max(shared_steps, default=0)
                < budget - int(profiles[task]["none"]["NUM_ENVS"] * profiles[task]["none"]["NUM_STEPS"])):
            raise RuntimeError(f"Training return grids are not sufficiently matched in {task}")
        selections[task] = {
            "task": task, "budget": budget, "seeds": SEEDS,
            "baseline": groups["none"], "best": groups["arec"],
            "baseline_params": None,
            "best_params": (
                float(profiles[task]["arec"]["ACTOR_SCORE_RECOVERY_COEF"]),
                int(profiles[task]["arec"]["ACTOR_SCORE_RECOVERY_Q_STEPS"]),
                float(profiles[task]["arec"]["ACTOR_SCORE_RECOVERY_Q_LR"]),
                float(profiles[task]["arec"]["ACTOR_SCORE_RECOVERY_FISHER_RIDGE"]),
            ),
            "histories": histories, "aucs": aucs,
        }
    return index, entries, jobs, (selections, checkpoint_steps)


def write_table(path: Path, summary: list[dict]) -> None:
    def estimate(row: dict, name: str) -> str:
        return (
            f"{row[name + '_mean']:.3f} "
            f"[{row[name + '_ci95_low']:.3f}, {row[name + '_ci95_high']:.3f}]"
        )

    lines = [
        "| Task | Method | Training return AUC (95% CI) | Final held-out return, last 5 checkpoints (95% CI) | Paired ΔAUC vs none (95% CI) | Paired Δfinal vs none (95% CI) |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append("| " + " | ".join((
            row["task"], "Isolated" if row["condition"] == "none" else "ARec",
            estimate(row, "return_auc"),
            estimate(row, "final_eval_return_last5_ckpt"),
            estimate(row, "delta_return_auc_vs_none"),
            estimate(row, "delta_final_eval_return_last5_ckpt_vs_none"),
        )) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def report(args: argparse.Namespace) -> Path:
    collection = args.collection_root.expanduser().resolve()
    output = args.output_root.expanduser().resolve()
    if output == collection or collection not in output.parents:
        raise ValueError("Output root must be a dedicated child of the collection")
    index, entries, jobs, (selections, checkpoint_steps) = prepare(collection, output)
    output.mkdir(parents=True, exist_ok=True)
    previous.evaluate_jobs(
        jobs, episodes=args.eval_episodes, num_envs=args.eval_num_envs,
        policy=args.eval_policy, evaluate_missing=args.evaluate_missing,
        gpus=args.gpus, max_runs_per_gpu=args.max_runs_per_gpu,
        reuse_evaluation_root=args.reuse_evaluation_root,
    )
    for job in jobs:
        record = previous.checked_eval(job, args.eval_episodes, args.eval_policy)
        method = "none" if job.condition == "none" else "arec"
        row = entries[job.task, method, job.seed]
        expected_condition = CONDITION[method] if row["origin"] == "first_four_panel_v1" else method
        if int(record.get("num_envs", -1)) != args.eval_num_envs or record.get("condition") != expected_condition:
            raise RuntimeError(f"Held-out evaluation settings differ: {job.output}")
    summary, seeds, curves = previous.build_rows(
        selections, jobs, checkpoint_steps, episodes=args.eval_episodes,
        policy=args.eval_policy, bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    for row in summary:
        row["selection_metric"] = "frozen from first four-panel selection; not reselected on 10 seeds"
        row["source_code_cohort"] = "historical seeds 1-4; current-code extension seeds 5-10"
        step_sets = {
            checkpoint_steps[row["task"], row["condition"], seed] for seed in SEEDS
        }
        if len(step_sets) > 1:
            row["final_checkpoint_steps"] = "varies slightly by code cohort; see seed_level_all_tasks.csv"
    for row in seeds:
        method = "none" if row["condition"] == "none" else "arec"
        entry = entries[row["task"], method, int(row["seed"])]
        row["origin"] = entry["origin"]
        row["source_root"] = entry["source_root"]
        row["training_git_commit"] = entry["training_git_commit"]
        row["manifest_git_commit"] = entry.get(
            "manifest_git_commit", entry["training_git_commit"]
        )
        row["commit_reconciled"] = entry.get("commit_reconciled", False)
    previous.write_csv(output / "summary_all_tasks.csv", summary)
    previous.write_csv(output / "seed_level_all_tasks.csv", seeds)
    previous.write_csv(output / "return_curve_all_tasks.csv", curves)
    for task in TASKS:
        task_output = output / task
        previous.write_csv(task_output / "summary.csv", [r for r in summary if r["task"] == task])
        previous.write_csv(task_output / "seed_level.csv", [r for r in seeds if r["task"] == task])
        previous.write_csv(task_output / "return_curve.csv", [r for r in curves if r["task"] == task])
    write_table(output / "summary_table.md", summary)
    figure = output / "smax-arec-selected-return-curves-10seed"
    previous.render_figure(curves, selections, figure)
    manifest = {
        "schema_version": 1,
        "figure_id": "smax-arec-selected-return-curves-10seed",
        "source_collection_index": str(collection / "collection_index.json"),
        "source_figure_id": index["figure_id"],
        "tasks": list(TASKS), "methods": list(METHODS), "seeds": list(SEEDS),
        "selected_configs": {task: {
            method: str(Path("configs/smax_first_four_panel") / f"{task}_{method}.yaml")
            for method in METHODS
        } for task in TASKS},
        "source_runs": [{
            "task": row["task"], "method": row["method"], "seed": row["seed"],
            "run_name": row["run_name"], "origin": row["origin"],
            "training_git_commit": row["training_git_commit"],
            "manifest_git_commit": row.get("manifest_git_commit", row["training_git_commit"]),
            "commit_reconciled": row.get("commit_reconciled", False),
            "source_root": row["source_root"],
        } for _, row in sorted(entries.items())],
        "curve_metric": "unsmoothed training episode return",
        "auc_definition": "trapezoidal return integral over nominal environment-step budget, divided by budget; first and last observed returns extended to boundaries",
        "final_definition": "per-seed mean of held-out stochastic-policy return at five distinct final saved checkpoints, then mean across ten paired training seeds",
        "eval_policy": args.eval_policy,
        "eval_episodes_per_checkpoint": args.eval_episodes,
        "eval_num_envs": args.eval_num_envs,
        "eval_seed_rule": "100000 + 100 * training_seed + last-five-checkpoint index",
        "uncertainty": "Ordinary paired training-seed bootstrap resamples; pointwise 95% percentile CI for curves",
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_resamples": args.bootstrap_samples,
        "code_cohort_warning": "Historical seeds 1-4 and extension seeds 5-10 use different code commits; matched effective YAML configs, not a bitwise homogeneous replay",
        "checkpoint_step_warning": "Historical final checkpoints use the nominal task budget; extension final checkpoints use the last complete rollout (within one rollout block of nominal). Each none/ARec seed pair is matched at the same five checkpoint steps.",
        "selection_warning": "Settings were chosen using the original four seeds, which are included in this ten-seed report; inference is partly selection-biased",
        "figure_contract": {
            "claim": "Compare frozen ARec and isolated NPS MAPPO across four tasks and ten paired seeds",
            "five_second_takeaway": "Blue ARec versus gray isolated return within each task panel",
            "audience": "multi-agent reinforcement learning researchers",
            "publication_width_mm": 178,
            "layout": "2 by 2 aligned training-curve small multiples",
            "visual_anchor": "task-specific isolated baseline",
            "semantic_encoding": {"none": "gray dashed", "arec": "blue solid", "uncertainty": "transparent 95% CI"},
            "editable_source": "scripts/report_smax_arec_best_returns.py:render_figure",
            "data_pipeline": "scripts/report_smax_first_four_panel_10seed.py",
            "exports": ["svg", "pdf", "png"],
        },
    }
    (output / "report_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "figure_caption.txt").write_text(
        "Unsmoothed SMAX/SMACv2 training episode return for frozen isolated NPS MAPPO "
        "(gray dashed) and actor-side score recovery (blue solid) across ten paired "
        "training seeds per task. Bands are pointwise 95% percentile bootstrap "
        "confidence intervals resampled by training seed. The table reports "
        "training-return AUC and held-out stochastic-policy return averaged over "
        f"each seed's last five distinct checkpoints ({args.eval_episodes} episodes per checkpoint). "
        "Hyperparameters were selected on historical seeds 1-4, "
        "which are included here; seeds 5-10 use a later code commit with the "
        "same effective frozen configurations. Comparisons are therefore partly "
        "selection-biased and not a bitwise homogeneous-code replication. "
        "Extension final checkpoints are at the last complete rollout, slightly before "
        "the nominal budget; each same-seed method pair uses matching checkpoint steps.\n",
        encoding="utf-8",
    )
    print(figure.with_suffix(".png"), flush=True)
    print(output / "summary_table.md", flush=True)
    print(output / "summary_all_tasks.csv", flush=True)
    return figure.with_suffix(".png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reuse-evaluation-root", type=Path)
    parser.add_argument("--evaluate-missing", action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=256)
    parser.add_argument("--eval-num-envs", type=int, default=128)
    parser.add_argument("--eval-policy", choices=("stochastic", "deterministic"), default="stochastic")
    parser.add_argument("--gpus", type=lambda value: tuple(value.split(",")), default=("0", "1", "2", "3"))
    parser.add_argument("--max-runs-per-gpu", type=int, default=2)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260924)
    args = parser.parse_args()
    if min(args.eval_episodes, args.eval_num_envs, args.bootstrap_samples) <= 0:
        parser.error("Evaluation size and bootstrap count must be positive")
    if not args.gpus or any(not gpu for gpu in args.gpus) or not 1 <= args.max_runs_per_gpu <= 2:
        parser.error("Give GPU IDs and at most two simultaneous runs per GPU")
    report(args)


if __name__ == "__main__":
    main()
