#!/usr/bin/env python3
"""Compare each task's best ARec sweep cell with its paired isolated runs.

Selection uses mean seed-paired *training return AUC* within each task. The
learning curves also use training returns; final performance instead evaluates
the last five distinct saved checkpoints on held-out episodes. These two data
sources are deliberately labelled separately in the output table. All task
selection and aggregation remain task-specific, even in a combined figure.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from scripts.analyze_smax_actor_score_recovery_sweep import (
        PROTOCOL,
        history,
        metric_summary,
        write_csv,
    )
except ModuleNotFoundError:  # Direct execution from scripts/.
    from analyze_smax_actor_score_recovery_sweep import (
        PROTOCOL,
        history,
        metric_summary,
        write_csv,
    )


TASKS = ("10m_vs_11m", "3s5z_vs_3s6z")
THIRD_TASK = "6s9z_vs_6s10z"
FOURTH_TASK = "smacv2_10_units"
REPO = Path(__file__).resolve().parents[1]
BASELINE_COLOR = "#343A40"
RECOVERY_COLOR = "#0072B2"


@dataclass(frozen=True)
class EvalJob:
    task: str
    condition: str
    seed: int
    nominal_step: int
    checkpoint: Path
    output: Path
    eval_seed: int


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def clean_history(path: Path, budget: int) -> list[dict]:
    rows = []
    for row in history(path):
        step = int(row["env_step"])
        try:
            returns = float(row["returns"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 < step <= budget and math.isfinite(returns):
            rows.append({"env_step": step, "returns": returns})
    if len(rows) < 2:
        raise RuntimeError(f"Need two finite return observations: {path}")
    return rows


def setting(run: dict) -> tuple[float, int, float, float]:
    return (
        float(run["coef"]),
        int(run["q_steps"]),
        float(run["q_learning_rate"]),
        float(run["fisher_ridge"]),
    )


def table_parameters(run: dict) -> dict:
    keys = ("coef", "q_steps", "q_learning_rate", "fisher_ridge")
    if run["condition"] == "none":
        return dict.fromkeys(keys, "")
    return dict(zip(keys, setting(run)))


def select_runs(root: Path, task: str, expected_seeds=(1, 2, 3, 4)) -> dict:
    manifest = read_json(root / "experiment_manifest.json")
    if manifest.get("protocol") != PROTOCOL or manifest.get("maps") != [task]:
        raise RuntimeError(f"Expected a single-task ARec sweep for {task}: {root}")
    seeds = tuple(int(seed) for seed in manifest["seeds"])
    if seeds != tuple(expected_seeds):
        raise RuntimeError(
            f"{task} needs matched training seeds {expected_seeds}; found {seeds} "
            f"in {root}. Point to its four-seed confirmation/sweep root."
        )
    budget = int(manifest["budgets"][task])
    runs = manifest["runs"]
    baseline = {}
    candidates: dict[tuple, dict[int, dict]] = defaultdict(dict)
    histories = {}
    aucs = {}
    for run in runs:
        if run["map_name"] != task or int(run["steps"]) != budget:
            raise RuntimeError(f"Unexpected map or budget in {root}: {run['run_name']}")
        name, seed = run["run_name"], int(run["seed"])
        state_path = root / "status" / f"{name}.json"
        if (
            not state_path.is_file()
            or read_json(state_path).get("status") != "completed"
        ):
            raise RuntimeError(f"Sweep run not completed: {name}")
        rows = clean_history(root / "metrics" / f"{name}.jsonl", budget)
        histories[name] = rows
        aucs[name] = metric_summary(rows, "returns", budget)[0]
        if run["condition"] == "none":
            if seed in baseline:
                raise RuntimeError(f"Duplicate baseline for {task} seed {seed}")
            baseline[seed] = run
        elif run["condition"] == "actor_score_recovery":
            cell = candidates[setting(run)]
            if seed in cell:
                raise RuntimeError(f"Duplicate candidate for {task} seed {seed}")
            cell[seed] = run
        else:
            raise RuntimeError(f"Unexpected condition in {root}: {run['condition']}")
    if set(baseline) != set(seeds) or not candidates:
        raise RuntimeError(f"Missing baseline or ARec candidate for {task}")
    if any(set(cell) != set(seeds) for cell in candidates.values()):
        raise RuntimeError(f"Incomplete candidate seed group for {task}")

    def rank(item):
        params, cell = item
        delta = np.mean(
            [
                aucs[cell[seed]["run_name"]] - aucs[baseline[seed]["run_name"]]
                for seed in seeds
            ]
        )
        # Fixed tie-breakers are independent of held-out checkpoint evaluation.
        return (float(delta), -params[0], -params[1], -params[2], -params[3])

    best_params, best = max(candidates.items(), key=rank)
    return {
        "root": root,
        "task": task,
        "budget": budget,
        "seeds": seeds,
        "baseline": baseline,
        "best": best,
        "best_params": best_params,
        "histories": histories,
        "aucs": aucs,
        "paired_return_auc_gain": rank((best_params, best))[0],
    }


def last_five_checkpoints(root: Path, run: dict, budget: int) -> list[tuple[int, Path]]:
    name = run["run_name"]
    matches = list((root / "checkpoints").glob(f"**/{name}-*/final/model.safetensors"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one final checkpoint for {name}; found {len(matches)}"
        )
    run_dir = matches[0].parent.parent
    by_step = {}
    for ckpt in run_dir.iterdir():
        if ckpt.name == "initial" or not (ckpt / "model.safetensors").is_file():
            continue
        metadata_path = ckpt / "metadata.json"
        config_path = ckpt / "config.json"
        if not metadata_path.is_file() or not config_path.is_file():
            raise RuntimeError(f"Incomplete checkpoint: {ckpt}")
        metadata = read_json(metadata_path)
        config = read_json(config_path)
        if (
            config.get("MAP_NAME") != run["map_name"]
            or int(config.get("SEED", -1)) != int(run["seed"])
            or config.get("EXPERIMENT_CONDITION") != run["condition"]
            or config.get("ACTOR_PARAMETER_SHARING") is not False
        ):
            raise RuntimeError(f"Checkpoint identity mismatch: {ckpt}")
        step = int(metadata["nominal_env_step"])
        if 0 < step <= budget and (step not in by_step or ckpt.name == "final"):
            by_step[step] = ckpt
    if budget not in by_step or by_step[budget].name != "final" or len(by_step) < 5:
        raise RuntimeError(
            f"Need final and five distinct saved checkpoint steps: {name}"
        )
    return [(step, by_step[step]) for step in sorted(by_step)[-5:]]


def make_eval_jobs(selections: dict, output_root: Path) -> tuple[list[EvalJob], dict]:
    jobs = []
    checkpoint_steps = {}
    for task in selections:
        selected = selections[task]
        for seed in selected["seeds"]:
            for condition, group in (
                ("none", selected["baseline"]),
                ("actor_score_recovery", selected["best"]),
            ):
                run = group[seed]
                checkpoints = last_five_checkpoints(
                    selected["root"], run, selected["budget"]
                )
                steps = tuple(step for step, _ in checkpoints)
                checkpoint_steps[task, condition, seed] = steps
                for index, (step, ckpt) in enumerate(checkpoints):
                    jobs.append(
                        EvalJob(
                            task,
                            condition,
                            seed,
                            step,
                            ckpt,
                            output_root
                            / "evaluation"
                            / task
                            / condition
                            / f"seed_{seed}"
                            / f"step_{step:012d}.json",
                            100_000 + 100 * seed + index,
                        )
                    )
        expected = checkpoint_steps[task, "none", selected["seeds"][0]]
        if any(
            steps != expected
            for key, steps in checkpoint_steps.items()
            if key[0] == task
        ):
            raise RuntimeError(f"Last-five checkpoint steps do not match within {task}")
    return jobs, checkpoint_steps


def checked_eval(job: EvalJob, episodes: int, policy: str) -> dict:
    result = read_json(job.output)
    if (
        Path(result.get("checkpoint", "")).resolve() != job.checkpoint.resolve()
        or int(result.get("episodes", -1)) != episodes
        or int(result.get("eval_seed", -1)) != job.eval_seed
        or result.get("policy") != policy
        or result.get("map_name") != job.task
        or int(result.get("training_seed", -1)) != job.seed
        or int(result.get("checkpoint_nominal_env_step", -1)) != job.nominal_step
        or not math.isfinite(float(result["return_mean"]))
    ):
        raise RuntimeError(f"Evaluation protocol mismatch: {job.output}")
    return result


def evaluate_jobs(
    jobs: list[EvalJob],
    *,
    episodes: int,
    num_envs: int,
    policy: str,
    evaluate_missing: bool,
    gpus: tuple[str, ...],
    max_runs_per_gpu: int,
) -> None:
    pending = []
    for job in jobs:
        if job.output.is_file():
            checked_eval(job, episodes, policy)
        else:
            pending.append(job)
    if not pending:
        return
    if not evaluate_missing:
        raise RuntimeError(
            f"Missing {len(pending)} held-out checkpoint evaluations. "
            "Rerun with --evaluate-missing to evaluate only those checkpoints."
        )
    slots = [gpu for gpu in gpus for _ in range(max_runs_per_gpu)]
    queues = [[] for _ in slots]
    for index, job in enumerate(pending):
        queues[index % len(slots)].append(job)

    def worker(gpu: str, queue: list[EvalJob]) -> list[str]:
        errors = []
        for job in queue:
            job.output.parent.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                str(REPO / "baselines/MAPPO/eval_mappo_rnn_smax.py"),
                "--checkpoint",
                str(job.checkpoint),
                "--episodes",
                str(episodes),
                "--num-envs",
                str(num_envs),
                "--seed",
                str(job.eval_seed),
                "--policy",
                policy,
                "--output",
                str(job.output),
            ]
            env = dict(os.environ)
            env.pop("LD_LIBRARY_PATH", None)
            env.update(
                {"CUDA_VISIBLE_DEVICES": gpu, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
            )
            print(
                f"EVAL GPU {gpu} {job.task}/{job.condition}/seed{job.seed}/step{job.nominal_step}",
                flush=True,
            )
            with job.output.with_suffix(".log").open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    command,
                    cwd=REPO,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if result.returncode:
                errors.append(f"{job.output}: exit={result.returncode}")
            else:
                checked_eval(job, episodes, policy)
        return errors

    with ThreadPoolExecutor(max_workers=len(slots)) as pool:
        futures = [pool.submit(worker, gpu, queue) for gpu, queue in zip(slots, queues)]
        errors = list(
            itertools.chain.from_iterable(future.result() for future in futures)
        )
    if errors:
        raise RuntimeError(f"{len(errors)} checkpoint evaluations failed: {errors[:5]}")


def bootstrap_mean(
    values: np.ndarray, indices: np.ndarray
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) != indices.shape[1] or not np.isfinite(values).all():
        raise RuntimeError("Bootstrap input has missing or non-finite seed values")
    sampled = values[indices].mean(axis=1)
    low, high = np.quantile(sampled, (0.025, 0.975))
    return float(values.mean()), float(low), float(high)


def bootstrap_indices(n_seeds: int, n_samples: int, seed: int) -> np.ndarray:
    if n_seeds <= 6:
        return np.asarray(list(itertools.product(range(n_seeds), repeat=n_seeds)))
    return np.random.default_rng(seed).integers(0, n_seeds, size=(n_samples, n_seeds))


def build_rows(
    selections: dict,
    jobs: list[EvalJob],
    checkpoint_steps: dict,
    *,
    episodes: int,
    policy: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
):
    evaluation = {
        (job.task, job.condition, job.seed, job.nominal_step): checked_eval(
            job, episodes, policy
        )
        for job in jobs
    }
    summary_rows, seed_rows, curve_rows = [], [], []
    for task in selections:
        selected = selections[task]
        seeds, budget = selected["seeds"], selected["budget"]
        indices = bootstrap_indices(len(seeds), bootstrap_samples, bootstrap_seed)
        per_condition = {}
        for condition, group in (
            ("none", selected["baseline"]),
            ("actor_score_recovery", selected["best"]),
        ):
            per_seed = {}
            series = []
            for seed in seeds:
                run = group[seed]
                points = selected["histories"][run["run_name"]]
                step_values = checkpoint_steps[task, condition, seed]
                x = np.asarray([row["env_step"] for row in points], dtype=np.float64)
                y = np.asarray([row["returns"] for row in points], dtype=np.float64)
                train_final5 = float(np.interp(step_values, x, y).mean())
                eval_final5 = float(
                    np.mean(
                        [
                            evaluation[task, condition, seed, step]["return_mean"]
                            for step in step_values
                        ]
                    )
                )
                per_seed[seed] = {
                    "return_auc": selected["aucs"][run["run_name"]],
                    "final_train_return_last5_ckpt": train_final5,
                    "final_eval_return_last5_ckpt": eval_final5,
                }
                seed_rows.append(
                    {
                        "task": task,
                        "condition": condition,
                        "seed": seed,
                        "run_name": run["run_name"],
                        "budget_env_steps": budget,
                        **table_parameters(run),
                        "final_checkpoint_steps": ";".join(map(str, step_values)),
                        **per_seed[seed],
                    }
                )
                by_step = {
                    int(row["env_step"]): float(row["returns"]) for row in points
                }
                by_step[0] = float(y[0])
                by_step[budget] = float(np.interp(budget, x, y))
                series.append(by_step)
            per_condition[condition] = per_seed
            common_steps = sorted(set.intersection(*(set(item) for item in series)))
            if len(common_steps) < 2:
                raise RuntimeError(
                    f"No shared environment-step grid for {task}/{condition}"
                )
            matrix = np.asarray(
                [[item[step] for step in common_steps] for item in series]
            )
            boot = matrix[indices].mean(axis=1)
            low, high = np.quantile(boot, (0.025, 0.975), axis=0)
            for i, step in enumerate(common_steps):
                curve_rows.append(
                    {
                        "task": task,
                        "condition": condition,
                        "env_step": step,
                        "mean_training_return": float(matrix[:, i].mean()),
                        "ci95_low": float(low[i]),
                        "ci95_high": float(high[i]),
                        "n_seeds": len(seeds),
                    }
                )
        for condition in ("none", "actor_score_recovery"):
            group = selected["baseline"] if condition == "none" else selected["best"]
            row = {
                "task": task,
                "condition": condition,
                "n_seeds": len(seeds),
                "seeds": ";".join(map(str, seeds)),
                "budget_env_steps": budget,
                **table_parameters(group[seeds[0]]),
                "selection_metric": "mean seed-paired training return AUC gain",
                "final_eval_policy": policy,
                "final_eval_episodes_per_checkpoint": episodes,
                "final_checkpoint_steps": ";".join(
                    map(str, checkpoint_steps[task, condition, seeds[0]])
                ),
            }
            for metric in (
                "return_auc",
                "final_train_return_last5_ckpt",
                "final_eval_return_last5_ckpt",
            ):
                values = np.asarray(
                    [per_condition[condition][seed][metric] for seed in seeds]
                )
                mean, low, high = bootstrap_mean(values, indices)
                row[f"{metric}_mean"] = mean
                row[f"{metric}_ci95_low"] = low
                row[f"{metric}_ci95_high"] = high
                paired = np.asarray(
                    [
                        per_condition[condition][seed][metric]
                        - per_condition["none"][seed][metric]
                        for seed in seeds
                    ]
                )
                delta, delta_low, delta_high = bootstrap_mean(paired, indices)
                row[f"delta_{metric}_vs_none_mean"] = delta
                row[f"delta_{metric}_vs_none_ci95_low"] = delta_low
                row[f"delta_{metric}_vs_none_ci95_high"] = delta_high
            summary_rows.append(row)
    return summary_rows, seed_rows, curve_rows


def render_figure(curve_rows: list[dict], selections: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    tasks = tuple(selections)
    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    ):
        if len(tasks) == 4:
            fig, grid = plt.subplots(2, 2, figsize=(178 / 25.4, 130 / 25.4))
            axes = tuple(grid.flat)
        else:
            fig, grid = plt.subplots(1, len(tasks), figsize=(178 / 25.4, 94 / 25.4))
            axes = tuple(np.atleast_1d(grid).flat)
        for index, (ax, task) in enumerate(zip(axes, tasks)):
            for condition, color, linestyle in (
                ("none", BASELINE_COLOR, (0, (5, 2))),
                ("actor_score_recovery", RECOVERY_COLOR, "-"),
            ):
                rows = sorted(
                    (
                        row
                        for row in curve_rows
                        if row["task"] == task and row["condition"] == condition
                    ),
                    key=lambda row: row["env_step"],
                )
                x = np.asarray([row["env_step"] / 1e6 for row in rows])
                y = np.asarray([row["mean_training_return"] for row in rows])
                low = np.asarray([row["ci95_low"] for row in rows])
                high = np.asarray([row["ci95_high"] for row in rows])
                ax.fill_between(x, low, high, color=color, alpha=0.14, linewidth=0)
                ax.plot(x, y, color=color, linestyle=linestyle, linewidth=1.35)
            if task == FOURTH_TASK:
                ax.set_title("SMACv2\n10 units")
            else:
                task_label = task.replace("_vs_", " vs ").replace("_", " ")
                ax.set_title(
                    "SMAX\n" + task_label if len(tasks) >= 3 else "SMAX — " + task_label
                )
            if len(tasks) != 4:
                ax.set_xlabel("Environment steps (millions)")
            elif index >= 2:
                ax.set_xlabel("Environment steps (millions)")
            ax.set_xlim(0, selections[task]["budget"] / 1e6)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.grid(axis="y", color="#D9DEE5", linewidth=0.8)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(direction="out", width=0.8, length=3)
        if len(tasks) == 4:
            axes[0].set_ylabel("Training episode return")
            axes[2].set_ylabel("Training episode return")
        else:
            axes[0].set_ylabel("Training episode return")
        fig.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    color=BASELINE_COLOR,
                    linestyle="--",
                    linewidth=1.35,
                    label="Isolated",
                ),
                Line2D(
                    [0],
                    [0],
                    color=RECOVERY_COLOR,
                    linewidth=1.35,
                    label="Score recovery",
                ),
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, 1),
            ncol=2,
            frameon=False,
        )
        if len(tasks) == 4:
            fig.subplots_adjust(
                left=0.11,
                right=0.985,
                bottom=0.11,
                top=0.84,
                wspace=0.32,
                hspace=0.54,
            )
        else:
            fig.subplots_adjust(
                left=0.085,
                right=0.985,
                bottom=0.15,
                top=0.79 if len(tasks) == 3 else 0.82,
                wspace=0.29 if len(tasks) == 3 else 0.21,
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        for suffix in ("png", "pdf", "svg"):
            fig.savefig(output.with_suffix(f".{suffix}"), dpi=350)
        plt.close(fig)


def report(args) -> Path:
    roots = dict(zip(TASKS, (args.root_10m, args.root_3s5z)))
    if getattr(args, "root_6s9z", None) is not None:
        roots[THIRD_TASK] = args.root_6s9z
    if getattr(args, "root_smacv2", None) is not None:
        roots[FOURTH_TASK] = args.root_smacv2
    tasks = tuple(roots)
    selections = {
        task: select_runs(root.resolve(), task) for task, root in roots.items()
    }
    output = args.output_root.expanduser().resolve()
    jobs, checkpoint_steps = make_eval_jobs(selections, output)
    for task, selected in selections.items():
        print(
            f"{task}: λ={selected['best_params'][0]:.10g} "
            f"q={selected['best_params'][1]} "
            f"paired return-AUC gain={selected['paired_return_auc_gain']:.6g} "
            f"last-five steps={checkpoint_steps[task, 'none', 1]}",
            flush=True,
        )
    evaluate_jobs(
        jobs,
        episodes=args.eval_episodes,
        num_envs=args.eval_num_envs,
        policy=args.eval_policy,
        evaluate_missing=args.evaluate_missing,
        gpus=args.gpus,
        max_runs_per_gpu=args.max_runs_per_gpu,
    )
    summary, seeds, curves = build_rows(
        selections,
        jobs,
        checkpoint_steps,
        episodes=args.eval_episodes,
        policy=args.eval_policy,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    for task in tasks:
        task_out = output / task
        write_csv(
            task_out / "summary.csv", [row for row in summary if row["task"] == task]
        )
        write_csv(
            task_out / "seed_level.csv", [row for row in seeds if row["task"] == task]
        )
        write_csv(
            task_out / "return_curve.csv",
            [row for row in curves if row["task"] == task],
        )
    write_csv(output / "summary_all_tasks.csv", summary)
    manifest = {
        "schema_version": 1,
        "figure_contract": {
            "claim": "Task-specific selected actor score recovery versus each task's isolated baseline",
            "five_second_takeaway": "Compare each blue return curve with the gray baseline in the same task panel",
            "publication_width_mm": 178,
            "layout": "2x2 small multiples for four tasks; one row for two or three tasks",
            "audience": "multi-agent reinforcement learning researchers",
            "visual_anchor": "task-specific isolated MAPPO baseline",
            "semantic_encoding": {
                "isolated": "dark gray dashed line",
                "selected_actor_score_recovery": "blue solid line",
                "uncertainty": "translucent pointwise 95% training-seed bootstrap band",
            },
            "source": "local completed sweep metrics and held-out evaluation of saved checkpoints",
            "editable_source": "scripts/report_smax_arec_best_returns.py",
            "exports": ["svg", "pdf", "png"],
        },
        "source_roots": {task: str(roots[task].resolve()) for task in tasks},
        "tasks_are_never_pooled": True,
        "selection": {
            task: {
                "metric": "mean seed-paired training return AUC gain vs own none",
                "parameters": dict(
                    zip(
                        ("coef", "q_steps", "q_learning_rate", "fisher_ridge"),
                        selections[task]["best_params"],
                    )
                ),
                "paired_return_auc_gain": selections[task]["paired_return_auc_gain"],
                "seeds": selections[task]["seeds"],
            }
            for task in tasks
        },
        "curve_source": "local training metrics JSONL; exact shared env_step grid; no smoothing",
        "auc_definition": "trapezoidal training return integral divided by task budget",
        "final_train_definition": "per-seed mean of interpolated training return at last five distinct saved checkpoint steps",
        "final_eval_definition": "per-seed mean of held-out episode return over last five distinct saved checkpoint policies",
        "eval_policy": args.eval_policy,
        "eval_episodes_per_checkpoint": args.eval_episodes,
        "evaluation_seed_rule": "100000 + 100 * training_seed + last-five-checkpoint index",
        "bootstrap_unit": "training seed",
        "bootstrap_ci": "pointwise 95% percentile ordinary bootstrap; exact ordered resamples for four seeds",
        "bootstrap_resamples": int(
            bootstrap_indices(4, args.bootstrap_samples, args.bootstrap_seed).shape[0]
        ),
        "selection_warning": "The same four seeds select the hyperparameters and estimate the plotted performance; this is exploratory, not independent confirmation.",
    }
    (output / "report_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    figure = output / "smax-arec-selected-return-curves"
    render_figure(curves, selections, figure)
    caption = (
        "SMAX and SMACv2 training episode return for the isolated NPS MAPPO baseline and "
        "the task-specific actor-side score-recovery setting selected by paired "
        "return AUC on four training seeds. Lines are seed means; bands are "
        "pointwise 95% exact seed-bootstrap intervals. Each task is "
        "analyzed separately. The table reports training return AUC and "
        "held-out return averaged over the last five distinct saved checkpoints "
        "per seed. Hyperparameters were selected on these same seeds, so the "
        "comparison is exploratory.\n"
    )
    (output / "figure_caption.txt").write_text(caption, encoding="utf-8")
    print(figure.with_suffix(".png"))
    print(output / "summary_all_tasks.csv")
    return figure.with_suffix(".png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-10m", type=Path, required=True)
    parser.add_argument("--root-3s5z", type=Path, required=True)
    parser.add_argument("--root-6s9z", type=Path)
    parser.add_argument("--root-smacv2", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--evaluate-missing", action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=256)
    parser.add_argument("--eval-num-envs", type=int, default=128)
    parser.add_argument(
        "--eval-policy", choices=("deterministic", "stochastic"), default="stochastic"
    )
    parser.add_argument(
        "--gpus", type=lambda raw: tuple(raw.split(",")), default=("0", "1", "2", "3")
    )
    parser.add_argument("--max-runs-per-gpu", type=int, default=2)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260923)
    args = parser.parse_args()
    if args.eval_episodes <= 0 or args.eval_num_envs <= 0:
        parser.error("Evaluation episodes and environments must be positive")
    if not args.gpus or any(not gpu for gpu in args.gpus) or args.max_runs_per_gpu <= 0:
        parser.error("Provide GPUs and positive per-GPU concurrency")
    if args.bootstrap_samples < 100:
        parser.error("Bootstrap samples must be at least 100")
    report(args)


if __name__ == "__main__":
    main()
