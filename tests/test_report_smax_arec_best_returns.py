import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.report_smax_arec_best_returns import (
    TASKS,
    THIRD_TASK,
    bootstrap_indices,
    build_rows,
    evaluate_jobs,
    last_five_checkpoints,
    make_eval_jobs,
    report,
    render_figure,
    select_runs,
)
from scripts.run_smax_actor_score_recovery_sweep import PROTOCOL, run_matrix


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def make_sweep(root: Path, task: str, budget: int) -> None:
    runs = run_matrix(
        (task,),
        (1, 2, 3, 4),
        {task: budget},
        (3e-5, 1e-4),
        (4,),
        (1e-3,),
        (1e-3,),
    )
    write_json(
        root / "experiment_manifest.json",
        {
            "protocol": PROTOCOL,
            "maps": [task],
            "seeds": [1, 2, 3, 4],
            "budgets": {task: budget},
            "runs": [dict(run.__dict__, run_name=run.name) for run in runs],
        },
    )
    for run in runs:
        write_json(root / "status" / f"{run.name}.json", {"status": "completed"})
        gain = 0 if run.condition == "none" else (0.1 if run.coef == 3e-5 else 0.2)
        path = root / "metrics" / f"{run.name}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            for step in (
                budget // 5,
                2 * budget // 5,
                3 * budget // 5,
                4 * budget // 5,
                budget,
            ):
                file.write(
                    json.dumps(
                        {
                            "env_step": step,
                            "returns": step / budget + gain + 0.01 * run.seed,
                        }
                    )
                    + "\n"
                )
        ckpt_root = root / "checkpoints" / "project" / f"{run.name}-runid"
        for index in range(1, 6):
            step = budget * index // 5
            for directory in (
                (f"step_{step:012d}", "final") if index == 5 else (f"step_{step:012d}",)
            ):
                ckpt = ckpt_root / directory
                ckpt.mkdir(parents=True, exist_ok=True)
                (ckpt / "model.safetensors").write_bytes(b"synthetic")
                write_json(ckpt / "metadata.json", {"nominal_env_step": step})
                write_json(
                    ckpt / "config.json",
                    {
                        "MAP_NAME": task,
                        "SEED": run.seed,
                        "EXPERIMENT_CONDITION": run.condition,
                        "ACTOR_PARAMETER_SHARING": False,
                    },
                )


def test_best_selection_and_last_five_real_checkpoint_evaluation(tmp_path):
    pytest.importorskip("matplotlib")
    roots = {task: tmp_path / task for task in TASKS}
    budgets = {TASKS[0]: 10_000_000, TASKS[1]: 20_000_000}
    for task in TASKS:
        make_sweep(roots[task], task, budgets[task])
    selections = {task: select_runs(roots[task], task) for task in TASKS}
    assert all(selected["best_params"][0] == 1e-4 for selected in selections.values())
    assert all(
        selected["paired_return_auc_gain"] == pytest.approx(0.2)
        for selected in selections.values()
    )
    output = tmp_path / "report"
    jobs, steps = make_eval_jobs(selections, output)
    assert len(jobs) == 2 * 2 * 4 * 5
    assert all(len(set(steps[key])) == 5 for key in steps)
    assert all(
        job.checkpoint.name == "final"
        for job in jobs
        if job.nominal_step == budgets[job.task]
    )
    with pytest.raises(
        RuntimeError, match="Missing 80 held-out checkpoint evaluations"
    ):
        evaluate_jobs(
            jobs,
            episodes=256,
            num_envs=128,
            policy="stochastic",
            evaluate_missing=False,
            gpus=("0",),
            max_runs_per_gpu=1,
        )

    for job in jobs:
        gain = 0 if job.condition == "none" else 0.4
        write_json(
            job.output,
            {
                "checkpoint": str(job.checkpoint),
                "episodes": 256,
                "eval_seed": job.eval_seed,
                "policy": "stochastic",
                "map_name": job.task,
                "training_seed": job.seed,
                "checkpoint_nominal_env_step": job.nominal_step,
                "return_mean": job.nominal_step / budgets[job.task]
                + gain
                + job.seed * 0.01,
            },
        )
    summary, seed_rows, curves = build_rows(
        selections,
        jobs,
        steps,
        episodes=256,
        policy="stochastic",
        bootstrap_samples=1000,
        bootstrap_seed=7,
    )
    assert len(summary) == 4
    assert len(seed_rows) == 16
    assert {row["task"] for row in curves} == set(TASKS)
    for task in TASKS:
        baseline = next(
            row for row in summary if row["task"] == task and row["condition"] == "none"
        )
        recovery = next(
            row
            for row in summary
            if row["task"] == task and row["condition"] == "actor_score_recovery"
        )
        assert recovery["delta_return_auc_vs_none_mean"] == pytest.approx(0.2)
        assert recovery[
            "delta_final_train_return_last5_ckpt_vs_none_mean"
        ] == pytest.approx(0.2)
        assert recovery[
            "delta_final_eval_return_last5_ckpt_vs_none_mean"
        ] == pytest.approx(0.4)
        assert baseline["delta_final_eval_return_last5_ckpt_vs_none_mean"] == 0
        assert (
            recovery["final_eval_return_last5_ckpt_mean"]
            > baseline["final_eval_return_last5_ckpt_mean"]
        )
    assert bootstrap_indices(4, 20_000, 7).shape == (256, 4)
    figure = output / "selected-returns"
    render_figure(curves, selections, figure)
    assert all(
        figure.with_suffix(f".{suffix}").is_file() for suffix in ("png", "pdf", "svg")
    )
    report(
        SimpleNamespace(
            root_10m=roots[TASKS[0]],
            root_3s5z=roots[TASKS[1]],
            output_root=output,
            evaluate_missing=False,
            eval_episodes=256,
            eval_num_envs=128,
            eval_policy="stochastic",
            gpus=("0",),
            max_runs_per_gpu=1,
            bootstrap_samples=1000,
            bootstrap_seed=7,
        )
    )
    assert (output / "summary_all_tasks.csv").is_file()
    assert (output / "figure_caption.txt").is_file()
    assert (output / "report_manifest.json").is_file()
    assert all((output / task / "summary.csv").is_file() for task in TASKS)


def test_selection_requires_all_four_seeds_and_final_checkpoint(tmp_path):
    task = TASKS[0]
    make_sweep(tmp_path, task, 10_000_000)
    manifest_path = tmp_path / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["seeds"] = [1, 2]
    write_json(manifest_path, manifest)
    with pytest.raises(RuntimeError, match="four-seed"):
        select_runs(tmp_path, task)
    manifest["seeds"] = [1, 2, 3, 4]
    write_json(manifest_path, manifest)
    selected = select_runs(tmp_path, task)
    run = selected["baseline"][1]
    final = next(
        (tmp_path / "checkpoints").glob(
            f"**/{run['run_name']}-*/final/model.safetensors"
        )
    )
    final.unlink()
    with pytest.raises(RuntimeError, match="one final checkpoint"):
        last_five_checkpoints(tmp_path, run, 10_000_000)


def test_three_task_report_adds_6s9z_without_pooling(tmp_path):
    pytest.importorskip("matplotlib")
    tasks = (*TASKS, THIRD_TASK)
    budgets = {TASKS[0]: 10_000_000, TASKS[1]: 20_000_000, THIRD_TASK: 20_000_000}
    roots = {task: tmp_path / f"sweep-{task}" for task in tasks}
    output = tmp_path / "report"
    for task in tasks:
        make_sweep(roots[task], task, budgets[task])
    selections = {task: select_runs(roots[task], task) for task in tasks}
    jobs, steps = make_eval_jobs(selections, output)
    assert len(jobs) == 3 * 2 * 4 * 5
    assert {key[0] for key in steps} == set(tasks)
    for job in jobs:
        write_json(
            job.output,
            {
                "checkpoint": str(job.checkpoint),
                "episodes": 256,
                "eval_seed": job.eval_seed,
                "policy": "stochastic",
                "map_name": job.task,
                "training_seed": job.seed,
                "checkpoint_nominal_env_step": job.nominal_step,
                "return_mean": job.nominal_step / budgets[job.task]
                + (0.4 if job.condition == "actor_score_recovery" else 0.0),
            },
        )
    figure = report(
        SimpleNamespace(
            root_10m=roots[TASKS[0]],
            root_3s5z=roots[TASKS[1]],
            root_6s9z=roots[THIRD_TASK],
            output_root=output,
            evaluate_missing=False,
            eval_episodes=256,
            eval_num_envs=128,
            eval_policy="stochastic",
            gpus=("0",),
            max_runs_per_gpu=1,
            bootstrap_samples=1000,
            bootstrap_seed=7,
        )
    )
    assert figure.name == "smax-arec-selected-return-curves.png"
    assert all(
        figure.with_suffix(f".{suffix}").is_file() for suffix in ("png", "pdf", "svg")
    )
    for task in tasks:
        assert (output / task / "summary.csv").is_file()
        assert (output / task / "seed_level.csv").is_file()
        assert (output / task / "return_curve.csv").is_file()
    with (output / "summary_all_tasks.csv").open(newline="", encoding="utf-8") as file:
        summary = list(csv.DictReader(file))
    assert len(summary) == 6
    assert {row["task"] for row in summary} == set(tasks)
    manifest = json.loads((output / "report_manifest.json").read_text())
    assert set(manifest["selection"]) == set(tasks)
    assert manifest["tasks_are_never_pooled"] is True
    assert manifest["figure_contract"]["publication_width_mm"] == 178
