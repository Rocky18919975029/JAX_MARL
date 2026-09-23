"""Regression tests for combining completed legacy sweep roots in one report."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.report_smax_arec_best_returns import (
    BASELINE_SWEEP_PROTOCOL,
    PROTOCOL,
    discover_roots,
    make_eval_jobs,
    report_training_curves,
    select_runs,
)


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _record(root: Path, run: dict, reward: float) -> None:
    name = run["run_name"]
    _write(root / "status" / f"{name}.json", {"status": "completed"})
    path = root / "metrics" / f"{name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps({"env_step": step, "returns": reward})
                              for step in (2, 6, 10, 14, 18, 20)) + "\n",
                    encoding="utf-8")


def _arec_root(root: Path, task: str, with_none: bool = True) -> None:
    runs = []
    for seed in (1, 2, 3, 4):
        if with_none:
            run = dict(run_name=f"old-none-{seed}", map_name=task,
                       seed=seed, steps=20, condition="none")
            runs.append(run)
            _record(root, run, 1.0)
        for coef, reward in ((0.1, 1.5), (0.2, 1.2)):
            run = dict(run_name=f"arec-{coef}-{seed}", map_name=task,
                       seed=seed, steps=20, condition="actor_score_recovery",
                       coef=coef, q_steps=4, q_learning_rate=0.001,
                       fisher_ridge=0.001)
            runs.append(run)
            _record(root, run, reward)
    _write(root / "experiment_manifest.json", dict(
        protocol=PROTOCOL, maps=[task], seeds=[1, 2, 3, 4],
        budgets={task: 20}, runs=runs,
    ))


def test_external_tuned_baseline_replaces_old_none(tmp_path: Path) -> None:
    task = "6s9z_vs_6s10z"
    arec = tmp_path / "arec"
    baseline = tmp_path / "baseline"
    _arec_root(arec, task)
    runs = []
    for seed in (1, 2, 3, 4):
        for lr, reward in ((0.001, 1.1), (0.002, 1.3)):
            run = dict(run_name=f"none-{lr}-{seed}", seed=seed, steps=20,
                       learning_rate=lr, update_epochs=4)
            runs.append(run)
            _record(baseline, run, reward)
    _write(baseline / "experiment_manifest.json", dict(
        protocol=BASELINE_SWEEP_PROTOCOL, map_name=task, budget=20,
        seeds=[1, 2, 3, 4], runs=runs,
    ))
    selected = select_runs(arec, task, baseline_root=baseline)
    assert selected["baseline_root"] == baseline
    assert selected["baseline_params"] == (0.002, 4)
    assert selected["best_params"][0] == 0.1
    assert all(run["run_name"].startswith("none-0.002")
               for run in selected["baseline"].values())
    for source_root, group in ((baseline, selected["baseline"]),
                               (arec, selected["best"])):
        for run in group.values():
            parent = source_root / "checkpoints" / "project" / f"{run['run_name']}-abc"
            for step in (4, 8, 12, 16, 20):
                ckpt = parent / ("final" if step == 20 else f"step_{step}")
                ckpt.mkdir(parents=True)
                (ckpt / "model.safetensors").touch()
                _write(ckpt / "metadata.json", {"nominal_env_step": step})
                _write(ckpt / "config.json", dict(
                    MAP_NAME=task, SEED=run["seed"],
                    EXPERIMENT_CONDITION=run["condition"],
                    ACTOR_PARAMETER_SHARING=False,
                ))
    jobs, _ = make_eval_jobs({task: selected}, tmp_path / "report")
    assert len(jobs) == 40
    assert all(baseline in job.checkpoint.parents for job in jobs
               if job.condition == "none")
    assert all(arec in job.checkpoint.parents for job in jobs
               if job.condition == "actor_score_recovery")
    figure = report_training_curves(
        {task: selected}, tmp_path / "training-only",
        SimpleNamespace(bootstrap_samples=100, bootstrap_seed=123),
    )
    assert figure.is_file()
    assert (tmp_path / "training-only" / "training_summary_all_tasks.csv").is_file()
    report = json.loads((tmp_path / "training-only" /
                         "training_report_manifest.json").read_text())
    assert "not held-out" in report["final_metric"]


def test_arec_only_sweep_uses_other_root_none(tmp_path: Path) -> None:
    task = "smacv2_10_units"
    old_root = tmp_path / "old"
    new_root = tmp_path / "new"
    _arec_root(old_root, task)
    _arec_root(new_root, task, with_none=False)
    baseline = select_runs(new_root, task, baseline_root=old_root)
    assert baseline["baseline_root"] == old_root
    assert baseline["baseline_params"] is None
    tuned = tmp_path / "tuned-6s9z"
    tuned_runs = []
    for seed in (1, 2, 3, 4):
        run = dict(run_name=f"tuned-none-{seed}", seed=seed, steps=20,
                   learning_rate=0.001, update_epochs=4)
        tuned_runs.append(run)
        _record(tuned, run, 1.0)
    _write(tuned / "experiment_manifest.json", dict(
        protocol=BASELINE_SWEEP_PROTOCOL, map_name="6s9z_vs_6s10z", budget=20,
        seeds=[1, 2, 3, 4], runs=tuned_runs,
    ))
    discovered = discover_roots(tmp_path)
    assert discovered[0] == tuned
    assert discovered[1] == new_root
    assert discovered[2] == old_root


def test_incomplete_external_baseline_rejected(tmp_path: Path) -> None:
    task = "6s9z_vs_6s10z"
    arec, baseline = tmp_path / "arec", tmp_path / "baseline"
    _arec_root(arec, task)
    _write(baseline / "experiment_manifest.json", dict(
        protocol=BASELINE_SWEEP_PROTOCOL, map_name=task, budget=20,
        seeds=[1, 2, 3, 4], runs=[],
    ))
    with pytest.raises(RuntimeError, match="complete identical seed sets"):
        select_runs(arec, task, baseline_root=baseline)
