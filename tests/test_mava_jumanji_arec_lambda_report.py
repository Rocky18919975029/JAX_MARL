"""Checks for the completed same-four-seed ARec lambda-sweep report."""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import pytest


HERE = Path(__file__).resolve().parents[1] / "experiments/mava_jumanji"
sys.path.insert(0, str(HERE))
import report_arec_lambda_sweep as report  # noqa: E402


def fake_metric(task: str, condition: str, seed: int, value: float, stride: int, count: int) -> dict:
    return {
        report.paired_plot.ENV_NAMES[task]: {
            task.split("_", 1)[1]: {
                report.paired_plot.ALGORITHM_NAMES[condition]: {
                    f"seed_{seed}": {
                        f"step_{index}": {
                            "step_count": index * stride,
                            "mean_episode_return": value + index / 1000,
                        }
                        for index in range(1, count + 1)
                    }
                }
            }
        }
    }


def synthetic_sweep(tmp_path: Path) -> Path:
    sweep = report.sweep
    config = sweep.paired.read_json(sweep.paired.CONFIG_PATH)
    source = tmp_path / "original"
    sweep.paired.write_json(source / "experiment_manifest.json", {"source": "synthetic"})
    mava = tmp_path / "Mava"
    baseline = mava / "mava/systems/ppo/anakin/rec_mappo.py"
    baseline.parent.mkdir(parents=True)
    baseline.write_text("# synthetic\n")
    reused = []
    jobs = sweep.paired.make_jobs(config, sweep.TASKS, sweep.SEEDS, False)
    for job in jobs:
        stride = sweep.total_steps(job) // job["shared_overrides"]["arch.num_evaluation"]
        coefficient = None if job["condition"] == "none" else sweep.REUSED_COEFFICIENT
        value = 0.2 if coefficient is None else 0.2 + coefficient * 1000
        metric = source / "runs" / job["name"] / "json" / "metrics.json"
        sweep.paired.write_json(metric, fake_metric(
            job["task"], job["condition"], job["seed"], value,
            stride, job["shared_overrides"]["arch.num_evaluation"],
        ))
        reused.append({
            "task": job["task"], "condition": job["condition"], "seed": job["seed"],
            "coefficient": coefficient, "metric_file": str(metric),
            "metric_sha256": sweep.paired.digest(metric),
        })
    root = tmp_path / "sweep"
    manifest = sweep.build_manifest(config, source, root, mava, reused)
    sweep.paired.write_json(root / "experiment_manifest.json", manifest)
    for group in manifest["new_groups"]:
        child = Path(group["run_root"])
        sweep.paired.write_json(child / "experiment_manifest.json", {
            "jobs": group["jobs"], "conditions": ["arec"],
            "arec": {
                "coef": group["coefficient"], "q_steps": sweep.Q_STEPS,
                "q_lr": sweep.Q_LR, "fisher_ridge": sweep.FISHER_RIDGE,
            },
        })
        for job in group["jobs"]:
            sweep.paired.write_json(sweep.paired.status_path(child, job), {
                "status": "completed", "exit_code": 0,
            })
            stride = sweep.total_steps(job) // job["shared_overrides"]["arch.num_evaluation"]
            metric = child / "runs" / job["name"] / "json" / "metrics.json"
            sweep.paired.write_json(metric, fake_metric(
                job["task"], "arec", job["seed"],
                0.2 + group["coefficient"] * 1000,
                stride, job["shared_overrides"]["arch.num_evaluation"],
            ))
    return root


def test_report_reuses_16_runs_joins_32_new_and_corrects_eval_lag(tmp_path):
    root = synthetic_sweep(tmp_path)
    _, curves, provenance = report.validated_curves(root)
    assert len(curves) == 48
    assert len(provenance["sources"]) == 48
    for task in report.sweep.TASKS:
        for coefficient in (None, *report.sweep.COEFFICIENTS):
            for seed in report.sweep.SEEDS:
                steps = sorted(curves[(task, coefficient, seed)])
                assert len(steps) == 122
                assert steps[0] == 0
                assert steps[1] == provenance["evaluation_lag_steps"][task]


def test_report_metrics_are_paired_and_last_five(tmp_path):
    root = synthetic_sweep(tmp_path)
    _, curves, _ = report.validated_curves(root)
    seed_rows, summary, points = report.aggregate(curves, 500, 19)
    assert len(seed_rows) == 48
    assert len(summary) == 12
    assert len(points) == 2 * 6 * 122
    for task in report.sweep.TASKS:
        selected = [row for row in summary if row["task"] == task and row["selected_by_auc"]]
        assert len(selected) == 1
        assert selected[0]["coefficient"] == report.sweep.COEFFICIENTS[-1]
        baseline = next(row for row in summary if row["task"] == task and row["condition"] == "none")
        assert baseline["paired_auc_delta_mean"] == 0
        assert baseline["paired_final_delta_mean"] == 0
        arec = next(row for row in summary if row["task"] == task and row["coefficient"] == 1e-4)
        assert arec["paired_auc_delta_mean"] == pytest.approx(0.1)
        assert arec["paired_final_delta_mean"] == pytest.approx(0.1)
        assert arec["final_five_mean"] == pytest.approx(0.3 + 0.12)


def test_report_writes_plot_table_and_provenance(tmp_path):
    root = synthetic_sweep(tmp_path)
    output = tmp_path / "report"
    report.report(root, output, 100, 19)
    assert (output / f"{report.STEM}.png").stat().st_size > 1000
    assert (output / f"{report.STEM}.svg").stat().st_size > 1000
    assert (output / f"{report.STEM}.pdf").stat().st_size > 1000
    with (output / "summary.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 12
    with (output / "seed_level.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 48
    assert (output / "provenance.json").is_file()
