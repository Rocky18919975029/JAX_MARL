"""Synthetic marl-eval JSON tests for the live paired-return plot."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


MODULE = Path(__file__).resolve().parents[1] / "experiments/mava_jumanji/plot_matched_optimal_returns.py"
spec = importlib.util.spec_from_file_location("mava_live_return", MODULE)
plot = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(plot)


def _fixture(root: Path, *, incomplete: bool = False) -> None:
    tasks = list(plot.ENV_NAMES)
    jobs = []
    for seed in range(1, 5):
        for task in tasks:
            for condition in plot.CONDITIONS:
                name = f"{task}--{condition}--seed{seed}"
                jobs.append({"name": name, "task": task, "condition": condition, "seed": seed})
                offset = 0.5 if condition == "arec" else 0.0
                entries = {
                    "step_0": {"step_count": 1_000_000, "mean_episode_return": [seed + offset]},
                    "step_1": {"step_count": 2_000_000, "mean_episode_return": [seed + 1 + offset]},
                    "absolute_metrics": {"mean_episode_return": [999.0]},
                }
                if incomplete and seed == 4 and condition == "arec":
                    entries.pop("step_1")
                data = {
                    plot.ENV_NAMES[task]: {
                        task.split("_", 1)[1]: {
                            plot.ALGORITHM_NAMES[condition]: {f"seed_{seed}": entries}
                        }
                    }
                }
                path = root / "runs" / name / "json" / plot.ALGORITHM_NAMES[condition] / "time" / "metrics.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(data))
    (root / "experiment_manifest.json").write_text(json.dumps({
        "protocol": plot.EXPECTED_PROTOCOL,
        "smoke": False,
        "tasks": tasks,
        "seeds": [1, 2, 3, 4],
        "jobs": jobs,
    }))


def test_loader_reads_only_eval_return_and_exact_steps(tmp_path):
    _fixture(tmp_path, incomplete=True)
    manifest, curves, warnings, sources = plot.load_curves(tmp_path)
    assert not warnings
    assert len(sources) == 16
    assert len(manifest["jobs"]) == 16
    assert curves[("lbf_15x15-4p-5f", "none")][1] == {1_000_000: 1.0, 2_000_000: 2.0}
    assert curves[("rware_large-8ag", "arec")][4] == {1_000_000: 4.5}


def test_ci_only_when_all_four_seeds_and_is_deterministic(tmp_path):
    _fixture(tmp_path, incomplete=True)
    manifest, curves, _, _ = plot.load_curves(tmp_path)
    rows = plot.aggregate(curves, manifest["tasks"], manifest["seeds"], 10_000, 7)
    rows_again = plot.aggregate(curves, manifest["tasks"], manifest["seeds"], 10_000, 7)
    assert rows == rows_again
    full = next(row for row in rows if row["task"] == "lbf_15x15-4p-5f"
                and row["condition"] == "arec" and row["env_step"] == 1_000_000)
    partial = next(row for row in rows if row["task"] == "lbf_15x15-4p-5f"
                   and row["condition"] == "arec" and row["env_step"] == 2_000_000)
    assert full["n_seeds"] == 4 and full["seed_ids"] == "1,2,3,4"
    assert full["mean_return"] == 3.0
    assert full["ci_low"] < full["mean_return"] < full["ci_high"]
    assert partial["n_seeds"] == 3 and partial["mean_return"] == 3.5
    assert partial["ci_low"] is None and partial["ci_high"] is None
    assert np.isfinite([row["mean_return"] for row in rows]).all()


def test_live_partial_write_is_skipped_without_splicing_old_attempt(tmp_path):
    _fixture(tmp_path)
    name = "lbf_15x15-4p-5f--none--seed1"
    newer = tmp_path / "runs" / name / "json" / "rec_mappo" / "newer" / "metrics.json"
    newer.parent.mkdir(parents=True)
    newer.write_text('{"incomplete":')
    _, curves, warnings, _ = plot.load_curves(tmp_path)
    assert any(name in warning for warning in warnings)
    assert 1 not in curves[("lbf_15x15-4p-5f", "none")]


def test_render_produces_raster_vector_and_plotted_csv(tmp_path):
    _fixture(tmp_path, incomplete=True)
    manifest, curves, _, _ = plot.load_curves(tmp_path)
    rows = plot.aggregate(curves, manifest["tasks"], manifest["seeds"], 1000, 7)
    stem = tmp_path / "plots" / plot.OUTPUT_STEM
    plot.render(rows, manifest["tasks"], stem)
    plot.write_csv(rows, stem.with_suffix(".csv"))
    assert stem.with_suffix(".png").read_bytes().startswith(b"\x89PNG")
    assert "<svg" in stem.with_suffix(".svg").read_text()
    assert stem.with_suffix(".pdf").read_bytes().startswith(b"%PDF")
    assert "n_seeds" in stem.with_suffix(".csv").read_text()


def test_cli_refresh_writes_live_counts_and_can_run_before_first_eval(tmp_path):
    _fixture(tmp_path)
    for path in (tmp_path / "runs").glob("**/metrics.json"):
        path.unlink()
    result = subprocess.run(
        [sys.executable, str(MODULE), "--run-root", str(tmp_path), "--bootstrap-resamples", "100"],
        capture_output=True, text=True, check=True,
    )
    assert "waiting for evaluations" in result.stdout
    stem = tmp_path / "plots" / plot.OUTPUT_STEM
    assert stem.with_suffix(".png").is_file()
    assert stem.with_suffix(".csv").read_text().count("\n") == 1
    metadata = json.loads(stem.with_suffix(".json").read_text())
    assert metadata["latest"]["lbf_15x15-4p-5f/none"] is None
