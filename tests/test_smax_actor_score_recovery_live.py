import json

import pytest

from scripts.analyze_smax_actor_score_recovery_sweep import PROTOCOL
from scripts.plot_smax_actor_score_recovery_live import (
    make_snapshot,
    read_available_series,
    snapshot_curves,
)
from scripts.run_smax_actor_score_recovery_sweep import run_matrix


def _live_root(tmp_path):
    runs = run_matrix(
        ("10m_vs_11m",),
        (1, 2, 3),
        {"10m_vs_11m": 100},
        (1e-5, 3e-5),
        (2, 4),
        (1e-3,),
        (1e-3,),
    )
    (tmp_path / "experiment_manifest.json").write_text(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "maps": ["10m_vs_11m"],
                "seeds": [1, 2, 3],
                "budgets": {"10m_vs_11m": 100},
                "runs": [dict(run.__dict__, run_name=run.name) for run in runs],
            }
        )
    )
    (tmp_path / "status").mkdir()
    (tmp_path / "metrics").mkdir()
    return runs


def _write_run(root, run, values, status="running", partial=False):
    (root / "status" / f"{run.name}.json").write_text(json.dumps({"status": status}))
    text = "".join(
        json.dumps({"env_step": step, "win_rate": value}) + "\n"
        for step, value in values
    )
    if partial:
        text += '{"env_step": 75, "win_rate":'
    (root / "metrics" / f"{run.name}.jsonl").write_text(text)


def test_live_snapshot_keeps_partial_seed_counts_and_refreshes(tmp_path):
    runs = _live_root(tmp_path)
    baseline = [run for run in runs if run.condition == "none"]
    for run, value in zip(baseline, (0.1, 0.2, 0.3)):
        _write_run(tmp_path, run, [(25, value), (50, value + 0.1)])
    target = [
        run
        for run in runs
        if run.condition == "actor_score_recovery"
        and run.coef == 1e-5
        and run.q_steps == 2
    ]
    _write_run(tmp_path, target[0], [(25, 0.4), (50, 0.5)], partial=True)
    _write_run(tmp_path, target[1], [(25, 0.6), (50, 0.7)])
    _write_run(tmp_path, target[2], [(25, 0.8)])  # Not enough for one curve.
    single = next(
        run
        for run in runs
        if run.condition == "actor_score_recovery"
        and run.coef == 3e-5
        and run.q_steps == 4
        and run.seed == 1
    )
    _write_run(tmp_path, single, [(25, 0.7), (50, 0.8)])
    failed = next(
        run
        for run in runs
        if run.condition == "actor_score_recovery"
        and run.coef == 3e-5
        and run.q_steps == 2
        and run.seed == 1
    )
    _write_run(tmp_path, failed, [(25, 0.9), (50, 0.9)], status="failed")

    _, rows, coverage = snapshot_curves(
        tmp_path, bootstrap_samples=1000, bootstrap_seed=7
    )
    baseline_row = next(row for row in rows if row["condition"] == "none")
    two_row = next(
        row
        for row in rows
        if row["condition"] != "none"
        and row["coef"] == 1e-5
        and row["q_steps"] == 2
        and row["env_step"] == 25
    )
    one_row = next(
        row
        for row in rows
        if row["condition"] != "none" and row["coef"] == 3e-5 and row["q_steps"] == 4
    )
    assert baseline_row["n_seeds"] == 3
    assert two_row["n_seeds"] == 2
    assert two_row["mean_win_rate"] == pytest.approx(0.5)
    assert two_row["ci95_low"] == pytest.approx(0.4)
    assert two_row["ci95_high"] == pytest.approx(0.6)
    assert one_row["n_seeds"] == 1
    assert one_row["ci95_low"] is None
    assert not any(row["coef"] == 3e-5 and row["q_steps"] == 2 for row in rows)
    assert (
        next(row for row in coverage if row["run_name"] == failed.name)["included"]
        is False
    )

    outputs = make_snapshot(tmp_path, bootstrap_samples=1000, bootstrap_seed=7)
    image = outputs[0]
    assert image.is_file()
    assert image.with_suffix(".svg").is_file()
    assert image.with_suffix(".pdf").is_file()
    csv_path = image.with_name(image.stem + "-curves.csv")
    assert csv_path.is_file()
    before = csv_path.read_text()
    _write_run(tmp_path, target[2], [(25, 0.8), (50, 0.9)])
    make_snapshot(tmp_path, bootstrap_samples=1000, bootstrap_seed=7)
    assert csv_path.read_text() != before
    assert "1,2,3" in csv_path.read_text()


def test_live_reader_ignores_unfinished_final_line(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"env_step": 25, "win_rate": 0.3}\n{"env_step": 50,')
    assert read_available_series(path, "win_rate", 100) == {25: 0.3}


def test_live_snapshot_before_launcher_has_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Sweep has not started"):
        snapshot_curves(tmp_path)
