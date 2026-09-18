import json

import pytest

from experiments.benchmarl_vmas.protocol import (
    CALIBRATION_PROTOCOL_VERSION,
    CONDITIONS,
    DEFAULT_SEEDS,
    TASKS,
    load_cka_coefficient,
    matrix,
)


def test_phase_one_matrix_is_exactly_36_nps_runs():
    runs = matrix(cka_coefficient=0.37)
    assert len(runs) == 36
    assert len({run.name for run in runs}) == 36
    assert {run.task for run in runs} == set(TASKS)
    assert {run.condition for run in runs} == set(CONDITIONS)
    assert {run.seed for run in runs} == set(DEFAULT_SEEDS)
    for task in TASKS:
        for seed in DEFAULT_SEEDS:
            cell = [run for run in runs if run.task == task and run.seed == seed]
            assert [run.condition for run in cell] == list(CONDITIONS)


def test_direction_and_distance_are_locked():
    runs = matrix(seeds=(1,), tasks=("discovery_5",), cka_coefficient=0.37)
    assert [(run.align_mode, run.align_distance) for run in runs] == [
        ("none", "ln_mse"),
        ("c_to_a", "ln_mse"),
        ("c_to_a", "linear_cka"),
    ]
    assert [run.coefficient for run in runs] == [0.0, 0.1, 0.37]


def test_calibration_artifact_rejects_other_task_sets(tmp_path):
    payload = {
        "protocol_version": CALIBRATION_PROTOCOL_VERSION,
        "selection_uses_return": False,
        "performance_fields_persisted": False,
        "reference_distance": "ln_mse",
        "target_distance": "linear_cka",
        "tasks": list(TASKS),
        "actor_parameterization": "nps",
        "global_alignment_coef": 0.42,
    }
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(payload))
    assert load_cka_coefficient(path) == pytest.approx(0.42)
    payload["tasks"] = ["discovery_5"]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="incompatible"):
        load_cka_coefficient(path)


def test_invalid_cka_coefficient_is_rejected():
    with pytest.raises(ValueError):
        matrix(cka_coefficient=0.0)
