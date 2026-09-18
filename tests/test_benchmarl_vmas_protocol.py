import json
from pathlib import Path

import pytest

from experiments.benchmarl_vmas.protocol import (
    CALIBRATION_MINIBATCHES,
    CALIBRATION_PILOT_SEED,
    CALIBRATION_PROTOCOL_VERSION,
    CONDITIONS,
    DEFAULT_SEEDS,
    TASKS,
    load_cka_coefficients,
    matrix,
)


def test_phase_one_matrix_is_exactly_36_nps_runs():
    coefficients = {task: 0.2 + 0.1 * index for index, task in enumerate(TASKS)}
    runs = matrix(cka_coefficients=coefficients)
    assert len(runs) == 36
    assert len({run.name for run in runs}) == 36
    assert {run.task for run in runs} == set(TASKS)
    assert {run.condition for run in runs} == set(CONDITIONS)
    assert {run.seed for run in runs} == set(DEFAULT_SEEDS)
    for task in TASKS:
        for seed in DEFAULT_SEEDS:
            cell = [run for run in runs if run.task == task and run.seed == seed]
            assert [run.condition for run in cell] == list(CONDITIONS)
            assert cell[-1].coefficient == pytest.approx(coefficients[task])


def test_protocol_uses_official_task_names_without_agent_count_overrides():
    assert TASKS == ("discovery", "passage", "football")
    source = (
        Path(__file__).parents[1]
        / "experiments"
        / "benchmarl_vmas"
        / "train_alignment.py"
    ).read_text(encoding="utf-8")
    assert ".config.update(" not in source


def test_direction_and_distance_are_locked():
    runs = matrix(
        seeds=(1,),
        tasks=("discovery",),
        cka_coefficients={"discovery": 0.37},
    )
    assert [(run.align_mode, run.align_distance) for run in runs] == [
        ("none", "ln_mse"),
        ("c_to_a", "ln_mse"),
        ("c_to_a", "linear_cka"),
    ]
    assert [run.coefficient for run in runs] == [0.0, 0.1, 0.37]


def calibration_payload():
    return {
        "protocol_version": CALIBRATION_PROTOCOL_VERSION,
        "selection_uses_return": False,
        "performance_fields_persisted": False,
        "reference_distance": "ln_mse",
        "target_distance": "linear_cka",
        "tasks": list(TASKS),
        "actor_parameterization": "nps",
        "minibatch_sampling": "training_replay_buffer_random",
        "pilot_seed": CALIBRATION_PILOT_SEED,
        "calibration_minibatches": CALIBRATION_MINIBATCHES,
        "task_alignment_coefs": {
            task: 0.2 + 0.1 * index for index, task in enumerate(TASKS)
        },
    }


def test_calibration_artifact_uses_task_specific_coefficients(tmp_path):
    payload = calibration_payload()
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(payload))
    assert load_cka_coefficients(path) == pytest.approx(payload["task_alignment_coefs"])


def test_calibration_artifact_rejects_other_task_sets(tmp_path):
    payload = calibration_payload()
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(payload))
    payload["tasks"] = ["discovery"]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="incompatible"):
        load_cka_coefficients(path)


def test_legacy_global_calibration_is_rejected(tmp_path):
    payload = calibration_payload()
    payload.pop("task_alignment_coefs")
    payload["global_alignment_coef"] = 1.0
    path = tmp_path / "legacy-calibration.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="one coefficient per task"):
        load_cka_coefficients(path)


def test_invalid_cka_coefficient_is_rejected():
    with pytest.raises(ValueError):
        matrix(cka_coefficients={task: 0.0 for task in TASKS})
