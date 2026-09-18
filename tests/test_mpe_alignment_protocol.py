import json

import pytest

from experiments.mpe_alignment.protocol import (
    CALIBRATION_PILOT_SEED,
    CALIBRATION_PROTOCOL_VERSION,
    CONDITIONS,
    DEFAULT_SEEDS,
    TASKS,
    load_cka_coefficient,
    matrix,
)


def test_matrix_is_twelve_nps_runs():
    runs = matrix(cka_coefficient=0.37)
    assert len(runs) == 12
    assert len({run.name for run in runs}) == 12
    assert {run.task for run in runs} == set(TASKS)
    assert {run.condition for run in runs} == set(CONDITIONS)
    assert {run.seed for run in runs} == set(DEFAULT_SEEDS)


def test_direction_distance_and_coefficients_are_locked():
    runs = matrix(seeds=(1,), cka_coefficient=0.37)
    assert [(run.align_mode, run.align_distance) for run in runs] == [
        ("none", "ln_mse"),
        ("c_to_a", "ln_mse"),
        ("c_to_a", "linear_cka"),
    ]
    assert [run.coefficient for run in runs] == pytest.approx([0.0, 0.1, 0.37])


def calibration_payload():
    return {
        "protocol_version": CALIBRATION_PROTOCOL_VERSION,
        "task": TASKS[0],
        "pilot_seed": CALIBRATION_PILOT_SEED,
        "actor_parameterization": "nps",
        "selection_uses_return": False,
        "reference_distance": "ln_mse",
        "target_distance": "linear_cka",
        "alignment_coefficient": 0.37,
    }


def test_calibration_artifact_is_strict(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(calibration_payload()))
    assert load_cka_coefficient(path) == pytest.approx(0.37)
    payload = calibration_payload()
    payload["selection_uses_return"] = True
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="incompatible"):
        load_cka_coefficient(path)


def test_non_positive_cka_coefficient_is_rejected():
    with pytest.raises(ValueError):
        matrix(cka_coefficient=0.0)
