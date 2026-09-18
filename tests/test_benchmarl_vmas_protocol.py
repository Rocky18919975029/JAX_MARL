import ast
import json
from pathlib import Path

import pytest

from experiments.benchmarl_vmas.protocol import (
    CALIBRATION_MINIBATCHES,
    CALIBRATION_PILOT_SEED,
    CALIBRATION_PROTOCOL_VERSION,
    CONDITIONS,
    DEFAULT_SEEDS,
    REFERENCE_MSE_ALIGNMENT_COEF,
    TASKS,
    load_alignment_coefficients,
    load_cka_coefficients,
    matrix,
)
from experiments.benchmarl_vmas.run_matrix import gpu_slots


def test_phase_one_matrix_is_exactly_36_nps_runs():
    coefficients = {
        task: {
            "c_to_a_mse": REFERENCE_MSE_ALIGNMENT_COEF,
            "c_to_a_cka": 0.2 + index,
        }
        for index, task in enumerate(TASKS)
    }
    runs = matrix(alignment_coefficients=coefficients)
    assert len(runs) == 36
    assert len({run.name for run in runs}) == 36
    assert {run.task for run in runs} == set(TASKS)
    assert {run.condition for run in runs} == set(CONDITIONS)
    assert {run.seed for run in runs} == set(DEFAULT_SEEDS)
    for task in TASKS:
        for seed in DEFAULT_SEEDS:
            cell = [run for run in runs if run.task == task and run.seed == seed]
            assert [run.condition for run in cell] == list(CONDITIONS)
            assert cell[1].coefficient == pytest.approx(
                coefficients[task]["c_to_a_mse"]
            )
            assert cell[2].coefficient == pytest.approx(
                coefficients[task]["c_to_a_cka"]
            )


def test_protocol_uses_official_task_names_without_agent_count_overrides():
    assert TASKS == ("discovery", "passage", "football")
    source = (
        Path(__file__).parents[1]
        / "experiments"
        / "benchmarl_vmas"
        / "train_alignment.py"
    ).read_text(encoding="utf-8")
    assert ".config.update(" not in source


def test_custom_experiment_class_is_module_level_for_callback_pickling():
    source = (
        Path(__file__).parents[1]
        / "experiments"
        / "benchmarl_vmas"
        / "train_alignment.py"
    ).read_text(encoding="utf-8")
    module = ast.parse(source)
    top_level_classes = {
        node.name for node in module.body if isinstance(node, ast.ClassDef)
    }
    assert "MatchedNpsExperiment" in top_level_classes


def test_gpu_slots_are_interleaved_before_devices_are_reused():
    assert gpu_slots(("0", "1", "2", "3"), 3) == [
        "0",
        "1",
        "2",
        "3",
        "0",
        "1",
        "2",
        "3",
        "0",
        "1",
        "2",
        "3",
    ]


def test_direction_and_distance_are_locked():
    runs = matrix(
        seeds=(1,),
        tasks=("discovery",),
        alignment_coefficients={
            "discovery": {
                "c_to_a_mse": REFERENCE_MSE_ALIGNMENT_COEF,
                "c_to_a_cka": 0.37,
            }
        },
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
        "reference_alignment_coef": REFERENCE_MSE_ALIGNMENT_COEF,
        "target_distance": "linear_cka",
        "tasks": list(TASKS),
        "actor_parameterization": "nps",
        "minibatch_sampling": "training_replay_buffer_random",
        "pilot_seed": CALIBRATION_PILOT_SEED,
        "calibration_minibatches": CALIBRATION_MINIBATCHES,
        "task_alignment_coefs": {
            task: {
                "c_to_a_mse": REFERENCE_MSE_ALIGNMENT_COEF,
                "c_to_a_cka": 0.3 + 0.1 * index,
            }
            for index, task in enumerate(TASKS)
        },
    }


def test_calibration_artifact_uses_task_specific_coefficients(tmp_path):
    payload = calibration_payload()
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(payload))
    assert load_alignment_coefficients(path) == payload["task_alignment_coefs"]
    assert load_cka_coefficients(path) == pytest.approx(
        {
            task: values["c_to_a_cka"]
            for task, values in payload["task_alignment_coefs"].items()
        }
    )


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
    with pytest.raises(ValueError, match="every task"):
        load_cka_coefficients(path)


def test_invalid_alignment_coefficient_is_rejected():
    with pytest.raises(ValueError):
        matrix(
            alignment_coefficients={
                task: {"c_to_a_mse": 0.1, "c_to_a_cka": 0.0}
                for task in TASKS
            }
        )


def test_mse_reference_coefficient_cannot_drift():
    with pytest.raises(ValueError, match="must remain"):
        matrix(
            seeds=(1,),
            tasks=("discovery",),
            alignment_coefficients={
                "discovery": {"c_to_a_mse": 0.2, "c_to_a_cka": 0.3}
            },
        )
