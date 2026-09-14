import json
from types import SimpleNamespace

import pytest

from scripts.calibrate_h1_cka import pooled_rms_coefficient
from scripts import run_h1_smax_confirmatory as launcher


def test_full_matrix_replaces_shuffled_controls_with_cka_runs():
    args = SimpleNamespace(
        matrix_profile="confirmatory",
        maps=launcher.MAPS,
        actor_variants=tuple(dict(launcher.ACTOR_VARIANTS)),
        conditions=launcher.CONDITIONS,
        seeds=launcher.CONFIRMATORY_SEEDS,
        cka_alignment_coef=0.037,
    )
    tasks = launcher.task_matrix(args)

    assert len(tasks) == 280
    assert sum(task.align_distance == "ln_mse" for task in tasks) == 200
    assert sum(task.align_distance == "linear_cka" for task in tasks) == 80
    assert not any(task.shuffled for task in tasks)
    assert {
        task.condition for task in tasks if task.align_distance == "linear_cka"
    } == {
        "a_to_c_cka",
        "c_to_a_cka",
    }
    assert all(
        task.alignment_coef == pytest.approx(0.037)
        for task in tasks
        if task.align_distance == "linear_cka"
    )


def test_pooled_rms_calibration_uses_relative_rl_gradient_scale():
    cells = [
        {
            "ln_mse_cross_to_rl_ratio": 2.0,
            "linear_cka_cross_to_rl_ratio": 4.0,
        },
        {
            "ln_mse_cross_to_rl_ratio": 4.0,
            "linear_cka_cross_to_rl_ratio": 8.0,
        },
    ]
    assert pooled_rms_coefficient(cells, 0.1) == pytest.approx(0.05)


def test_calibration_file_must_be_return_independent(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps(
            {
                "selection_uses_return": True,
                "performance_fields_persisted": False,
                "reference_distance": "ln_mse",
                "reference_alignment_coef": 0.1,
                "target_distance": "linear_cka",
                "global_alignment_coef": 0.03,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no return selection"):
        launcher.load_cka_calibration(path)
