import json
from types import SimpleNamespace

import pytest

from scripts.calibrate_h1_cka import pooled_rms_coefficient
from scripts import run_h1_smax_confirmatory as launcher


def reduced_cka_args(**overrides):
    values = {
        "matrix_profile": "nps-linear-cka",
        "maps": launcher.MAPS,
        "actor_variants": ("nps",),
        "conditions": launcher.CKA_CONDITIONS,
        "seeds": launcher.SEEDS,
        "cka_alignment_coef": 0.037,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_reduced_cka_matrix_exactly_pairs_prior_nps_seeds():
    args = reduced_cka_args()
    launcher.validate_matrix_profile(args)
    tasks = launcher.task_matrix(args)

    assert len(tasks) == 16
    assert {task.map_name for task in tasks} == set(launcher.MAPS)
    assert {task.actor_label for task in tasks} == {"nps"}
    assert {task.sharing for task in tasks} == {False}
    assert {task.condition for task in tasks} == set(launcher.CKA_CONDITIONS)
    assert {task.seed for task in tasks} == {1, 2, 3, 4}
    assert all(task.align_distance == "linear_cka" for task in tasks)
    assert all(task.run_name.startswith("H1-nps-") for task in tasks)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("maps", ("10m_vs_11m",)),
        ("actor_variants", ("ps",)),
        ("conditions", ("c_to_a_cka",)),
        ("seeds", (1, 2, 3)),
    ),
)
def test_reduced_cka_matrix_rejects_unpaired_scope(field, value):
    with pytest.raises(ValueError, match="locked to exactly 16 runs"):
        launcher.validate_matrix_profile(reduced_cka_args(**{field: value}))


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
