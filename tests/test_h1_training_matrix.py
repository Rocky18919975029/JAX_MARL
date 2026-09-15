from types import SimpleNamespace

import pytest

from scripts import run_h1_smax_confirmatory as launcher


def profile_args(profile="nps-ln-mse", **overrides):
    conditions = (
        launcher.CKA_CONDITIONS
        if profile == "nps-linear-cka"
        else launcher.LN_MSE_CONDITIONS
    )
    values = {
        "matrix_profile": profile,
        "maps": launcher.MAPS,
        "actor_variants": ("nps",),
        "conditions": conditions,
        "seeds": launcher.SEEDS,
        "cka_alignment_coef": 0.037,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    ("profile", "expected_runs", "expected_conditions"),
    (
        ("nps-ln-mse", 24, {"none", "a_to_c", "c_to_a"}),
        ("nps-linear-cka", 16, {"a_to_c_cka", "c_to_a_cka"}),
    ),
)
def test_canonical_training_matrices(profile, expected_runs, expected_conditions):
    args = profile_args(profile)
    launcher.validate_matrix_profile(args)
    tasks = launcher.task_matrix(args)
    assert len(tasks) == expected_runs
    assert {task.sharing for task in tasks} == {False}
    assert {task.condition for task in tasks} == expected_conditions
    assert {task.seed for task in tasks} == {1, 2, 3, 4}
    assert all(task.run_name.startswith("H1-nps-") for task in tasks)


def test_canonical_matrix_rejects_scope_changes():
    with pytest.raises(ValueError, match="locked to exactly 24 runs"):
        launcher.validate_matrix_profile(profile_args("nps-ln-mse", seeds=(1, 2, 3)))
