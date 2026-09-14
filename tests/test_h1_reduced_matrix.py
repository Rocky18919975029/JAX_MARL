from types import SimpleNamespace

import pytest

from scripts import run_h1_smax_confirmatory as launcher


def reduced_args(**overrides):
    values = {
        "matrix_profile": "reduced-nps-4condition",
        "maps": launcher.MAPS,
        "actor_variants": launcher.REDUCED_ACTOR_VARIANTS,
        "conditions": launcher.REDUCED_CONDITIONS,
        "seeds": launcher.REDUCED_SEEDS,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_reduced_matrix_is_exactly_32_named_runs():
    args = reduced_args()
    launcher.validate_matrix_profile(args)
    tasks = launcher.task_matrix(args)

    assert len(tasks) == 32
    assert {task.actor_label for task in tasks} == {"nps"}
    assert {task.sharing for task in tasks} == {False}
    assert {task.condition for task in tasks} == {
        "none",
        "c_to_a",
        "a_to_c",
        "joint",
    }
    assert {task.seed for task in tasks} == {1, 2, 3, 4}
    assert all(task.run_name.startswith("H1-reduced-") for task in tasks)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("maps", ("10m_vs_11m",)),
        ("actor_variants", ("ps",)),
        ("conditions", ("none", "c_to_a", "a_to_c", "reciprocal", "joint")),
        ("seeds", (1, 2, 3)),
    ),
)
def test_reduced_matrix_rejects_scope_changes(field, value):
    with pytest.raises(ValueError, match="locked to exactly 32 runs"):
        launcher.validate_matrix_profile(reduced_args(**{field: value}))
