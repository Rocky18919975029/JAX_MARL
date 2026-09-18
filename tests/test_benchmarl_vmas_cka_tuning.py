import pytest

from experiments.benchmarl_vmas.protocol import TASKS
from experiments.benchmarl_vmas.run_cka_tuning import (
    DEFAULT_MULTIPLIERS,
    candidate_matrix,
)
from experiments.benchmarl_vmas.select_cka_tuning import summarize


def test_default_tuning_matrix_contains_only_cka_candidates():
    coefficients = {task: 0.4 + index for index, task in enumerate(TASKS)}
    candidates = candidate_matrix(
        coefficients,
        TASKS,
        (1, 2),
        DEFAULT_MULTIPLIERS,
    )
    assert len(candidates) == 3 * 2 * 8
    assert len({candidate.name for candidate in candidates}) == len(candidates)
    for candidate in candidates:
        assert candidate.coefficient == pytest.approx(
            coefficients[candidate.task] * candidate.multiplier
        )


def test_selection_summary_remains_task_separated_and_seed_paired():
    rows = []
    for task_index, task in enumerate(TASKS[:2]):
        for multiplier in (0.25, 0.5):
            for seed in (1, 2):
                rows.append(
                    {
                        "task": task,
                        "seed": seed,
                        "cka_multiplier": multiplier,
                        "cka_coefficient": (task_index + 1) * multiplier,
                        "paired_auc_difference": task_index + multiplier + seed / 10,
                        "paired_final_difference": task_index - multiplier + seed / 10,
                    }
                )
    summary = summarize(rows, (1, 2))
    assert len(summary) == 4
    assert {row["task"] for row in summary} == set(TASKS[:2])
    assert all(row["seed_count"] == 2 for row in summary)
