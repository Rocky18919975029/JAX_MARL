import json

from scripts.h1_protocol import MANUAL_REFERENCE
from scripts.run_smax_alignment_10seed import reusable_tasks, task_matrix


def test_extension_adds_six_seeds_without_duplicate_none_runs():
    tasks = task_matrix(
        ("10m_vs_11m", "smacv2_10_units"),
        ("ps", "nps"),
        (5, 6, 7, 8, 9, 10),
        0.037,
    )
    assert len(tasks) == 168
    assert len({task.run_name for task in tasks}) == 168
    assert {task.seed for task in tasks} == {5, 6, 7, 8, 9, 10}
    assert {task.actor_label for task in tasks} == {"ps", "nps"}
    none = [task for task in tasks if task.align_mode == "none"]
    assert len(none) == 24
    assert {task.distance_label for task in none} == {"distance_free"}


def test_full_ten_seed_matrix_has_280_unique_training_runs():
    tasks = task_matrix(
        ("10m_vs_11m", "smacv2_10_units"),
        ("ps", "nps"),
        tuple(range(1, 11)),
        0.037,
    )
    assert len(tasks) == 280
    assert sum(task.align_distance == "linear_cka" for task in tasks) == 120
    assert sum(task.align_distance == "ln_mse" for task in tasks) == 160
    assert all(
        task.alignment_coef == 0.037
        for task in tasks
        if task.align_distance == "linear_cka"
    )
    for mode in ("none", "c_to_a", "a_to_c", "joint"):
        assert any(task.align_mode == mode for task in tasks)


def test_exact_prior_checkpoint_is_reused_but_mismatched_config_is_not(tmp_path):
    tasks = task_matrix(("10m_vs_11m",), ("nps",), (1,), 0.037)
    target = next(task for task in tasks if task.condition == "a_to_c_cka")
    final = tmp_path / "checkpoints" / "run" / "final"
    final.mkdir(parents=True)
    metadata = {
        "map_name": target.map_name,
        "seed": target.seed,
        "actor_parameter_sharing": target.sharing,
        "matched_comparison": True,
        "align_mode": target.align_mode,
        "align_distance": target.align_distance,
        "alignment_coef": target.alignment_coef,
    }
    (final / "metadata.json").write_text(json.dumps(metadata))
    (final / "config.json").write_text(json.dumps(MANUAL_REFERENCE))
    (final / "model.safetensors").write_bytes(b"model")
    assert reusable_tasks((tmp_path,), tasks, MANUAL_REFERENCE) == {
        target.key: final
    }

    wrong = dict(MANUAL_REFERENCE)
    wrong["NUM_ENVS"] = 64
    (final / "config.json").write_text(json.dumps(wrong))
    assert reusable_tasks((tmp_path,), tasks, MANUAL_REFERENCE) == {}
