import json

from scripts.h1_protocol import MANUAL_REFERENCE
from scripts.run_smax_alignment_benchmark import (
    BENCHMARK_MAPS,
    DEFAULT_MAPS,
    reusable_tasks,
    task_matrix,
)


def test_other_benchmark_maps_have_168_unique_four_seed_runs():
    tasks = task_matrix(
        DEFAULT_MAPS,
        ("ps", "nps"),
        (1, 2, 3, 4),
        0.037,
    )
    assert len(tasks) == 168
    assert len({task.run_name for task in tasks}) == 168
    assert {task.seed for task in tasks} == {1, 2, 3, 4}
    assert {task.actor_label for task in tasks} == {"ps", "nps"}
    none = [task for task in tasks if task.align_mode == "none"]
    assert len(none) == 24
    assert {task.distance_label for task in none} == {"distance_free"}


def test_benchmark_scope_is_explicit_and_distance_free_none_is_not_duplicated():
    assert BENCHMARK_MAPS == (
        "2s3z",
        "3s5z_vs_3s6z",
        "smacv2_10_units",
        "6h_vs_8z",
    )
    assert DEFAULT_MAPS == ("2s3z", "3s5z_vs_3s6z", "6h_vs_8z")
    tasks = task_matrix(DEFAULT_MAPS, ("ps", "nps"), (1, 2, 3, 4), 0.037)
    assert sum(task.align_distance == "linear_cka" for task in tasks) == 72
    assert sum(task.align_distance == "ln_mse" for task in tasks) == 96
    assert all(
        task.alignment_coef == 0.037
        for task in tasks
        if task.align_distance == "linear_cka"
    )
    for mode in ("none", "c_to_a", "a_to_c", "joint"):
        assert any(task.align_mode == mode for task in tasks)


def test_exact_prior_checkpoint_is_reused_but_mismatched_config_is_not(tmp_path):
    tasks = task_matrix(("2s3z",), ("nps",), (1,), 0.037)
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


def test_legacy_checkpoint_without_align_distance_is_reused(tmp_path):
    tasks = task_matrix(("2s3z",), ("nps",), (1,), 0.037)
    target = next(task for task in tasks if task.condition == "c_to_a")
    final = tmp_path / "checkpoints" / "legacy-mse" / "final"
    final.mkdir(parents=True)
    metadata = {
        "map_name": target.map_name,
        "seed": target.seed,
        "actor_parameter_sharing": target.sharing,
        "matched_comparison": True,
        "align_mode": target.align_mode,
        "alignment_coef": target.alignment_coef,
        "condition": target.condition,
    }
    (final / "metadata.json").write_text(json.dumps(metadata))
    (final / "config.json").write_text(json.dumps(MANUAL_REFERENCE))
    (final / "model.safetensors").write_bytes(b"model")

    assert reusable_tasks((tmp_path,), tasks, MANUAL_REFERENCE) == {
        target.key: final
    }


def test_align_distance_can_be_recovered_from_legacy_config_or_condition(tmp_path):
    tasks = task_matrix(("2s3z",), ("nps",), (1,), 0.037)
    target = next(task for task in tasks if task.condition == "a_to_c_cka")
    final = tmp_path / "checkpoints" / "legacy-cka" / "final"
    final.mkdir(parents=True)
    metadata = {
        "map_name": target.map_name,
        "seed": target.seed,
        "actor_parameter_sharing": target.sharing,
        "matched_comparison": True,
        "align_mode": target.align_mode,
        "alignment_coef": target.alignment_coef,
        "condition": target.condition,
    }
    config = dict(MANUAL_REFERENCE)
    config["ALIGN_DISTANCE"] = "linear_cka"
    (final / "metadata.json").write_text(json.dumps(metadata))
    (final / "config.json").write_text(json.dumps(config))
    (final / "model.safetensors").write_bytes(b"model")

    assert reusable_tasks((tmp_path,), tasks, MANUAL_REFERENCE) == {
        target.key: final
    }

    config.pop("ALIGN_DISTANCE")
    (final / "config.json").write_text(json.dumps(config))
    assert reusable_tasks((tmp_path,), tasks, MANUAL_REFERENCE) == {
        target.key: final
    }
