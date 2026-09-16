import json

from scripts.eval_h1_checkpoints import CHECKPOINT_SPECS, discover_tasks


def make_checkpoint(directory):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model.safetensors").touch()


def test_all_checkpoint_discovery_preserves_preregistered_tasks(tmp_path):
    run_dir = tmp_path / "checkpoints" / "project" / "run-id"
    for name, _ in CHECKPOINT_SPECS:
        make_checkpoint(run_dir / name)
    make_checkpoint(run_dir / "step_000001500000")
    make_checkpoint(run_dir / "step_000002500000")
    (run_dir / "initial" / "config.json").write_text(
        json.dumps({"PROTOCOL_VERSION": "h1-v1.0", "SEED": 1}),
        encoding="utf-8",
    )

    preregistered, missing = discover_tasks(tmp_path)
    dense, dense_missing = discover_tasks(tmp_path, include_all=True)

    assert not missing
    assert not dense_missing
    assert len(preregistered) == len(CHECKPOINT_SPECS)
    assert len(dense) == len(CHECKPOINT_SPECS) + 2
    assert {task.checkpoint_dir.name for task in dense} >= {
        "step_000001500000",
        "step_000002500000",
    }
    assert [task.checkpoint_index for task in dense[-2:]] == [8, 9]


def test_discovery_filters_benchmark_before_missing_checkpoint_validation(tmp_path):
    selected = tmp_path / "checkpoints" / "project" / "selected-id"
    unrelated = tmp_path / "checkpoints" / "project" / "unrelated-id"
    for directory, run_name, map_name, sharing in (
        (selected, "SMAXB4-3s5z_vs_3s6z-nps-none-seed1", "3s5z_vs_3s6z", False),
        (unrelated, "SMAXB4-6h_vs_8z-ps-none-seed1", "6h_vs_8z", True),
    ):
        make_checkpoint(directory / "initial")
        (directory / "initial" / "config.json").write_text(
            json.dumps(
                {
                    "PROTOCOL_VERSION": "smax-alignment-benchmark-4seed-v1.0",
                    "MAP_NAME": map_name,
                    "ACTOR_PARAMETER_SHARING": sharing,
                    "SEED": 1,
                }
            ),
            encoding="utf-8",
        )
        (directory / "initial" / "metadata.json").write_text(
            json.dumps({"wandb_run_name": run_name}), encoding="utf-8"
        )
    for name, _ in CHECKPOINT_SPECS[1:]:
        make_checkpoint(selected / name)

    tasks, missing = discover_tasks(
        tmp_path,
        protocol_versions=("smax-alignment-benchmark-4seed-v1.0",),
        run_name_glob="SMAXB4-3s5z_vs_3s6z-nps-*",
        map_names=("3s5z_vs_3s6z",),
        actor_variants=("nps",),
    )

    assert not missing
    assert len(tasks) == len(CHECKPOINT_SPECS)
    assert {task.run_name for task in tasks} == {"SMAXB4-3s5z_vs_3s6z-nps-none-seed1"}
