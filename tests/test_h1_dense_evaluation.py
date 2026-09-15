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
