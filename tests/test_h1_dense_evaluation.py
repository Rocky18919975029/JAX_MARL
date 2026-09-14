import json

import numpy as np

from scripts.analyze_h1_performance import seed_endpoints
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


def test_dense_points_do_not_change_preregistered_auc():
    records = []
    for step in (
        0,
        500_000,
        1_000_000,
        2_000_000,
        4_000_000,
        6_000_000,
        8_000_000,
        10_000_000,
    ):
        records.append(
            {
                "task": "10m_vs_11m",
                "actor_parameterization": "nps",
                "condition": "none",
                "seed": 1,
                "nominal_step": step,
                "checkpoint": "final" if step == 10_000_000 else f"step_{step}",
                "return_mean": step / 10_000_000,
                "win_rate": step / 10_000_000,
                "preregistered_checkpoint": True,
            }
        )
    records.append(
        {
            **records[2],
            "nominal_step": 1_500_000,
            "checkpoint": "step_000001500000",
            "return_mean": 999.0,
            "win_rate": 999.0,
            "preregistered_checkpoint": False,
        }
    )

    endpoint = seed_endpoints(records)[0]

    assert np.isclose(endpoint["return_auc"], 0.5)
    assert np.isclose(endpoint["win_rate_auc"], 0.5)
    assert endpoint["num_checkpoints"] == 8
