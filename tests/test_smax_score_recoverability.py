import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.run_smax_score_recoverability import (
    common_sample_counts,
    select_matched_checkpoints,
)
from scripts.smax_score_recoverability import (
    available_samples_by_agent,
    episode_split_masks,
    fisher_whiten,
    prepare_probe_data,
)


def synthetic_arrays(episodes=20, timesteps=8, agents=3, latent_dim=5, action_dim=4):
    rng = np.random.default_rng(5)
    return {
        "active": np.ones((episodes, timesteps), dtype=bool),
        "alive": np.ones((episodes, timesteps, agents), dtype=bool),
        "available_actions": np.ones(
            (episodes, timesteps, agents, action_dim), dtype=np.float32
        ),
        "action": rng.integers(
            0, action_dim, size=(episodes, timesteps, agents), dtype=np.int32
        ),
        "actor_latent": rng.normal(
            size=(episodes, timesteps, agents, latent_dim)
        ).astype(np.float32),
        "critic_latent": rng.normal(
            size=(episodes, timesteps, agents, latent_dim)
        ).astype(np.float32),
        "actor_score": rng.normal(
            size=(episodes, timesteps, agents, latent_dim)
        ).astype(np.float32),
        "reward": rng.normal(size=(episodes, timesteps, agents)).astype(np.float32),
        "diagnostic_episode_id": np.arange(episodes),
    }


def test_three_episode_splits_are_deterministic_disjoint_and_complete():
    first = episode_split_masks(20, 0.70, 0.15, 17)
    second = episode_split_masks(20, 0.70, 0.15, 17)
    assert all(np.array_equal(first[name], second[name]) for name in first)
    assert [int(first[name].sum()) for name in ("fit", "validation", "test")] == [
        14,
        3,
        3,
    ]
    assert np.all(sum(first.values()) == 1)


def test_fisher_whitening_uses_only_fit_scores():
    fit = np.asarray(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 2.0], [0.0, -2.0]],
        dtype=np.float32,
    )
    validation = np.asarray([[1.0, 2.0]], dtype=np.float32)
    test = np.asarray([[2.0, 1.0]], dtype=np.float32)
    fit_u, validation_u, test_u, eigenvalues = fisher_whiten(
        fit, validation, test, ridge=0.5
    )
    assert np.allclose(eigenvalues, [0.5, 2.0])
    assert np.allclose(validation_u, [[1.0, 2.0 / np.sqrt(2.5)]])
    assert np.allclose(test_u, [[2.0, 1.0 / np.sqrt(2.5)]])
    assert np.allclose(
        fit_u.T @ fit_u / len(fit_u),
        np.diag(eigenvalues / (eigenvalues + 0.5)),
    )


def test_probe_preparation_never_mixes_episode_splits():
    arrays = synthetic_arrays()
    prepared = prepare_probe_data(
        arrays,
        fit_fraction=0.70,
        validation_fraction=0.15,
        split_seed=11,
        sampling_seed=12,
        fit_samples_per_agent=32,
        validation_samples_per_agent=12,
        test_samples_per_agent=12,
        fisher_ridge=1e-3,
    )
    for name, count in (("fit", 32), ("validation", 12), ("test", 12)):
        assert prepared[f"{name}_x"].shape == (3, count, 9)
        assert prepared[f"{name}_y"].shape == (3, count, 5)
    for agent in range(3):
        used = prepared["selected_episodes"]
        assert used["fit"][agent].isdisjoint(used["validation"][agent])
        assert used["fit"][agent].isdisjoint(used["test"][agent])
        assert used["validation"][agent].isdisjoint(used["test"][agent])


def test_invalid_stored_action_is_rejected():
    arrays = synthetic_arrays()
    arrays["available_actions"][..., 0] = 0
    arrays["action"][...] = 0
    with pytest.raises(RuntimeError, match="invalid under the stored SMAX mask"):
        prepare_probe_data(
            arrays,
            fit_fraction=0.70,
            validation_fraction=0.15,
            split_seed=1,
            sampling_seed=2,
            fit_samples_per_agent=10,
            validation_samples_per_agent=5,
            test_samples_per_agent=5,
            fisher_ridge=1e-3,
        )


def test_availability_counts_only_active_alive_steps():
    arrays = synthetic_arrays(episodes=20, timesteps=3, agents=2)
    arrays["active"][:, -1] = False
    arrays["alive"][:, 0, 1] = False
    counts = available_samples_by_agent(arrays, 0.70, 0.15, 1)
    assert counts["fit"].tolist() == [28, 14]
    assert counts["validation"].tolist() == [6, 3]
    assert counts["test"].tolist() == [6, 3]


def test_checkpoint_plan_uses_only_common_saved_steps(tmp_path):
    sources = []
    for name, extra in (("none", (3_000_000,)), ("cka", (4_000_000,))):
        run = tmp_path / name
        for step in (2_000_000, 5_000_000, 8_000_000, 10_000_000, *extra):
            directory = run / ("final" if step == 10_000_000 else f"step_{step:012d}")
            directory.mkdir(parents=True)
            (directory / "model.safetensors").write_bytes(b"model")
            (directory / "metadata.json").write_text(
                json.dumps({"nominal_env_step": step, "is_initial": False})
            )
        sources.append(
            SimpleNamespace(
                task="10m_vs_11m",
                budget=10_000_000,
                checkpoint=run / "final",
                key=name,
            )
        )
    plan, paths = select_matched_checkpoints(sources, (0.25, 0.5, 0.75, 1.0))
    assert [row["env_step"] for row in plan["10m_vs_11m"]] == [
        2_000_000,
        5_000_000,
        8_000_000,
        10_000_000,
    ]
    assert paths[("none", 10_000_000)].name == "final"


def test_launcher_dry_run_expands_three_conditions_to_four_steps(tmp_path):
    matrix = tmp_path / "matrix"
    for condition, distance in (
        ("none", "ln_mse"),
        ("c_to_a_mse", "ln_mse"),
        ("c_to_a_cka", "linear_cka"),
    ):
        run = matrix / "checkpoints" / condition
        for step in (2_000_000, 5_000_000, 8_000_000, 10_000_000):
            checkpoint = run / ("final" if step == 10_000_000 else f"step_{step:012d}")
            checkpoint.mkdir(parents=True)
            (checkpoint / "model.safetensors").write_bytes(b"model")
            metadata = {
                "nominal_env_step": step,
                "is_initial": False,
                "map_name": "10m_vs_11m",
                "seed": 1,
                "total_timesteps": 10_000_000,
                "actor_parameter_sharing": False,
                "align_mode": "none" if condition == "none" else "c_to_a",
                "align_distance": distance,
                "alignment_coef": 0 if condition == "none" else 0.1,
                "wandb_project": "test",
                "wandb_run_id": condition,
                "wandb_run_name": condition,
            }
            (checkpoint / "metadata.json").write_text(json.dumps(metadata))
            if step == 10_000_000:
                (checkpoint / "config.json").write_text(
                    json.dumps(
                        {
                            "MAP_NAME": "10m_vs_11m",
                            "SEED": 1,
                            "TOTAL_TIMESTEPS": 10_000_000,
                            "ACTOR_PARAMETER_SHARING": False,
                            "ALIGN_MODE": metadata["align_mode"],
                            "ALIGN_DISTANCE": distance,
                            "ALIGNMENT_COEF": metadata["alignment_coef"],
                        }
                    )
                )
    output = tmp_path / "measurement"
    script = (
        Path(__file__).resolve().parents[1] / "scripts/run_smax_score_recoverability.py"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--matrix-root",
            str(matrix),
            "--run-root",
            str(output),
            "--tasks",
            "10m_vs_11m",
            "--seeds",
            "1",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    protocol = json.loads((output / "protocol.json").read_text())
    assert len(protocol["sources"]) == 12
    assert [row["env_step"] for row in protocol["checkpoint_plan"]["10m_vs_11m"]] == [
        2_000_000,
        5_000_000,
        8_000_000,
        10_000_000,
    ]


def test_common_sample_count_is_task_and_checkpoint_matched(tmp_path):
    jobs = []
    for condition, alive_count in (("none", 4), ("c_to_a_cka", 3)):
        checkpoint = tmp_path / condition / "step_000000000100"
        output = tmp_path / "runs" / condition
        collected = output / "collected"
        checkpoint.mkdir(parents=True)
        collected.mkdir(parents=True)
        (checkpoint / "metadata.json").write_text(json.dumps({"nominal_env_step": 100}))
        active = np.ones((100, 5), dtype=bool)
        alive = np.ones((100, 5, 2), dtype=bool)
        alive[:, alive_count:, 1] = False
        reward = np.zeros((100, 5, 2), dtype=np.float32)
        np.savez_compressed(
            collected / "episodes_0000.npz",
            active=active,
            alive=alive,
            reward=reward,
        )
        (collected / "metadata.json").write_text(
            json.dumps(
                {
                    "checkpoint": str(checkpoint),
                    "checkpoint_nominal_env_step": 100,
                    "episodes": 100,
                    "shards": [{"path": "episodes_0000.npz", "episodes": 100}],
                }
            )
        )
        jobs.append(
            (condition, SimpleNamespace(task="test_map", checkpoint=checkpoint), output)
        )
    counts = common_sample_counts(
        jobs,
        fit_fraction=0.70,
        validation_fraction=0.15,
        split_seed=7,
        fit_cap=300,
        validation_cap=100,
        test_cap=100,
    )
    chosen = counts["test_map"]["100"]
    assert chosen["fit_samples_per_agent"] == 210
    assert chosen["validation_samples_per_agent"] >= 32
    assert chosen["test_samples_per_agent"] >= 32
    assert len(chosen["census"]) == 2


def test_residual_probe_selects_on_validation_not_test():
    pytest.importorskip("jax")
    pytest.importorskip("optax")
    from scripts.smax_score_recoverability import fit_independent_probes

    rng = np.random.default_rng(2)
    x = rng.normal(size=(2, 64, 6)).astype(np.float32)
    y = (x[..., :3] * 0.3).astype(np.float32)
    kwargs = dict(
        hidden_dim=16,
        residual_blocks=3,
        steps=20,
        batch_size=16,
        learning_rate=1e-3,
        validation_interval=5,
        patience_evaluations=3,
        seed=9,
    )
    first = fit_independent_probes(
        x, y, x[:, :16], y[:, :16], x[:, 16:32], y[:, 16:32], **kwargs
    )
    changed_test = fit_independent_probes(
        x, y, x[:, :16], y[:, :16], x[:, 16:32], y[:, 16:32] + 100, **kwargs
    )
    assert first["best_step"].tolist() == changed_test["best_step"].tolist()
    assert np.all(first["best_step"] > 0)
    assert np.all(first["best_step"] <= 20)
    assert not np.allclose(first["test_normalized"], changed_test["test_normalized"])


def test_measurement_writes_three_split_audit_and_unclipped_test(tmp_path):
    pytest.importorskip("jax")
    pytest.importorskip("optax")
    from scripts.smax_score_recoverability import measure_score_recoverability

    collected = tmp_path / "collected"
    collected.mkdir()
    arrays = synthetic_arrays(episodes=20, timesteps=8, agents=2)
    np.savez_compressed(collected / "episodes_0000.npz", **arrays)
    (collected / "metadata.json").write_text(
        json.dumps(
            {
                "array_profile": "score_recoverability",
                "actor_parameter_sharing": False,
                "map_name": "10m_vs_11m",
                "condition": "c_to_a",
                "align_distance": "linear_cka",
                "training_seed": 1,
                "checkpoint": "/some/frozen/checkpoint",
                "checkpoint_nominal_env_step": 5_000_000,
                "diagnostic_seed": 23,
                "episodes": 20,
                "shards": [{"path": "episodes_0000.npz", "episodes": 20}],
            }
        )
    )
    output = tmp_path / "result"
    rows, summary = measure_score_recoverability(
        collected,
        output,
        fit_fraction=0.70,
        validation_fraction=0.15,
        split_seed=11,
        sampling_seed=12,
        fit_samples_per_agent=32,
        validation_samples_per_agent=12,
        test_samples_per_agent=12,
        fisher_ridge=1e-3,
        probe_hidden_dim=16,
        probe_residual_blocks=3,
        probe_steps=10,
        probe_batch_size=16,
        probe_learning_rate=1e-3,
        probe_validation_interval=5,
        probe_patience_evaluations=3,
        probe_seed=9,
    )
    assert summary["protocol"] == "smax-score-recoverability-resmlp-v2.0"
    assert summary["checkpoint_env_step"] == 5_000_000
    assert summary["split_episodes"] == {"fit": 14, "validation": 3, "test": 3}
    assert summary["probe_best_steps_per_agent"] == [
        row["probe_best_step"] for row in rows
    ]
    assert summary["probe_test_used_for_selection"] is False
    assert np.isclose(
        summary["on_policy_episode_return_mean"],
        arrays["reward"][:, :, 0].sum(axis=1).mean(),
    )
    assert (output / "agent_metrics.csv").is_file()
    assert (output / "summary.json").is_file()
