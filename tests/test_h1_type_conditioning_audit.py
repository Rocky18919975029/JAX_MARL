import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from scripts.h1_type_conditioning_metrics import (
    FisherAccumulator,
    LinearCKAAccumulator,
    compute_audit,
)


def make_diagnostic(tmp_path, unit_types):
    rng = np.random.default_rng(41)
    episodes = len(unit_types)
    timesteps = 3
    agents = 1
    dimension = 4
    diagnostic = tmp_path / "diagnostics" / "run" / "step_000000500000"
    diagnostic.mkdir(parents=True)
    active = np.ones((episodes, timesteps), dtype=bool)
    alive = np.ones((episodes, timesteps, agents), dtype=bool)
    type_array = np.broadcast_to(
        np.asarray(unit_types, dtype=np.int32)[:, None, None],
        (episodes, timesteps, agents),
    ).copy()
    score_pattern = np.asarray(
        [[1.0, -0.5, 0.25, 0.0], [0.5, 1.0, -0.5, 0.25], [1.0, 0.0, 0.5, -0.5]],
        dtype=np.float32,
    )
    actor_score = np.broadcast_to(
        score_pattern[None, :, None, :],
        (episodes, timesteps, agents, dimension),
    ).copy()
    signs = np.where(np.asarray(unit_types) == 0, 1.0, -1.0)
    value = np.broadcast_to(signs[:, None, None], (episodes, timesteps, agents)).astype(
        np.float32
    )
    reward = np.zeros_like(value)
    global_done = np.zeros((episodes, timesteps), dtype=bool)
    global_done[:, -1] = True
    arrays = {
        "active": active,
        "alive": alive,
        "diagnostic_episode_id": np.arange(episodes, dtype=np.int32),
        "mc_return": np.zeros_like(value),
        "reward": reward,
        "value": value,
        "global_done": global_done,
        "actor_score": actor_score,
        "actor_latent": rng.normal(
            size=(episodes, timesteps, agents, dimension)
        ).astype(np.float32),
        "critic_latent": rng.normal(
            size=(episodes, timesteps, agents, dimension)
        ).astype(np.float32),
        "state_unit_types": type_array,
    }
    np.savez_compressed(diagnostic / "episodes_0000.npz", **arrays)
    metadata = {
        "shards": [{"path": "episodes_0000.npz", "episodes": episodes}],
        "run_id": "fixture",
        "run_name": "H1-fixture-nps-none-seed1",
        "map_name": "fixture",
        "training_seed": 1,
        "checkpoint_env_step": 500_000,
        "checkpoint_nominal_env_step": 500_000,
        "num_agents": agents,
        "actor_parameter_sharing": False,
        "condition": "none",
        "align_distance": "ln_mse",
        "alignment_coef": 0.1,
        "gamma": 0.0,
        "gae_lambda": 0.95,
        "training_rollout_steps": timesteps,
        "protocol_version": "fixture",
        "git_commit": "fixture",
    }
    (diagnostic / "metadata.json").write_text(json.dumps(metadata))
    return diagnostic


def test_type_conditioning_prevents_opposite_gradient_mismatch_cancellation(tmp_path):
    diagnostic = make_diagnostic(tmp_path, [0, 1] * 4)
    result = compute_audit(diagnostic, fisher_ridge_absolute=0.001)
    assert result["epsilon_lat_slot"] < 1e-20
    assert result["epsilon_lat_slot_type"] > 0.1
    assert result["epsilon_lat_type_minus_slot"] > 0.1
    assert result["observed_unit_types"] == [0, 1]


def test_one_type_conditioning_exactly_matches_slot_pooling(tmp_path):
    diagnostic = make_diagnostic(tmp_path, [0] * 8)
    result = compute_audit(diagnostic, fisher_ridge_absolute=0.001)
    assert np.isclose(
        result["epsilon_lat_slot"], result["epsilon_lat_slot_type"], rtol=1e-12
    )
    assert np.isclose(
        result["linear_cka_distance_slot"],
        result["linear_cka_distance_slot_type"],
        rtol=1e-12,
    )


def test_streaming_sufficient_statistics_equal_single_batch():
    rng = np.random.default_rng(17)
    scores = rng.normal(size=(40, 5))
    reference = rng.normal(size=40)
    critic = rng.normal(size=40)
    actor = rng.normal(size=(40, 6))
    target = rng.normal(size=(40, 7))
    fisher_full = FisherAccumulator()
    fisher_full.update(scores, reference, critic)
    fisher_split = FisherAccumulator()
    cka_full = LinearCKAAccumulator()
    cka_full.update(actor, target)
    cka_split = LinearCKAAccumulator()
    for indices in (slice(0, 13), slice(13, 29), slice(29, None)):
        fisher_split.update(scores[indices], reference[indices], critic[indices])
        cka_split.update(actor[indices], target[indices])
    for key in ("g_reference", "g_critic", "delta", "fisher"):
        assert np.allclose(
            fisher_full.statistics()[key], fisher_split.statistics()[key]
        )
    assert np.isclose(
        cka_full.metrics(1e-8)["distance"],
        cka_split.metrics(1e-8)["distance"],
    )


def test_offline_runner_reuses_baseline_and_writes_tables_and_figures(tmp_path):
    source = make_diagnostic(tmp_path / "source", [0] * 8)
    mse_root = tmp_path / "mse"
    cka_root = tmp_path / "cka"

    def install(root, run_name, condition, distance):
        destination = root / "diagnostics_raw" / run_name / source.name
        shutil.copytree(source, destination)
        metadata_path = destination / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata.update(
            {
                "run_name": run_name,
                "map_name": "10m_vs_11m",
                "condition": condition,
                "align_distance": distance,
            }
        )
        metadata_path.write_text(json.dumps(metadata))

    install(mse_root, "H1-test-none-seed1", "none", "ln_mse")
    install(mse_root, "H1-test-c_to_a-seed1", "c_to_a", "ln_mse")
    install(cka_root, "H1-test-c_to_a_cka-seed1", "c_to_a_cka", "linear_cka")
    output = tmp_path / "audit"
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_h1_type_conditioning_audit.py"
    )
    subprocess.run(
        (
            sys.executable,
            str(script),
            "--mse-root",
            str(mse_root),
            "--cka-root",
            str(cka_root),
            "--output-root",
            str(output),
            "--maps",
            "10m_vs_11m",
            "--seeds",
            "1",
            "--fisher-ridge-absolute",
            "0.001",
            "--workers",
            "2",
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads((output / "audit_report.json").read_text())
    assert report["single_type_consistency"]["status"] == "pass"
    assert report["native_checkpoint_rows"] == 3
    assert report["reused_baseline_rows"] == 1
    checkpoint_table = (
        output / "tables" / "checkpoint_type_conditioning.csv"
    ).read_text()
    assert "reused_ln_mse_none" in checkpoint_table
    assert (output / "figures" / "type-audit-10m_vs_11m-ln_mse.png").is_file()
    assert (output / "figures" / "type-audit-10m_vs_11m-linear_cka.png").is_file()
