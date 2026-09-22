"""Score-recovery target and launcher protocol checks."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.run_smax_score_recovery_training import Run, command
from scripts.smax_score_recoverability import canonical_condition


def test_launcher_selects_critic_only_condition(tmp_path):
    args = SimpleNamespace(
        fisher_ridge=1e-3,
        update_epochs=4,
        learning_rate=0.002,
        num_envs=128,
        num_minibatches=4,
        checkpoint_interval=1_000_000,
        wandb_mode="disabled",
        project="test",
    )
    run = Run("3s5z_vs_3s6z", 2, 20_000_000, 0.1)
    parts = command(Path("/repo"), tmp_path, args, run)
    assert "ACTOR_PARAMETER_SHARING=false" in parts
    assert "MATCHED_COMPARISON=true" in parts
    assert "ALIGN_MODE=none" in parts
    assert "ALIGNMENT_COEF=0" in parts
    assert "ORACLE_LATENT_DISTORTION=false" in parts
    assert "SCORE_RECOVERY=true" in parts
    assert "SCORE_RECOVERY_COEF=0.1" in parts
    assert "TOTAL_TIMESTEPS=20000000" in parts
    assert "EXPERIMENT_CONDITION=score_recovery" in parts
    assert "lam0p1-seed2" in run.name


def test_offline_measurement_recognizes_new_condition():
    assert (
        canonical_condition(
            {"condition": "score_recovery", "align_distance": "ln_mse"}
        )
        == "score_recovery"
    )


def test_fisher_target_is_masked_agentwise_and_stop_gradient():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from baselines.MAPPO.score_recovery_target import whiten_rollout_scores

    score = jnp.asarray(
        [
            [[[1.0, 0.0]], [[-1.0, 0.0]], [[100.0, 100.0]]],
            [[[0.0, 2.0]], [[0.0, -2.0]], [[100.0, 100.0]]],
        ]
    )
    valid = jnp.asarray([[[True], [True], [False]]] * 2)
    target, audit = whiten_rollout_scores(score, valid, ridge=0.5)
    np.testing.assert_allclose(np.asarray(target[:, 2]), 0)
    np.testing.assert_allclose(
        np.asarray(target[0, :2, 0, 0]),
        [1 / np.sqrt(1.5), -1 / np.sqrt(1.5)],
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(target[1, :2, 0, 1]),
        [2 / np.sqrt(4.5), -2 / np.sqrt(4.5)],
        atol=1e-5,
    )
    np.testing.assert_array_equal(np.asarray(audit["valid_count_per_agent"]), [2, 2])
    gradient = jax.grad(
        lambda sample: whiten_rollout_scores(sample, valid, ridge=0.5)[0].sum()
    )(score)
    np.testing.assert_allclose(np.asarray(gradient), 0)


def test_recovery_head_gradient_reaches_critic_encoder_not_value_head():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    flax_core = pytest.importorskip("flax.core")
    pytest.importorskip("distrax")
    pytest.importorskip("hydra")
    pytest.importorskip("optax")
    from baselines.MAPPO.mappo_rnn_smax import (
        CriticRNN,
        ScannedRNN,
        ScoreRecoveryHead,
    )

    critic = CriticRNN(config={"FC_DIM_SIZE": 8, "GRU_HIDDEN_DIM": 8})
    head = ScoreRecoveryHead(action_dim=4, latent_dim=8)
    hidden = ScannedRNN.initialize_carry(4, 8)
    state = jnp.ones((2, 4, 12))
    dones = jnp.zeros((2, 4), dtype=bool)
    actions = jnp.zeros((2, 4), dtype=jnp.int32)
    critic_params = critic.init(jax.random.PRNGKey(7), hidden, (state, dones))
    head_params = head.init(jax.random.PRNGKey(8), jnp.ones((2, 4, 8)), actions)
    combined = flax_core.unfreeze(critic_params)
    combined["params"]["ScoreRecoveryHead_0"] = flax_core.unfreeze(head_params)[
        "params"
    ]
    combined = flax_core.freeze(combined)

    def recovery_only(params):
        _, _, critic_latent = critic.apply(params, hidden, (state, dones))
        prediction = head.apply(
            {"params": params["params"]["ScoreRecoveryHead_0"]},
            critic_latent,
            actions,
        )
        return jnp.square(prediction - 1.0).sum()

    gradients = jax.grad(recovery_only)(combined)["params"]
    assert float(jnp.linalg.norm(gradients["Dense_0"]["kernel"])) > 0
    assert (
        float(jnp.linalg.norm(gradients["ScoreRecoveryHead_0"]["Dense_1"]["kernel"]))
        > 0
    )
    np.testing.assert_allclose(np.asarray(gradients["Dense_2"]["kernel"]), 0)


def test_analytic_score_matches_masked_log_probability_gradient():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("distrax")
    pytest.importorskip("hydra")
    pytest.importorskip("optax")
    from baselines.MAPPO.mappo_rnn_smax import categorical_latent_score

    latent = jnp.array([0.3, -0.2, 0.5, 0.1])
    hidden = {
        "kernel": jnp.array(
            [[0.5, -0.4, 0.1], [0.2, 0.3, -0.5], [0.1, 0.6, 0.2], [-0.2, 0.1, 0.7]]
        ),
        "bias": jnp.array([0.4, 0.5, 0.1]),
    }
    logits = {
        "kernel": jnp.array([[0.2, -0.1, 0.6], [-0.3, 0.4, 0.1], [0.5, 0.2, -0.2]]),
        "bias": jnp.array([0.1, -0.2, 0.3]),
    }
    available = jnp.array([1.0, 0.0, 1.0])
    action = jnp.asarray(2)

    def log_probability(z):
        h = jax.nn.relu(z @ hidden["kernel"] + hidden["bias"])
        masked_logits = h @ logits["kernel"] + logits["bias"]
        masked_logits = masked_logits - (1 - available) * 1e10
        return jax.nn.log_softmax(masked_logits)[action]

    analytic = categorical_latent_score(latent, action, available, hidden, logits)
    exact = jax.grad(log_probability)(latent)
    np.testing.assert_allclose(np.asarray(analytic), np.asarray(exact), atol=1e-6)
