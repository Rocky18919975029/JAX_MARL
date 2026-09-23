"""Actor-side score-recovery protocol and gradient-routing checks."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.run_smax_actor_score_recovery_training import (
    DEFAULT_BUDGETS,
    Run,
    command,
)
from scripts.smax_score_recoverability import canonical_condition


def test_launcher_keeps_alignment_and_critic_recovery_off(tmp_path):
    args = SimpleNamespace(
        fisher_ridge=1e-3,
        q_learning_rate=1e-3,
        q_steps=8,
        update_epochs=4,
        learning_rate=0.002,
        num_envs=128,
        num_minibatches=4,
        checkpoint_interval=1_000_000,
        wandb_mode="disabled",
        project="test",
    )
    run = Run("3s5z_vs_3s6z", 2, 20_000_000, 0.001)
    parts = command(Path("/repo"), tmp_path, args, run)
    for override in (
        "ACTOR_PARAMETER_SHARING=false",
        "MATCHED_COMPARISON=true",
        "ALIGN_MODE=none",
        "ALIGNMENT_COEF=0",
        "SCORE_RECOVERY=false",
        "SCORE_RECOVERY_COEF=0",
        "ACTOR_SCORE_RECOVERY=true",
        "ACTOR_SCORE_RECOVERY_COEF=0.001",
        "ACTOR_SCORE_RECOVERY_Q_STEPS=8",
        "EXPERIMENT_CONDITION=actor_score_recovery",
        "TOTAL_TIMESTEPS=20000000",
    ):
        assert override in parts
    assert "lam0p001-seed2" in run.name
    assert DEFAULT_BUDGETS["10m_vs_11m"] == 10_000_000
    assert DEFAULT_BUDGETS["3s5z_vs_3s6z"] == 20_000_000
    assert DEFAULT_BUDGETS["6s9z_vs_6s10z"] == 20_000_000
    assert DEFAULT_BUDGETS["smacv2_10_units"] == 10_000_000


def test_offline_measurement_can_identify_actor_intervention():
    assert (
        canonical_condition(
            {"condition": "actor_score_recovery", "align_distance": "ln_mse"}
        )
        == "actor_score_recovery"
    )


def test_fisher_matrix_and_rollout_target_are_stop_gradient():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from baselines.MAPPO.score_recovery_target import whiten_rollout_scores

    score = jnp.asarray([[[[1.0, 0.0]], [[0.0, 2.0]]]])
    valid = jnp.ones((1, 2, 1), dtype=bool)
    target, _, matrix = whiten_rollout_scores(
        score, valid, ridge=0.1, return_matrix=True
    )
    np.testing.assert_allclose(np.asarray(target), np.asarray(score @ matrix[0]))
    assert matrix.shape == (1, 2, 2)
    derivative = jax.grad(
        lambda sample: whiten_rollout_scores(
            sample, valid, ridge=0.1, return_matrix=True
        )[2].sum()
    )(score)
    np.testing.assert_allclose(np.asarray(derivative), 0)


def test_actor_score_objective_has_actor_gradient_but_no_critic_or_q_gradient():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("flax")
    pytest.importorskip("distrax")
    pytest.importorskip("hydra")
    pytest.importorskip("optax")
    from baselines.MAPPO.mappo_rnn_smax import (
        ScoreRecoveryHead,
        categorical_latent_score,
    )

    head = ScoreRecoveryHead(action_dim=3, latent_dim=4, zero_output=True)
    latent_c = jnp.ones((2, 1, 4))
    action = jnp.asarray([[0], [2]])
    q_params = head.init(jax.random.PRNGKey(3), latent_c, action)
    initial_prediction = head.apply(q_params, latent_c, action)
    np.testing.assert_allclose(np.asarray(initial_prediction), 0)
    hidden = {
        "kernel": jnp.array([[0.3, -0.2], [0.1, 0.4], [0.5, 0.2], [-0.1, 0.3]]),
        "bias": jnp.array([0.5, 0.7]),
    }
    logits = {
        "kernel": jnp.array([[0.2, -0.4, 0.3], [0.1, 0.5, -0.2]]),
        "bias": jnp.zeros((3,)),
    }
    latent_a = jnp.array([[[0.2, 0.1, 0.4, -0.1]], [[0.1, 0.3, 0.2, 0.2]]])
    available = jnp.ones((2, 1, 3))
    matrix = jnp.eye(4)

    def actor_objective(actor_latent, hidden_params, critic_latent, q_parameters):
        score = categorical_latent_score(
            actor_latent, action, available, hidden_params, logits
        )
        teacher = jax.lax.stop_gradient(
            head.apply(q_parameters, jax.lax.stop_gradient(critic_latent), action)
        )
        return jnp.square(score @ jax.lax.stop_gradient(matrix) - teacher).sum()

    grad_latent, grad_hidden, grad_critic, grad_q = jax.grad(
        actor_objective, argnums=(0, 1, 2, 3)
    )(latent_a, hidden, latent_c, q_params)
    assert float(jnp.linalg.norm(grad_latent)) > 0
    assert float(jnp.linalg.norm(grad_hidden["kernel"])) > 0
    np.testing.assert_allclose(np.asarray(grad_critic), 0)
    for leaf in jax.tree.leaves(grad_q):
        np.testing.assert_allclose(np.asarray(leaf), 0)
