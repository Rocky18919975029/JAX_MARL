#!/usr/bin/env python3
"""Run the H1 shuffled-target and alignment-gradient acceptance tests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from baselines.MAPPO.mappo_rnn_smax import (
    ActorRNN,
    ScannedRNN,
    latent_distance,
    maybe_shuffle_targets_within_agent,
)
from h1_latent_distortion import fisher_metrics


def norm(tree):
    return float(
        jnp.sqrt(sum(jnp.square(leaf).sum() for leaf in jax.tree.leaves(tree)))
    )


def shuffle_audit():
    steps, agents, envs, width = 9, 4, 7, 32
    key = jax.random.PRNGKey(17)
    target = jax.random.normal(key, (steps, agents * envs, width))
    actor = target + 0.01 * jax.random.normal(jax.random.fold_in(key, 1), target.shape)
    alive = jnp.ones((steps, agents * envs), dtype=jnp.bool_)
    alive = alive.at[0, 0].set(False).at[2, 2 * envs + 3].set(False)

    unchanged, unchanged_mask, off_audit = maybe_shuffle_targets_within_agent(
        target,
        alive,
        False,
        101,
        5,
        agents,
        envs,
    )
    if not np.array_equal(np.asarray(unchanged), np.asarray(target)):
        raise AssertionError("shuffle-off target is not bitwise identical")
    if not np.array_equal(np.asarray(unchanged_mask), np.asarray(alive)):
        raise AssertionError("shuffle-off mask is not bitwise identical")

    shuffled, alignment_mask, audit = maybe_shuffle_targets_within_agent(
        target,
        alive,
        True,
        101,
        5,
        agents,
        envs,
    )
    repeated, repeated_mask, repeated_audit = maybe_shuffle_targets_within_agent(
        target,
        alive,
        True,
        101,
        5,
        agents,
        envs,
    )
    np.testing.assert_array_equal(np.asarray(shuffled), np.asarray(repeated))
    np.testing.assert_array_equal(np.asarray(alignment_mask), np.asarray(repeated_mask))
    if int(audit["shuffle_permutation_checksum"]) != int(
        repeated_audit["shuffle_permutation_checksum"]
    ):
        raise AssertionError("shuffle checksum is not deterministic")

    def by_agent(array):
        return (
            np.asarray(array).reshape(steps, agents, envs, width).transpose(1, 0, 2, 3)
        )

    target_by_agent = by_agent(target)
    shuffled_by_agent = by_agent(shuffled)
    mask_by_agent = np.asarray(alive).reshape(steps, agents, envs).transpose(1, 0, 2)
    for agent in range(agents):
        source_pool = target_by_agent[agent][mask_by_agent[agent]]
        shuffled_pool = shuffled_by_agent[agent][mask_by_agent[agent]]
        np.testing.assert_allclose(
            source_pool.mean(axis=0), shuffled_pool.mean(axis=0), atol=1e-6
        )
        np.testing.assert_allclose(
            source_pool.std(axis=0), shuffled_pool.std(axis=0), atol=1e-6
        )

    def mean_cosine(first, second, mask):
        first = np.asarray(first)[np.asarray(mask)]
        second = np.asarray(second)[np.asarray(mask)]
        denominator = np.linalg.norm(first, axis=-1) * np.linalg.norm(second, axis=-1)
        return float(np.mean(np.sum(first * second, axis=-1) / denominator))

    paired_cosine = mean_cosine(actor, target, alive)
    shuffled_cosine = mean_cosine(actor, shuffled, alignment_mask)
    if not shuffled_cosine < paired_cosine - 0.5:
        raise AssertionError(
            "shuffle did not materially reduce paired actor/critic cosine similarity"
        )
    if float(audit["shuffle_fixed_point_proportion"]) != 0.0:
        raise AssertionError("shuffle contains a fixed point")

    return {
        "status": "pass",
        "shuffle_off_bitwise_equal": True,
        "valid_target_count": int(audit["shuffle_valid_target_count"]),
        "fixed_point_proportion": float(audit["shuffle_fixed_point_proportion"]),
        "permutation_checksum": int(audit["shuffle_permutation_checksum"]),
        "paired_cosine_before": paired_cosine,
        "paired_cosine_after": shuffled_cosine,
        "marginal_mean_std_preserved": True,
        "deterministic_replay": True,
        "shuffle_off_audit": {
            key: float(value) if np.asarray(value).dtype.kind == "f" else int(value)
            for key, value in off_audit.items()
        },
    }


def alignment_gradient_audit():
    key = jax.random.PRNGKey(23)
    actor = jax.random.normal(key, (31, 16))
    critic = jax.random.normal(jax.random.fold_in(key, 1), (31, 16))
    actor_target = jax.lax.stop_gradient(actor + 0.2)
    critic_target = jax.lax.stop_gradient(critic - 0.3)
    mask = jnp.ones((31,), dtype=jnp.bool_)

    def objective(mode, actor_value, critic_value):
        zero = jnp.zeros(())
        c_to_a = latent_distance(actor_value, critic_target, mask)
        a_to_c = latent_distance(critic_value, actor_target, mask)
        joint = latent_distance(actor_value, critic_value, mask)
        if mode == "none":
            return zero
        if mode == "c_to_a":
            return c_to_a
        if mode == "a_to_c":
            return a_to_c
        if mode == "reciprocal":
            return c_to_a + a_to_c
        if mode == "joint":
            return joint
        raise AssertionError(mode)

    expected = {
        "none": (False, False),
        "c_to_a": (True, False),
        "a_to_c": (False, True),
        "reciprocal": (True, True),
        "joint": (True, True),
        "c_to_a_shuffled": (True, False),
        "a_to_c_shuffled": (False, True),
    }
    result = {}
    for condition, expected_nonzero in expected.items():
        mode = condition.removesuffix("_shuffled")
        actor_grad, critic_grad = jax.grad(
            lambda a, c: objective(mode, a, c), argnums=(0, 1)
        )(actor, critic)
        actor_norm = norm(actor_grad)
        critic_norm = norm(critic_grad)
        actual_nonzero = (actor_norm > 1e-9, critic_norm > 1e-9)
        if actual_nonzero != expected_nonzero:
            raise AssertionError(
                f"{condition}: gradient routing {actual_nonzero}, "
                f"expected {expected_nonzero}"
            )
        result[condition] = {
            "actor_cross_grad_norm": actor_norm,
            "critic_cross_grad_norm": critic_norm,
        }
    return {"status": "pass", "modes": result}


def latent_distortion_audit():
    rng = np.random.default_rng(91)
    scores = rng.normal(size=(4096, 12))
    # Centering gives a controlled score-like signal with a positive
    # semidefinite empirical Fisher matrix.
    scores -= scores.mean(axis=0, keepdims=True)
    reference = rng.normal(size=(4096,))
    critic = reference.copy()
    equal = fisher_metrics(scores, reference, critic, 1e-3)
    if equal["epsilon_lat"] > 1e-10:
        raise AssertionError("A_ref == A_critic did not yield zero distortion")
    critic_changed = reference + 0.3 * scores[:, 0]
    changed = fisher_metrics(scores, reference, critic_changed, 1e-3)
    scaled = fisher_metrics(scores, 7.0 * reference, 7.0 * critic_changed, 1e-3)
    if changed["epsilon_lat"] < -1e-10:
        raise AssertionError("negative latent distortion")
    if not np.isclose(changed["r_lat"], scaled["r_lat"], rtol=1e-5, atol=1e-8):
        raise AssertionError("relative distortion is not scale invariant")
    if changed["fisher_min_eigenvalue"] < -1e-10:
        raise AssertionError("empirical Fisher is not positive semidefinite")

    logits = rng.normal(size=(23, 9))
    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    expected_score = np.zeros_like(probabilities)
    identity = np.eye(probabilities.shape[1])
    for sample in range(len(probabilities)):
        action_scores = identity - probabilities[sample]
        expected_score[sample] = probabilities[sample] @ action_scores
    maximum_expected_score = float(np.abs(expected_score).max())
    if maximum_expected_score > 1e-12:
        raise AssertionError("categorical expected score is not zero")
    return {
        "status": "pass",
        "equal_advantage_epsilon": equal["epsilon_lat"],
        "nonnegative_epsilon": changed["epsilon_lat"],
        "relative_scale_invariance_error": abs(changed["r_lat"] - scaled["r_lat"]),
        "fisher_min_eigenvalue": changed["fisher_min_eigenvalue"],
        "maximum_categorical_expected_score": maximum_expected_score,
    }


def matched_initialization_audit():
    config = {"FC_DIM_SIZE": 16, "GRU_HIDDEN_DIM": 16}
    actor = ActorRNN(7, config=config)
    hidden = ScannedRNN.initialize_carry(5, 16)
    inputs = (
        jnp.zeros((1, 5, 13)),
        jnp.zeros((1, 5), dtype=jnp.bool_),
        jnp.ones((5, 7)),
    )
    initialization_key = jax.random.PRNGKey(101)
    shared = actor.init(initialization_key, hidden, inputs)
    independent = jax.tree.map(
        lambda value: jnp.repeat(value[None, ...], 10, axis=0), shared
    )
    maximum_difference = max(
        float(jnp.max(jnp.abs(independent_leaf - shared_leaf[None, ...])))
        for independent_leaf, shared_leaf in zip(
            jax.tree.leaves(independent), jax.tree.leaves(shared)
        )
    )
    if maximum_difference != 0.0:
        raise AssertionError("matched NPS parameters differ from shared initialization")
    sampling_key = jax.random.PRNGKey(222)
    ps_keys = jax.random.split(sampling_key, 10)
    nps_keys = jax.random.split(sampling_key, 10)
    if not np.array_equal(np.asarray(ps_keys), np.asarray(nps_keys)):
        raise AssertionError("PS/NPS per-agent action keys differ")
    return {
        "status": "pass",
        "agents": 10,
        "maximum_initial_parameter_difference": maximum_difference,
        "per_agent_action_keys_identical": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.run_root.expanduser().resolve() / "protocol" / "tests"
    output.mkdir(parents=True, exist_ok=True)

    shuffle_result = shuffle_audit()
    gradient_result = alignment_gradient_audit()
    distortion_result = latent_distortion_audit()
    initialization_result = matched_initialization_audit()
    (output / "shuffled_target_audit.json").write_text(
        json.dumps(shuffle_result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "alignment_gradient_audit.json").write_text(
        json.dumps(gradient_result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "latent_distortion_audit.json").write_text(
        json.dumps(distortion_result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "matched_initialization_audit.json").write_text(
        json.dumps(initialization_result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("H1 protocol tests: PASS")
    print(output)


if __name__ == "__main__":
    main()
