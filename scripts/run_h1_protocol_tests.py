#!/usr/bin/env python3
"""Audit the canonical NPS H1 alignment routing and Fisher diagnostic."""

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
    linear_cka_distance,
)
from h1_latent_distortion import fisher_metrics


def gradient_routing_audit():
    key = jax.random.PRNGKey(23)
    actor = jax.random.normal(key, (31, 16))
    critic = jax.random.normal(jax.random.fold_in(key, 1), (31, 16))
    mask = jnp.ones((31,), dtype=jnp.bool_)
    expected = {
        "none": (False, False),
        "c_to_a": (True, False),
        "a_to_c": (False, True),
    }
    records = {}
    for distance_name, distance in (
        ("ln_mse", latent_distance),
        ("linear_cka", linear_cka_distance),
    ):
        for mode, recipients in expected.items():

            def objective(a, c):
                if mode == "none":
                    return jnp.zeros(())
                if mode == "c_to_a":
                    return distance(a, jax.lax.stop_gradient(c), mask)
                return distance(jax.lax.stop_gradient(a), c, mask)

            actor_grad, critic_grad = jax.grad(objective, argnums=(0, 1))(actor, critic)
            norms = tuple(
                float(jnp.sqrt(jnp.sum(jnp.square(gradient))))
                for gradient in (actor_grad, critic_grad)
            )
            if tuple(value > 1e-9 for value in norms) != recipients:
                raise AssertionError(
                    f"{distance_name}/{mode} gradient routing failed: {norms}"
                )
            records[f"{distance_name}/{mode}"] = {
                "actor_gradient_norm": norms[0],
                "critic_gradient_norm": norms[1],
            }
    rotated = float(linear_cka_distance(actor, -actor, mask))
    coordinates = float(latent_distance(actor, -actor, mask))
    if abs(rotated) > 1e-5 or coordinates < 3.9:
        raise AssertionError(
            "CKA and LN-MSE do not distinguish geometry from coordinates"
        )
    return records


def fisher_audit():
    rng = np.random.default_rng(91)
    scores = rng.normal(size=(4096, 12))
    scores -= scores.mean(axis=0, keepdims=True)
    signal = rng.normal(size=(4096,))
    equal = fisher_metrics(scores, signal, signal, 1e-3)
    changed = fisher_metrics(scores, signal, signal + 0.3 * scores[:, 0], 1e-3)
    if equal["epsilon_lat"] > 1e-10 or changed["epsilon_lat"] <= 0:
        raise AssertionError(
            "Fisher distortion is not zero iff reference gradients agree"
        )
    if changed["fisher_min_eigenvalue"] < -1e-10:
        raise AssertionError("The empirical Fisher matrix is not positive semidefinite")
    return {
        "identical_gradient_distortion": equal["epsilon_lat"],
        "different_gradient_distortion": changed["epsilon_lat"],
        "fisher_ridge_absolute": changed["fisher_ridge_absolute"],
    }


def nps_initialization_audit():
    actor = ActorRNN(7, config={"FC_DIM_SIZE": 16, "GRU_HIDDEN_DIM": 16})
    hidden = ScannedRNN.initialize_carry(5, 16)
    inputs = (
        jnp.zeros((1, 5, 13)),
        jnp.zeros((1, 5), dtype=jnp.bool_),
        jnp.ones((5, 7)),
    )
    base = actor.init(jax.random.PRNGKey(101), hidden, inputs)
    independent = jax.tree.map(lambda x: jnp.repeat(x[None, ...], 10, axis=0), base)
    difference = max(
        float(jnp.max(jnp.abs(separate - shared[None, ...])))
        for separate, shared in zip(jax.tree.leaves(independent), jax.tree.leaves(base))
    )
    if difference:
        raise AssertionError("Matched NPS initialization differs across agents")
    return {"num_agents": 10, "max_initial_parameter_difference": difference}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.run_root.expanduser().resolve() / "protocol" / "tests"
    output.mkdir(parents=True, exist_ok=True)
    results = {
        "gradient_routing": gradient_routing_audit(),
        "fisher": fisher_audit(),
        "nps_initialization": nps_initialization_audit(),
    }
    (output / "nps_h1_protocol_audit.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("NPS H1 protocol tests: PASS")
    print(output)


if __name__ == "__main__":
    main()
