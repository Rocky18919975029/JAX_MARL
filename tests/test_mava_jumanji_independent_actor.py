"""Independent recurrent actor checks in the pinned Mava runtime.

These run on the server's Mava virtual environment; the local development
environment may not have JAX/Mava and skips them in that case.
"""

from collections import namedtuple
import importlib.util
from pathlib import Path
import sys

import pytest


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
optax = pytest.importorskip("optax")
tfd = pytest.importorskip("tensorflow_probability.substrates.jax.distributions")
pytest.importorskip("mava")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/mava_jumanji"))
from independent_recurrent_actor import IndependentActor  # noqa: E402
from mava.networks.distributions import IdentityTransformation  # noqa: E402


Observation = namedtuple("Observation", "agents_view action_mask global_state")


class DummyRecurrentActor:
    def init(self, key, carry, observation_done):
        del carry
        observation, _ = observation_done
        return {"params": {"weight": jax.random.normal(
            key, (observation.agents_view.shape[-1], observation.action_mask.shape[-1])
        )}}

    def apply(self, params, *args, return_latent=False, method=None):
        weight = params["params"]["weight"]
        if method is not None:
            latent, action, mask = args
            logits = jnp.where(mask, latent @ weight, jnp.finfo(jnp.float32).min)
            return tfd.Categorical(logits=logits).log_prob(action)
        carry, (observation, _) = args
        latent = observation.agents_view
        logits = jnp.where(
            observation.action_mask, latent @ weight, jnp.finfo(jnp.float32).min
        )
        policy = IdentityTransformation(tfd.Categorical(logits=logits))
        if return_latent:
            return carry, policy, latent
        return carry, policy


def setup_actor():
    actor = IndependentActor(DummyRecurrentActor(), 2)
    observation = Observation(
        agents_view=jnp.ones((3, 2, 3)),
        action_mask=jnp.array([[[True, False], [True, True]]] * 3),
        global_state=jnp.zeros((3, 2)),
    )
    carry = jnp.zeros((3, 2, 3))
    done = jnp.zeros((3, 2), dtype=bool)
    params = actor.init(jax.random.PRNGKey(7), carry, (observation, done))
    return actor, params, carry, observation, done


def test_independent_parameters_and_masked_policy():
    actor, params, carry, observation, done = setup_actor()
    weights = params["params"]["weight"]
    assert weights.shape == (2, 3, 2)
    assert not bool(jnp.allclose(weights[0], weights[1]))
    next_carry, policy, latent = actor.apply(
        params, carry, (observation, done), return_latent=True
    )
    assert next_carry.shape == carry.shape
    assert latent.shape == observation.agents_view.shape
    assert policy.log_prob(jnp.zeros((3, 2), dtype=int)).shape == (3, 2)
    assert bool(jnp.all(policy.distribution.probs_parameter()[:, 0, 1] == 0))

    changed = {"params": {"weight": weights.at[1, :, 0].add(10)}}
    _, changed_policy = actor.apply(changed, carry, (observation, done))
    assert bool(jnp.allclose(
        policy.distribution.probs_parameter()[:, 0],
        changed_policy.distribution.probs_parameter()[:, 0],
    ))
    assert not bool(jnp.allclose(
        policy.distribution.probs_parameter()[:, 1],
        changed_policy.distribution.probs_parameter()[:, 1],
    ))


def test_score_and_optimizer_keep_agent_parameters_separate():
    actor, params, _, observation, _ = setup_actor()
    action = jnp.zeros((3, 2), dtype=int)

    def score_sum(latent):
        return actor.apply(
            params, latent, action, observation.action_mask, method=object()
        ).sum()

    score = jax.grad(score_sum)(observation.agents_view)
    assert score.shape == observation.agents_view.shape
    assert bool(jnp.all(jnp.isfinite(score)))

    optimizer = optax.adam(1e-3)
    state = jax.vmap(optimizer.init)(params)
    grads = jax.tree.map(jnp.ones_like, params)
    updates, new_state = jax.vmap(optimizer.update)(grads, state)
    updated = optax.apply_updates(params, updates)
    assert updated["params"]["weight"].shape == (2, 3, 2)
    assert new_state is not None


def test_real_mava_arec_score_reaches_independent_actor_parameters():
    from mava.networks import ScannedRNN
    from mava.networks.heads import DiscreteActionHead
    from mava.networks.torsos import MLPTorso
    from mava.types import ObservationGlobalState

    source = Path(__file__).resolve().parents[1] / "experiments/mava_jumanji/rec_mappo_arec_nps.py"
    spec = importlib.util.spec_from_file_location("mava_rware_arec_nps", source)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    actor = IndependentActor(module.Actor(
        pre_torso=MLPTorso(layer_sizes=(128,)),
        post_torso=MLPTorso(layer_sizes=(128,)),
        action_head=DiscreteActionHead(action_dim=5),
        hidden_state_dim=128,
    ), 2)
    observation = ObservationGlobalState(
        agents_view=jnp.ones((1, 2, 2, 6), dtype=jnp.float32),
        action_mask=jnp.ones((1, 2, 2, 5), dtype=jnp.bool_),
        global_state=jnp.ones((1, 2, 2, 12), dtype=jnp.float32),
        step_count=None,
    )
    observation = observation._replace(
        action_mask=observation.action_mask.at[..., 4].set(False)
    )
    done = jnp.zeros((1, 2, 2), dtype=bool)
    carry = ScannedRNN.initialize_carry((2, 2), 128)
    params = actor.init(jax.random.PRNGKey(15), carry, (observation, done))
    assert all(leaf.shape[0] == 2 for leaf in jax.tree.leaves(params))
    action = jnp.zeros((1, 2, 2), dtype=jnp.int32)

    def objective(p):
        _, policy, latent = actor.apply(
            p, carry, (observation, done), return_latent=True
        )
        score = module.score_from_latent(
            actor, p, latent, action, observation.action_mask
        )
        return jnp.square(score).sum() + policy.log_prob(action).sum()

    gradients = jax.grad(objective)(params)
    assert all(leaf.shape[0] == 2 for leaf in jax.tree.leaves(gradients))
    assert any(bool(jnp.any(leaf != 0)) for leaf in jax.tree.leaves(gradients))
