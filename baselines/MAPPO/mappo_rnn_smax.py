"""
Based on PureJaxRL Implementation of IPPO, with changes to give a centralised critic.
"""

import functools
import json
import math
import os
import re
import subprocess
from functools import partial
from pathlib import Path
from typing import Dict, NamedTuple, Sequence

import jax

# TensorFlow Probability still probes this legacy location when imported by
# Distrax. JAX 0.10 removed it from jax.interpreters.xla but keeps the mapping
# in jax.core. Providing the alias before importing Distrax avoids requiring a
# manual edit inside site-packages.
try:
    jax.interpreters.xla.pytype_aval_mappings
except AttributeError:
    jax.interpreters.xla.pytype_aval_mappings = jax.core.pytype_aval_mappings

import distrax
import flax.core
import flax.linen as nn
import hydra
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from jax.experimental import io_callback
from omegaconf import OmegaConf

import wandb
from tqdm.auto import tqdm
try:
    from baselines.MAPPO.alignment_utils import directional_subspace_containment
    from baselines.MAPPO.score_recovery_target import whiten_rollout_scores
except ModuleNotFoundError:  # Direct execution from baselines/MAPPO.
    from alignment_utils import directional_subspace_containment
    from score_recovery_target import whiten_rollout_scores
from jaxmarl.environments.smax import HeuristicEnemySMAX, map_name_to_scenario
from jaxmarl.wrappers.baselines import JaxMARLWrapper, SMAXLogWrapper, save_params


def _safe_path_component(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-_") or "run"


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def make_metrics_jsonl_callback(path):
    """Create an optional host callback for machine-readable training metrics."""

    if not path:
        return None
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    def callback(metrics):
        payload = {}
        for key, value in metrics.items():
            array = np.asarray(value)
            payload[key] = array.item() if array.ndim == 0 else array.tolist()
        with destination.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, sort_keys=True, allow_nan=True) + "\n")
            file.flush()

    return callback


def make_checkpoint_callback(config, run):
    """Create a host callback that writes evaluation-ready checkpoints."""

    checkpoint_root = Path(config["CHECKPOINT_DIR"]).expanduser().resolve()
    project_name = _safe_path_component(config.get("PROJECT") or "local")
    run_name = _safe_path_component(run.name)
    run_id = _safe_path_component(run.id)
    run_dir = checkpoint_root / project_name / f"{run_name}-{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    def write_json_atomic(path, payload):
        temporary_path = path.with_name(f".{path.name}.tmp")
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, sort_keys=True, default=_json_default)
            file.write("\n")
        os.replace(temporary_path, path)

    def checkpoint_callback(
        actor_params,
        critic_params,
        env_step,
        nominal_env_step,
        is_final,
        is_initial,
    ):
        env_step = int(np.asarray(env_step).item())
        nominal_env_step = int(np.asarray(nominal_env_step).item())
        is_final = bool(np.asarray(is_final).item())
        is_initial = bool(np.asarray(is_initial).item())
        if is_final:
            nominal_env_step = int(config["TOTAL_TIMESTEPS"])
        if is_initial:
            checkpoint_name = "initial"
        elif is_final:
            checkpoint_name = "final"
        else:
            checkpoint_name = f"step_{nominal_env_step:012d}"
        checkpoint_dir = run_dir / checkpoint_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        model_path = checkpoint_dir / "model.safetensors"
        temporary_model_path = checkpoint_dir / ".model.tmp.safetensors"
        save_params(
            {"actor": actor_params, "critic": critic_params},
            temporary_model_path,
        )
        os.replace(temporary_model_path, model_path)

        checkpoint_config = dict(config)
        checkpoint_config["CHECKPOINT_ENV_STEP"] = env_step
        checkpoint_config["CHECKPOINT_NOMINAL_ENV_STEP"] = nominal_env_step
        checkpoint_config["CHECKPOINT_IS_FINAL"] = is_final
        checkpoint_config["CHECKPOINT_IS_INITIAL"] = is_initial
        write_json_atomic(checkpoint_dir / "config.json", checkpoint_config)
        write_json_atomic(
            checkpoint_dir / "metadata.json",
            {
                "format_version": 1,
                "env_step": env_step,
                "nominal_env_step": nominal_env_step,
                "is_final": is_final,
                "is_initial": is_initial,
                "map_name": config["MAP_NAME"],
                "seed": config["SEED"],
                "actor_parameter_sharing": config["ACTOR_PARAMETER_SHARING"],
                "matched_comparison": config["MATCHED_COMPARISON"],
                "align_mode": config["ALIGN_MODE"],
                "align_distance": config["ALIGN_DISTANCE"],
                "align_distance_eps": config["ALIGN_DISTANCE_EPS"],
                "align_containment_ridge_ratio": config[
                    "ALIGN_CONTAINMENT_RIDGE_RATIO"
                ],
                "align_containment_epsilon": config["ALIGN_CONTAINMENT_EPS"],
                "align_containment_group_by_unit_type": config[
                    "ALIGN_CONTAINMENT_GROUP_BY_UNIT_TYPE"
                ],
                "align_containment_min_group_samples": config[
                    "ALIGN_CONTAINMENT_MIN_GROUP_SAMPLES"
                ],
                "alignment_coef": config["ALIGNMENT_COEF"],
                "oracle_latent_distortion": config[
                    "ORACLE_LATENT_DISTORTION"
                ],
                "oracle_distortion_coef": config["ORACLE_DISTORTION_COEF"],
                "oracle_fisher_ridge": config["ORACLE_FISHER_RIDGE"],
                "score_recovery": config["SCORE_RECOVERY"],
                "score_recovery_coef": config["SCORE_RECOVERY_COEF"],
                "score_recovery_fisher_ridge": config[
                    "SCORE_RECOVERY_FISHER_RIDGE"
                ],
                "oracle_reference_multiplier": config[
                    "ORACLE_REFERENCE_MULTIPLIER"
                ],
                "oracle_reference_horizon": config[
                    "ORACLE_REFERENCE_HORIZON"
                ],
                "oracle_reference_baseline": config[
                    "ORACLE_REFERENCE_BASELINE"
                ],
                "align_target_shuffle": config["ALIGN_TARGET_SHUFFLE"],
                "condition": config.get("EXPERIMENT_CONDITION", ""),
                "matrix_profile": config.get("MATRIX_PROFILE", ""),
                "protocol_version": config.get("PROTOCOL_VERSION", ""),
                "git_commit": config.get("GIT_COMMIT", ""),
                "wandb_project": run.project,
                "wandb_run_id": run.id,
                "wandb_run_name": run.name,
            },
        )
        write_json_atomic(
            run_dir / "latest.json",
            {
                "checkpoint": checkpoint_name,
                "env_step": env_step,
                "nominal_env_step": nominal_env_step,
                "is_final": is_final,
                "is_initial": is_initial,
            },
        )
        print(f"Checkpoint saved: {checkpoint_dir}", flush=True)

        if config["WANDB_UPLOAD_CHECKPOINTS"]:
            artifact = wandb.Artifact(
                f"{run_name}-{run_id}-checkpoint",
                type="model",
                metadata={
                    "env_step": env_step,
                    "nominal_env_step": nominal_env_step,
                    "is_final": is_final,
                    "is_initial": is_initial,
                    "map_name": config["MAP_NAME"],
                    "seed": config["SEED"],
                },
            )
            artifact.add_dir(str(checkpoint_dir))
            aliases = ["latest", f"step-{env_step}"]
            if is_initial:
                aliases.append("initial")
            if is_final:
                aliases.append("final")
            run.log_artifact(artifact, aliases=aliases)

        return np.int32(0)

    return checkpoint_callback, run_dir


class SMAXWorldStateWrapper(JaxMARLWrapper):
    """
    Provides a `"world_state"` observation for the centralised critic.
    world state observation of dimension: (num_agents, world_state_size)
    """

    def __init__(
        self,
        env: HeuristicEnemySMAX,
        obs_with_agent_id=True,
    ):
        super().__init__(env)
        self.obs_with_agent_id = obs_with_agent_id

        if not self.obs_with_agent_id:
            self._world_state_size = self._env.state_size
            self.world_state_fn = self.ws_just_env_state
        else:
            self._world_state_size = self._env.state_size + self._env.num_allies
            self.world_state_fn = self.ws_with_agent_id

    @partial(jax.jit, static_argnums=0)
    def reset(self, key):
        obs, env_state = self._env.reset(key)
        obs["world_state"] = self.world_state_fn(obs, env_state)
        return obs, env_state

    @partial(jax.jit, static_argnums=0)
    def step(self, key, state, action):
        obs, env_state, reward, done, info = self._env.step(key, state, action)
        obs["world_state"] = self.world_state_fn(obs, state)
        return obs, env_state, reward, done, info

    @partial(jax.jit, static_argnums=0)
    def ws_just_env_state(self, obs, state):
        # return all_obs
        world_state = obs["world_state"]
        world_state = world_state[None].repeat(self._env.num_allies, axis=0)
        return world_state

    @partial(jax.jit, static_argnums=0)
    def ws_with_agent_id(self, obs, state):
        # all_obs = jnp.array([obs[agent] for agent in self._env.agents])
        world_state = obs["world_state"]
        world_state = world_state[None].repeat(self._env.num_allies, axis=0)
        one_hot = jnp.eye(self._env.num_allies)
        return jnp.concatenate((world_state, one_hot), axis=1)

    def world_state_size(self):

        return self._world_state_size


class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        # print('ins', ins)
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones, avail_actions = x
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="Dense_0",
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, actor_latent = ScannedRNN(name="ScannedRNN_0")(hidden, rnn_in)

        actor_mean = nn.Dense(
            self.config["GRU_HIDDEN_DIM"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
            name="Dense_1",
        )(actor_latent)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="Dense_2",
        )(actor_mean)
        unavail_actions = 1 - avail_actions
        action_logits = actor_mean - (unavail_actions * 1e10)

        pi = distrax.Categorical(logits=action_logits)

        return hidden, pi, actor_latent


class CriticRNN(nn.Module):
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        world_state, dones = x
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(world_state)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, critic_latent = ScannedRNN()(hidden, rnn_in)

        critic = nn.Dense(
            self.config["GRU_HIDDEN_DIM"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(critic_latent)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, jnp.squeeze(critic, axis=-1), critic_latent


class ScoreRecoveryHead(nn.Module):
    """Train-time critic readout; separate from the post-hoc linear probe."""

    action_dim: int
    latent_dim: int

    @nn.compact
    def __call__(self, critic_latent, action):
        one_hot = jax.nn.one_hot(action, self.action_dim)
        features = jnp.concatenate((critic_latent, one_hot), axis=-1)
        features = nn.Dense(self.latent_dim, name="Dense_0")(features)
        features = nn.relu(features)
        return nn.Dense(self.latent_dim, name="Dense_1")(features)


class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    world_state: jnp.ndarray
    actor_latent_old: jnp.ndarray
    critic_latent_old: jnp.ndarray
    alive_mask: jnp.ndarray
    alignment_mask: jnp.ndarray
    unit_type: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray


class OracleReferenceTransition(NamedTuple):
    """Compact transition stored for the independent oracle rollout."""

    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    baseline: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    alive_mask: jnp.ndarray
    avail_actions: jnp.ndarray


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    # print('batchify', x.shape)
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def vmapped_optimizer(tx):
    """Apply one optimizer independently to every leading parameter slice."""

    def init_fn(params):
        return jax.vmap(tx.init)(params)

    def update_fn(updates, state, params=None):
        if params is None:
            return jax.vmap(lambda g, s: tx.update(g, s))(updates, state)
        return jax.vmap(tx.update)(updates, state, params)

    return optax.GradientTransformation(init_fn, update_fn)


def tree_l2_norm(tree):
    return jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in jax.tree.leaves(tree)))


def complete_mc_return_to_go(reward, global_done, gamma):
    """Return exact within-rollout MC returns and their validity mask.

    The scan deliberately does not bootstrap at the rollout boundary.  A
    transition is valid only when an episode termination is observed at or
    after that transition inside the same rollout.  Consequently, the
    unfinished suffix after the final observed termination is excluded from
    the oracle objective.
    """

    reward = jax.lax.stop_gradient(reward)
    global_done = jax.lax.stop_gradient(global_done.astype(bool))

    def reverse_step(carry, transition):
        next_return, future_has_terminal = carry
        current_reward, current_done = transition
        current_return = current_reward + gamma * (~current_done) * next_return
        current_valid = current_done | future_has_terminal
        return (current_return, current_valid), (current_return, current_valid)

    initial = (jnp.zeros_like(reward[-1]), jnp.zeros_like(global_done[-1]))
    _, (returns, valid) = jax.lax.scan(
        reverse_step,
        initial,
        (reward, global_done),
        reverse=True,
    )
    return jax.lax.stop_gradient(returns), jax.lax.stop_gradient(valid)


def crossfit_linear_reference_baseline(
    returns,
    frozen_critic_value,
    mask,
    num_agents,
    num_envs,
    ridge=1e-6,
):
    """Two-fold cross-fitted affine baseline built from pre-action values.

    Environment indices define independent folds.  For each agent, an affine
    map from the frozen pre-update critic value to complete MC return is fitted
    on one fold and evaluated on the other.  Thus a held-out transition's
    baseline never uses its own action or return.  The slope naturally shrinks
    toward zero when the initial critic has no predictive signal.
    """

    time_steps = returns.shape[0]
    returns_by_env = returns.reshape((time_steps, num_agents, num_envs))
    values_by_env = frozen_critic_value.reshape(
        (time_steps, num_agents, num_envs)
    )
    mask_by_env = mask.reshape((time_steps, num_agents, num_envs)).astype(
        returns.dtype
    )
    fold_a = (jnp.arange(num_envs) % 2 == 0)[None, None, :]
    fold_b = ~fold_a

    def fit_on_apply_to(train_fold, test_fold):
        train_mask = mask_by_env * train_fold.astype(returns.dtype)
        count = jnp.maximum(train_mask.sum(axis=(0, 2)), 1.0)
        mean_value = (
            (values_by_env * train_mask).sum(axis=(0, 2)) / count
        )
        mean_return = (
            (returns_by_env * train_mask).sum(axis=(0, 2)) / count
        )
        centered_value = values_by_env - mean_value[None, :, None]
        centered_return = returns_by_env - mean_return[None, :, None]
        covariance = (
            centered_value * centered_return * train_mask
        ).sum(axis=(0, 2))
        value_variance = (
            jnp.square(centered_value) * train_mask
        ).sum(axis=(0, 2))
        slope = covariance / (value_variance + ridge * count)
        intercept = mean_return - slope * mean_value
        prediction = (
            intercept[None, :, None]
            + slope[None, :, None] * values_by_env
        )
        return prediction * test_fold.astype(returns.dtype)

    baseline = fit_on_apply_to(fold_a, fold_b) + fit_on_apply_to(fold_b, fold_a)
    return jax.lax.stop_gradient(baseline.reshape(returns.shape))


def categorical_latent_score(
    actor_latent,
    action,
    avail_actions,
    policy_hidden_params,
    policy_logits_params,
):
    """Compute d log pi(a|z) / dz exactly for the two-layer policy head.

    This analytic expression is equivalent to differentiating the Categorical
    log probability with respect to ``actor_latent``.  Keeping it as ordinary
    JAX operations makes the outer derivative of the oracle distortion both
    explicit and substantially cheaper than nesting reverse-mode transforms.
    """

    hidden_pre = (
        actor_latent @ policy_hidden_params["kernel"]
        + policy_hidden_params["bias"]
    )
    hidden = jax.nn.relu(hidden_pre)
    logits = hidden @ policy_logits_params["kernel"] + policy_logits_params["bias"]
    logits = logits - (1 - avail_actions) * 1e10
    probabilities = jax.nn.softmax(logits, axis=-1)
    logit_score = jax.nn.one_hot(action, logits.shape[-1]) - probabilities
    hidden_score = logit_score @ policy_logits_params["kernel"].T
    hidden_pre_score = hidden_score * (hidden_pre > 0).astype(hidden_score.dtype)
    return hidden_pre_score @ policy_hidden_params["kernel"].T


def oracle_latent_distortion(
    reference_scores,
    reference_advantage,
    reference_return,
    reference_mask,
    reference_importance_weight,
    critic_scores,
    critic_advantage,
    critic_mask,
    critic_importance_weight,
    fisher_ridge,
):
    """Compare a large independent MC reference with the PPO minibatch GAE.

    Reference and critic samples are deliberately separate and may have
    different time/batch dimensions after the leading agent axis.  The
    reference Fisher is estimated only from the larger independent rollout.
    All returns/advantages/masks are stop-gradient; gradients flow through the
    current actor latent scores and importance ratios only.
    """

    reference_advantage = jax.lax.stop_gradient(reference_advantage)
    reference_return = jax.lax.stop_gradient(reference_return)
    reference_mask = jax.lax.stop_gradient(
        reference_mask.astype(reference_scores.dtype)
    )
    critic_advantage = jax.lax.stop_gradient(critic_advantage)
    critic_mask = jax.lax.stop_gradient(critic_mask.astype(critic_scores.dtype))

    def per_agent(
        agent_reference_scores,
        agent_reference,
        agent_return,
        agent_reference_mask,
        agent_reference_weight,
        agent_critic_scores,
        agent_critic,
        agent_critic_mask,
        agent_critic_weight,
    ):
        flat_reference_scores = agent_reference_scores.reshape(
            (-1, agent_reference_scores.shape[-1])
        )
        flat_reference = agent_reference.reshape((-1,))
        flat_return = agent_return.reshape((-1,))
        flat_reference_mask = agent_reference_mask.reshape((-1,))
        flat_reference_weight = agent_reference_weight.reshape((-1,))
        reference_count = flat_reference_mask.sum()
        reference_denominator = jnp.maximum(reference_count, 1.0)
        weighted_reference_scores = flat_reference_scores * (
            flat_reference_mask * flat_reference_weight
        )[:, None]
        g_reference = (
            weighted_reference_scores * flat_reference[:, None]
        ).sum(axis=0) / reference_denominator
        fisher = (
            weighted_reference_scores.T @ flat_reference_scores
            / reference_denominator
        )
        fisher = 0.5 * (fisher + fisher.T)

        flat_critic_scores = agent_critic_scores.reshape(
            (-1, agent_critic_scores.shape[-1])
        )
        flat_critic = agent_critic.reshape((-1,))
        flat_critic_mask = agent_critic_mask.reshape((-1,))
        flat_critic_weight = agent_critic_weight.reshape((-1,))
        critic_count = flat_critic_mask.sum()
        critic_denominator = jnp.maximum(critic_count, 1.0)
        weighted_critic_scores = flat_critic_scores * (
            flat_critic_mask * flat_critic_weight
        )[:, None]
        g_critic = (
            weighted_critic_scores * flat_critic[:, None]
        ).sum(axis=0) / critic_denominator

        delta = g_reference - g_critic
        ridge_matrix = fisher + fisher_ridge * jnp.eye(
            fisher.shape[0], dtype=fisher.dtype
        )
        solution = jnp.linalg.solve(ridge_matrix, delta)
        distortion = jnp.maximum(delta @ solution, 0.0)
        distortion = jnp.where(
            (reference_count > 0) & (critic_count > 0), distortion, 0.0
        )

        def masked_std(values):
            mean = (values * flat_reference_mask).sum() / reference_denominator
            variance = (
                jnp.square(values - mean) * flat_reference_mask
            ).sum() / reference_denominator
            return jnp.sqrt(jnp.maximum(variance, 0.0))

        return (
            distortion,
            reference_count,
            critic_count,
            g_reference,
            g_critic,
            masked_std(flat_return),
            masked_std(flat_reference),
        )

    (
        distortion,
        reference_count,
        critic_count,
        g_reference,
        g_critic,
        reference_return_std,
        reference_advantage_std,
    ) = jax.vmap(per_agent)(
        reference_scores,
        reference_advantage,
        reference_return,
        reference_mask,
        reference_importance_weight,
        critic_scores,
        critic_advantage,
        critic_mask,
        critic_importance_weight,
    )
    reference_norm = jnp.sqrt(jnp.square(g_reference).sum())
    critic_norm = jnp.sqrt(jnp.square(g_critic).sum())
    cosine_denominator = jnp.maximum(reference_norm * critic_norm, 1e-12)
    return {
        "epsilon_lat": distortion.sum(),
        "epsilon_lat_per_agent": distortion,
        "valid_samples": reference_count.sum(),
        "valid_samples_per_agent": reference_count,
        "reference_valid_samples": reference_count.sum(),
        "reference_valid_samples_per_agent": reference_count,
        "critic_valid_samples": critic_count.sum(),
        "critic_valid_samples_per_agent": critic_count,
        "reference_to_critic_sample_ratio": (
            reference_count.sum() / jnp.maximum(critic_count.sum(), 1.0)
        ),
        "reference_gradient_norm": reference_norm,
        "critic_gradient_norm": critic_norm,
        "reference_critic_gradient_cosine": (
            (g_reference * g_critic).sum() / cosine_denominator
        ),
        "reference_mc_return_std": reference_return_std.mean(),
        "reference_baselined_advantage_std": reference_advantage_std.mean(),
        "reference_baseline_variance_reduction": 1.0
        - jnp.square(reference_advantage_std.mean())
        / jnp.maximum(jnp.square(reference_return_std.mean()), 1e-12),
    }


def normalize_latent_samples(latent):
    """Parameter-free LayerNorm over each latent sample's feature axis."""

    mean = latent.mean(axis=-1, keepdims=True)
    variance = jnp.square(latent - mean).mean(axis=-1, keepdims=True)
    return (latent - mean) * jax.lax.rsqrt(variance + 1e-5)


def latent_distance(source, target, mask):
    """Masked per-sample LayerNorm MSE used by the original experiments."""

    distance = jnp.square(
        normalize_latent_samples(source) - normalize_latent_samples(target)
    ).mean(axis=-1)
    mask = mask.astype(distance.dtype)
    return (distance * mask).sum() / jnp.maximum(mask.sum(), 1.0)


def linear_cka_distance(source, target, mask, epsilon=1e-8):
    """Masked linear CKA distance after per-sample LayerNorm.

    All non-feature axes form the sample batch. Invalid samples have zero
    weight both when estimating the batch mean and when forming cross/self
    covariance matrices. A batch with fewer than two valid samples carries no
    alignment information and therefore contributes zero loss.
    """

    source = normalize_latent_samples(source).reshape((-1, source.shape[-1]))
    target = normalize_latent_samples(target).reshape((-1, target.shape[-1]))
    weights = mask.reshape((-1,)).astype(source.dtype)
    count = weights.sum()
    safe_count = jnp.maximum(count, 1.0)
    weights = weights[:, None]

    source_mean = (source * weights).sum(axis=0, keepdims=True) / safe_count
    target_mean = (target * weights).sum(axis=0, keepdims=True) / safe_count
    source_centered = (source - source_mean) * weights
    target_centered = (target - target_mean) * weights

    cross = source_centered.T @ target_centered
    source_self = source_centered.T @ source_centered
    target_self = target_centered.T @ target_centered
    numerator = jnp.square(cross).sum()
    denominator = jnp.sqrt(jnp.square(source_self).sum()) * jnp.sqrt(
        jnp.square(target_self).sum()
    )
    similarity = numerator / (denominator + jnp.asarray(epsilon, source.dtype))
    distance = 1.0 - similarity
    return jnp.where(count > 1, distance, jnp.zeros_like(distance))


def representation_distance(
    source,
    target,
    mask,
    distance_name="ln_mse",
    num_agent_groups=1,
    epsilon=1e-8,
    containment_ridge_ratio=1e-3,
    containment_epsilon=1e-6,
    group_labels=None,
    num_label_groups=0,
    min_group_samples=2,
    return_statistics=False,
):
    """Dispatch to the configured alignment distance.

    Linear CKA is evaluated independently within each agent's sample pool.
    This avoids comparing coordinates emitted by different independent actor
    encoders and gives PS and NPS runs the same loss definition.
    """

    if distance_name == "ln_mse":
        if return_statistics:
            raise ValueError("return_statistics is only supported for containment")
        return latent_distance(source, target, mask)
    if distance_name not in {"linear_cka", "containment"}:
        raise ValueError(f"Unknown alignment distance: {distance_name!r}")
    if source.shape[-2] % num_agent_groups != 0:
        raise ValueError("The latent sample axis must be divisible by num_agent_groups")

    def split_groups(value):
        value = value.reshape(
            (
                value.shape[0],
                num_agent_groups,
                value.shape[1] // num_agent_groups,
                *value.shape[2:],
            )
        )
        return jnp.swapaxes(value, 0, 1)

    grouped_source = split_groups(source)
    grouped_target = split_groups(target)
    grouped_mask = split_groups(mask)
    if distance_name == "containment":
        statistics = grouped_containment_statistics(
            grouped_source,
            grouped_target,
            grouped_mask,
            ridge_ratio=containment_ridge_ratio,
            epsilon=containment_epsilon,
            group_labels=(
                None if group_labels is None else split_groups(group_labels)
            ),
            num_label_groups=num_label_groups,
            min_group_samples=min_group_samples,
        )
        return statistics if return_statistics else statistics["loss"]
    if return_statistics:
        raise ValueError("return_statistics is only supported for containment")
    distances = jax.vmap(linear_cka_distance, in_axes=(0, 0, 0, None))(
        grouped_source,
        grouped_target,
        grouped_mask,
        epsilon,
    )
    valid_groups = grouped_mask.reshape((num_agent_groups, -1)).sum(axis=1) > 1
    valid_groups = valid_groups.astype(distances.dtype)
    return (distances * valid_groups).sum() / jnp.maximum(valid_groups.sum(), 1.0)


def grouped_containment_statistics(
    grouped_source,
    grouped_target,
    grouped_mask,
    *,
    ridge_ratio=1e-3,
    epsilon=1e-6,
    group_labels=None,
    num_label_groups=0,
    min_group_samples=2,
):
    """Compute DSC per slot, optionally splitting every slot by unit type."""

    if group_labels is None:
        loss, similarity, effective_rank, count = jax.vmap(
            directional_subspace_containment,
            in_axes=(0, 0, 0, None, None),
        )(
            grouped_source,
            grouped_target,
            grouped_mask,
            ridge_ratio,
            epsilon,
        )
    else:
        if num_label_groups <= 0:
            raise ValueError("num_label_groups must be positive with group_labels")
        labels = jnp.arange(num_label_groups, dtype=group_labels.dtype)
        typed_masks = grouped_mask[:, None, ...] & (
            group_labels[:, None, ...] == labels[None, :, None, None]
        )

        def per_slot(source, target, masks):
            return jax.vmap(
                directional_subspace_containment,
                in_axes=(None, None, 0, None, None),
            )(source, target, masks, ridge_ratio, epsilon)

        loss, similarity, effective_rank, count = jax.vmap(per_slot)(
            grouped_source, grouped_target, typed_masks
        )

    valid = (count >= min_group_samples).astype(grouped_source.dtype)
    denominator = jnp.maximum(valid.sum(), 1.0)
    return {
        "loss": (loss * valid).sum() / denominator,
        "similarity": (similarity * valid).sum() / denominator,
        "source_effective_rank": (effective_rank * valid).sum() / denominator,
        "valid_samples": (count * valid).sum() / denominator,
        "valid_groups": valid.sum(),
    }


def shuffle_targets_within_agent(
    target_latent,
    alive_mask,
    training_seed,
    update_index,
    num_agents,
    num_envs,
    seed_offset=700000,
):
    """Derange alive targets over environment x time within each agent.

    A random non-zero cyclic shift is a bijection over every agent's valid
    pool, so it exactly preserves that pool's marginal distribution and has
    no fixed points whenever at least two valid samples exist.  The key is an
    audit-friendly RNG substream independent from rollout/action sampling.
    """

    num_steps = target_latent.shape[0]
    latent_dim = target_latent.shape[-1]
    pool_size = num_steps * num_envs

    target_by_agent = target_latent.reshape(
        (num_steps, num_agents, num_envs, latent_dim)
    )
    target_by_agent = jnp.transpose(target_by_agent, (1, 0, 2, 3)).reshape(
        (num_agents, pool_size, latent_dim)
    )
    alive_by_agent = alive_mask.reshape((num_steps, num_agents, num_envs))
    alive_by_agent = jnp.transpose(alive_by_agent, (1, 0, 2)).reshape(
        (num_agents, pool_size)
    )

    valid_counts = alive_by_agent.sum(axis=1, dtype=jnp.int32)
    eligible = jnp.logical_and(alive_by_agent, valid_counts[:, None] > 1)
    positions = jnp.arange(pool_size, dtype=jnp.int32)
    positions_by_agent = jnp.broadcast_to(positions, alive_by_agent.shape)

    # Sorting [valid positions, invalid positions] gives a static-size lookup
    # from rank-within-valid-pool to the original environment/time index.
    valid_order = jnp.argsort(
        jnp.where(alive_by_agent, positions_by_agent, pool_size + positions_by_agent),
        axis=1,
    )
    valid_rank = jnp.cumsum(alive_by_agent, axis=1, dtype=jnp.int32) - 1

    shuffle_key = jax.random.PRNGKey(training_seed)
    shuffle_key = jax.random.fold_in(shuffle_key, seed_offset)
    shuffle_key = jax.random.fold_in(shuffle_key, update_index)
    uniforms = jax.random.uniform(shuffle_key, (num_agents,))
    shift_range = jnp.maximum(valid_counts - 1, 1)
    shifts = 1 + jnp.floor(uniforms * shift_range).astype(jnp.int32)
    shifts = jnp.where(valid_counts > 1, shifts, 0)

    safe_counts = jnp.maximum(valid_counts, 1)
    target_rank = jnp.mod(valid_rank + shifts[:, None], safe_counts[:, None])
    target_indices = jnp.take_along_axis(valid_order, target_rank, axis=1)
    target_indices = jnp.where(eligible, target_indices, positions_by_agent)
    shuffled_by_agent = jnp.take_along_axis(
        target_by_agent, target_indices[..., None], axis=1
    )

    shuffled = shuffled_by_agent.reshape((num_agents, num_steps, num_envs, latent_dim))
    shuffled = jnp.transpose(shuffled, (1, 0, 2, 3)).reshape(target_latent.shape)
    alignment_mask = jnp.transpose(
        eligible.reshape((num_agents, num_steps, num_envs)), (1, 0, 2)
    ).reshape(alive_mask.shape)

    eligible_count = alignment_mask.sum(dtype=jnp.int32)
    fixed_count = jnp.logical_and(eligible, target_indices == positions_by_agent).sum(
        dtype=jnp.int32
    )
    fixed_point_proportion = fixed_count.astype(jnp.float32) / jnp.maximum(
        eligible_count, 1
    )
    agent_ids = jnp.arange(num_agents, dtype=jnp.int32)[:, None]
    checksum_terms = jnp.where(
        eligible,
        (agent_ids + 1) * 1009
        + (positions_by_agent + 1) * 9176
        + (target_indices + 1) * 6361,
        0,
    )
    checksum = checksum_terms.sum(dtype=jnp.int32)
    audit = {
        "shuffle_enabled": jnp.asarray(1, dtype=jnp.int32),
        "shuffle_valid_target_count": eligible_count,
        "shuffle_fixed_point_proportion": fixed_point_proportion,
        "shuffle_permutation_checksum": checksum,
    }
    return shuffled, alignment_mask, audit


def maybe_shuffle_targets_within_agent(
    target_latent,
    alive_mask,
    enabled,
    training_seed,
    update_index,
    num_agents,
    num_envs,
    seed_offset=700000,
):
    """Apply the H1 control shuffle, preserving the exact input when disabled."""

    if not enabled:
        audit = {
            "shuffle_enabled": jnp.asarray(0, dtype=jnp.int32),
            "shuffle_valid_target_count": jnp.asarray(0, dtype=jnp.int32),
            "shuffle_fixed_point_proportion": jnp.asarray(0.0),
            "shuffle_permutation_checksum": jnp.asarray(0, dtype=jnp.int32),
        }
        return target_latent, alive_mask, audit
    return shuffle_targets_within_agent(
        target_latent,
        alive_mask,
        training_seed,
        update_index,
        num_agents,
        num_envs,
        seed_offset,
    )


def make_train(
    config,
    progress_bar=None,
    checkpoint_callback=None,
    metrics_jsonl_callback=None,
):
    config.setdefault("ALIGN_DISTANCE", "ln_mse")
    config.setdefault("ALIGN_DISTANCE_EPS", 1e-8)
    config.setdefault("ALIGN_CONTAINMENT_RIDGE_RATIO", 1e-3)
    config.setdefault("ALIGN_CONTAINMENT_EPS", 1e-6)
    config.setdefault("ALIGN_CONTAINMENT_GROUP_BY_UNIT_TYPE", True)
    config.setdefault("ALIGN_CONTAINMENT_MIN_GROUP_SAMPLES", 64)
    config.setdefault("ALIGN_CONTAINMENT_RIDGE_RATIO", 1e-3)
    config.setdefault("ALIGN_CONTAINMENT_EPS", 1e-6)
    config.setdefault("ALIGN_CONTAINMENT_GROUP_BY_UNIT_TYPE", True)
    config.setdefault("ALIGN_CONTAINMENT_MIN_GROUP_SAMPLES", 64)
    config.setdefault("ALIGN_GRADIENT_CALIBRATION", False)
    config.setdefault("ORACLE_LATENT_DISTORTION", False)
    config.setdefault("ORACLE_DISTORTION_COEF", 0.0)
    config.setdefault("ORACLE_FISHER_RIDGE", 1e-3)
    config.setdefault("ORACLE_REFERENCE_MULTIPLIER", 4)
    config.setdefault("ORACLE_REFERENCE_BASELINE", "crossfit_linear_critic")
    config.setdefault("ORACLE_REFERENCE_SEED_OFFSET", 900_000)
    config.setdefault("SCORE_RECOVERY", False)
    config.setdefault("SCORE_RECOVERY_COEF", 0.0)
    config.setdefault("SCORE_RECOVERY_FISHER_RIDGE", 1e-3)
    scenario = map_name_to_scenario(config["MAP_NAME"])
    env = HeuristicEnemySMAX(scenario=scenario, **config["ENV_KWARGS"])
    config["ORACLE_REFERENCE_HORIZON"] = (
        int(config["ORACLE_REFERENCE_MULTIPLIER"]) * int(config["NUM_STEPS"])
        + int(env.max_steps)
    )
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    valid_align_modes = {"none", "c_to_a", "a_to_c", "reciprocal", "joint"}
    if config["ALIGN_MODE"] not in valid_align_modes:
        raise ValueError(
            f"ALIGN_MODE must be one of {sorted(valid_align_modes)}, got "
            f"{config['ALIGN_MODE']!r}."
        )
    valid_align_distances = {"ln_mse", "linear_cka", "containment"}
    if config["ALIGN_DISTANCE"] not in valid_align_distances:
        raise ValueError(
            f"ALIGN_DISTANCE must be one of {sorted(valid_align_distances)}, got "
            f"{config['ALIGN_DISTANCE']!r}."
        )
    if float(config["ALIGN_DISTANCE_EPS"]) <= 0:
        raise ValueError("ALIGN_DISTANCE_EPS must be positive.")
    if float(config["ALIGN_CONTAINMENT_RIDGE_RATIO"]) < 0:
        raise ValueError("ALIGN_CONTAINMENT_RIDGE_RATIO must be nonnegative.")
    if float(config["ALIGN_CONTAINMENT_EPS"]) <= 0:
        raise ValueError("ALIGN_CONTAINMENT_EPS must be positive.")
    if int(config["ALIGN_CONTAINMENT_MIN_GROUP_SAMPLES"]) < 2:
        raise ValueError("ALIGN_CONTAINMENT_MIN_GROUP_SAMPLES must be at least 2.")
    if config["ALIGN_MODE"] != "none" and not config["MATCHED_COMPARISON"]:
        raise ValueError("Representation alignment requires MATCHED_COMPARISON=true.")
    if float(config["ORACLE_DISTORTION_COEF"]) < 0:
        raise ValueError("ORACLE_DISTORTION_COEF must be nonnegative.")
    if float(config["ORACLE_FISHER_RIDGE"]) <= 0:
        raise ValueError("ORACLE_FISHER_RIDGE must be positive.")
    if int(config["ORACLE_REFERENCE_MULTIPLIER"]) < 2:
        raise ValueError("ORACLE_REFERENCE_MULTIPLIER must be at least 2.")
    if config["ORACLE_REFERENCE_BASELINE"] not in {
        "crossfit_linear_critic",
        "frozen_critic",
        "zero",
    }:
        raise ValueError(
            "ORACLE_REFERENCE_BASELINE must be 'crossfit_linear_critic', "
            "'frozen_critic', or 'zero'."
        )
    if int(config["ORACLE_REFERENCE_SEED_OFFSET"]) <= 0:
        raise ValueError("ORACLE_REFERENCE_SEED_OFFSET must be positive.")
    if (
        not math.isfinite(float(config["SCORE_RECOVERY_FISHER_RIDGE"]))
        or float(config["SCORE_RECOVERY_FISHER_RIDGE"]) <= 0
    ):
        raise ValueError("SCORE_RECOVERY_FISHER_RIDGE must be finite and positive.")
    if config["SCORE_RECOVERY"]:
        if not config["MATCHED_COMPARISON"] or config["ACTOR_PARAMETER_SHARING"]:
            raise ValueError("Score recovery requires matched NPS actors.")
        if config["ALIGN_MODE"] != "none" or float(config["ALIGNMENT_COEF"]) != 0:
            raise ValueError("Score recovery requires alignment to be disabled.")
        if config["ORACLE_LATENT_DISTORTION"] or config["ALIGN_TARGET_SHUFFLE"]:
            raise ValueError("Score recovery cannot combine with oracle or shuffle.")
        if (
            not math.isfinite(float(config["SCORE_RECOVERY_COEF"]))
            or float(config["SCORE_RECOVERY_COEF"]) <= 0
        ):
            raise ValueError(
                "SCORE_RECOVERY_COEF must be finite and positive when enabled."
            )
    elif float(config["SCORE_RECOVERY_COEF"]) != 0:
        raise ValueError("SCORE_RECOVERY_COEF must be zero when disabled.")
    if config["ORACLE_LATENT_DISTORTION"]:
        if not config["MATCHED_COMPARISON"]:
            raise ValueError(
                "Oracle latent-distortion training requires MATCHED_COMPARISON=true."
            )
        if config["ACTOR_PARAMETER_SHARING"]:
            raise ValueError(
                "The oracle protocol is defined for independent (NPS) actors."
            )
        if config["ALIGN_MODE"] != "none":
            raise ValueError(
                "Oracle latent distortion and representation alignment are mutually "
                "exclusive experimental conditions."
            )
        if float(config["ORACLE_DISTORTION_COEF"]) <= 0:
            raise ValueError(
                "Oracle training requires ORACLE_DISTORTION_COEF > 0."
            )
    elif float(config["ORACLE_DISTORTION_COEF"]) != 0:
        raise ValueError(
            "ORACLE_DISTORTION_COEF must be zero when the oracle objective is disabled."
        )
    if config["ALIGN_MODE"] == "none" and config["ALIGN_DISTANCE"] != "ln_mse":
        raise ValueError(
            "ALIGN_MODE=none must use ALIGN_DISTANCE=ln_mse so the distance-free "
            "baseline is not duplicated."
        )
    if config["ALIGN_TARGET_SHUFFLE"]:
        if config["ALIGN_MODE"] not in {"a_to_c", "c_to_a"}:
            raise ValueError(
                "ALIGN_TARGET_SHUFFLE=true is only valid for ALIGN_MODE=a_to_c "
                "or ALIGN_MODE=c_to_a."
            )
        if config["ALIGN_TARGET_SHUFFLE_SCOPE"] != "same_agent_env_time":
            raise ValueError(
                "The confirmatory protocol only supports "
                "ALIGN_TARGET_SHUFFLE_SCOPE=same_agent_env_time."
            )
        if config["ALIGN_DISTANCE"] != "ln_mse":
            raise ValueError(
                "Shuffled controls belong to the LN-MSE protocol and cannot be "
                "combined with a relational alignment distance."
            )
    if config["ALIGN_GRADIENT_CALIBRATION"]:
        if not config["MATCHED_COMPARISON"]:
            raise ValueError("Gradient calibration requires MATCHED_COMPARISON=true.")
        if config["ALIGN_MODE"] not in {"a_to_c", "c_to_a"}:
            raise ValueError(
                "Gradient calibration requires ALIGN_MODE=a_to_c or c_to_a."
            )
        if config["ALIGN_TARGET_SHUFFLE"]:
            raise ValueError("Gradient calibration does not support shuffled targets.")
        if float(config["ALIGNMENT_COEF"]) != 0.0:
            raise ValueError(
                "Gradient calibration requires ALIGNMENT_COEF=0 so the probe does "
                "not alter the RL update."
            )
        if config["NUM_UPDATES"] != 1 or config["UPDATE_EPOCHS"] != 1:
            raise ValueError(
                "Gradient calibration requires exactly one rollout/update epoch."
            )
        if float(config["LR"]) != 0.0:
            raise ValueError(
                "Gradient calibration requires LR=0 to hold parameters fixed across "
                "the rollout's minibatches."
            )
    if (
        not config["ACTOR_PARAMETER_SHARING"] or config["MATCHED_COMPARISON"]
    ) and config["NUM_ENVS"] % config["NUM_MINIBATCHES"] != 0:
        raise ValueError(
            "NUM_ENVS must be divisible by NUM_MINIBATCHES for independent actors "
            "or the matched comparison protocol."
        )
    config["CLIP_EPS"] = (
        config["CLIP_EPS"] / env.num_agents
        if config["SCALE_CLIP_EPS"]
        else config["CLIP_EPS"]
    )

    env = SMAXWorldStateWrapper(env, config["OBS_WITH_AGENT_ID"])
    env = SMAXLogWrapper(env)

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        actor_network = ActorRNN(env.action_space(env.agents[0]).n, config=config)
        critic_network = CriticRNN(config=config)
        recovery_head = ScoreRecoveryHead(
            action_dim=env.action_space(env.agents[0]).n,
            latent_dim=config["GRU_HIDDEN_DIM"],
        )
        rng, _rng_actor, _rng_critic = jax.random.split(rng, 3)
        ac_init_x = (
            jnp.zeros(
                (1, config["NUM_ENVS"], env.observation_space(env.agents[0]).shape[0])
            ),
            jnp.zeros((1, config["NUM_ENVS"])),
            jnp.zeros((1, config["NUM_ENVS"], env.action_space(env.agents[0]).n)),
        )
        ac_init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]
        )
        if not config["ACTOR_PARAMETER_SHARING"] and not config["MATCHED_COMPARISON"]:
            actor_rngs = jax.random.split(_rng_actor, env.num_agents)
            actor_network_params = jax.vmap(
                actor_network.init, in_axes=(0, None, None)
            )(actor_rngs, ac_init_hstate, ac_init_x)
        else:
            actor_network_params = actor_network.init(
                _rng_actor, ac_init_hstate, ac_init_x
            )
            if not config["ACTOR_PARAMETER_SHARING"]:
                actor_network_params = jax.tree.map(
                    lambda x: jnp.repeat(x[None, ...], env.num_agents, axis=0),
                    actor_network_params,
                )
        cr_init_x = (
            jnp.zeros(
                (
                    1,
                    config["NUM_ENVS"],
                    env.world_state_size(),
                )
            ),
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        cr_init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]
        )
        critic_network_params = critic_network.init(
            _rng_critic, cr_init_hstate, cr_init_x
        )
        if config["SCORE_RECOVERY"]:
            # Fold in a new key so the actor and value-network initializations
            # remain identical to the seed-matched isolated MAPPO condition.
            recovery_params = recovery_head.init(
                jax.random.fold_in(_rng_critic, 60113),
                jnp.zeros((1, config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])),
                jnp.zeros((1, config["NUM_ENVS"]), dtype=jnp.int32),
            )
            critic_network_params = flax.core.unfreeze(critic_network_params)
            critic_network_params["params"]["ScoreRecoveryHead_0"] = (
                flax.core.unfreeze(recovery_params)["params"]
            )
            critic_network_params = flax.core.freeze(critic_network_params)

        if config["ANNEAL_LR"]:
            base_actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            base_actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        actor_tx = (
            base_actor_tx
            if config["ACTOR_PARAMETER_SHARING"]
            else vmapped_optimizer(base_actor_tx)
        )
        actor_train_state = TrainState.create(
            apply_fn=actor_network.apply,
            params=actor_network_params,
            tx=actor_tx,
        )
        critic_train_state = TrainState.create(
            apply_fn=critic_network.apply,
            params=critic_network_params,
            tx=critic_tx,
        )

        if checkpoint_callback is not None:
            io_callback(
                checkpoint_callback,
                jax.ShapeDtypeStruct((), jnp.int32),
                actor_train_state.params,
                critic_train_state.params,
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(False),
                jnp.asarray(True),
                ordered=True,
            )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        ac_init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"]
        )
        cr_init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"]
        )

        def collect_oracle_reference(train_states, reference_rng):
            """Collect a fresh, independent, high-sample reference rollout.

            The actor and critic parameters are frozen at the beginning of the
            PPO update.  The horizon guarantees at least
            ``ORACLE_REFERENCE_MULTIPLIER * NUM_STEPS`` complete transitions
            per environment: an unfinished suffix is shorter than
            ``env.max_steps + 1`` and is removed below.
            """

            actor_train_state, critic_train_state = train_states
            reference_rng, reset_key = jax.random.split(reference_rng)
            reset_rngs = jax.random.split(reset_key, config["NUM_ENVS"])
            reference_obs, reference_env_state = jax.vmap(
                env.reset, in_axes=(0,)
            )(reset_rngs)
            reference_actor_hstate = ScannedRNN.initialize_carry(
                config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"]
            )
            reference_critic_hstate = ScannedRNN.initialize_carry(
                config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"]
            )
            reference_initial_actor_hstate = reference_actor_hstate
            reference_last_done = jnp.zeros((config["NUM_ACTORS"]), dtype=bool)

            def reference_env_step(reference_state, unused):
                (
                    reference_env_state,
                    last_obs,
                    last_done,
                    actor_hstate,
                    critic_hstate,
                    step_rng,
                ) = reference_state

                step_rng, action_key = jax.random.split(step_rng)
                avail_actions = jax.vmap(env.get_avail_actions)(
                    reference_env_state.env_state
                )
                avail_actions = jax.lax.stop_gradient(
                    batchify(avail_actions, env.agents, config["NUM_ACTORS"])
                )
                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                actor_hstate_by_agent = actor_hstate.reshape(
                    (
                        env.num_agents,
                        config["NUM_ENVS"],
                        config["GRU_HIDDEN_DIM"],
                    )
                )
                actor_inputs = (
                    obs_batch.reshape((env.num_agents, config["NUM_ENVS"], -1))[
                        :, None, ...
                    ],
                    last_done.reshape((env.num_agents, config["NUM_ENVS"]))[
                        :, None, ...
                    ],
                    avail_actions.reshape((env.num_agents, config["NUM_ENVS"], -1)),
                )
                actor_rngs = jax.random.split(action_key, env.num_agents)

                def apply_and_sample(params, hidden, inputs, sample_rng):
                    hidden, pi, _ = actor_network.apply(params, hidden, inputs)
                    action = pi.sample(seed=sample_rng)
                    return hidden, action, pi.log_prob(action)

                actor_hstate, action, log_prob = jax.vmap(
                    apply_and_sample,
                    in_axes=(0, 0, 0, 0),
                )(
                    actor_train_state.params,
                    actor_hstate_by_agent,
                    actor_inputs,
                    actor_rngs,
                )
                action = action.reshape((1, config["NUM_ACTORS"]))
                log_prob = log_prob.reshape((1, config["NUM_ACTORS"]))
                actor_hstate = actor_hstate.reshape(
                    (config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"])
                )

                world_state = last_obs["world_state"].swapaxes(0, 1).reshape(
                    (config["NUM_ACTORS"], -1)
                )
                critic_hstate, baseline_value, _ = critic_network.apply(
                    critic_train_state.params,
                    critic_hstate,
                    (world_state[None, :], last_done[None, :]),
                )
                baseline_value = baseline_value.squeeze()
                if config["ORACLE_REFERENCE_BASELINE"] == "zero":
                    baseline_value = jnp.zeros_like(baseline_value)

                env_action = unbatchify(
                    action, env.agents, config["NUM_ENVS"], env.num_agents
                )
                env_action = {key: value.squeeze() for key, value in env_action.items()}
                step_rng, environment_key = jax.random.split(step_rng)
                environment_rngs = jax.random.split(
                    environment_key, config["NUM_ENVS"]
                )
                next_obs, next_env_state, reward, done, _ = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(environment_rngs, reference_env_state, env_action)
                done_batch = batchify(
                    done, env.agents, config["NUM_ACTORS"]
                ).squeeze()
                alive_mask = jnp.sum(avail_actions, axis=-1) > 1
                transition = OracleReferenceTransition(
                    jnp.tile(done["__all__"], env.num_agents),
                    last_done,
                    action.squeeze(),
                    batchify(
                        reward, env.agents, config["NUM_ACTORS"]
                    ).squeeze(),
                    baseline_value,
                    log_prob.squeeze(),
                    obs_batch,
                    alive_mask,
                    avail_actions,
                )
                next_state = (
                    next_env_state,
                    next_obs,
                    done_batch,
                    actor_hstate,
                    critic_hstate,
                    step_rng,
                )
                return next_state, transition

            reference_state = (
                reference_env_state,
                reference_obs,
                reference_last_done,
                reference_actor_hstate,
                reference_critic_hstate,
                reference_rng,
            )
            _, reference_traj = jax.lax.scan(
                reference_env_step,
                reference_state,
                None,
                config["ORACLE_REFERENCE_HORIZON"],
            )
            reference_traj = jax.tree.map(jax.lax.stop_gradient, reference_traj)
            return reference_initial_actor_hstate, reference_traj

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state

            def _env_step(runner_state, unused):
                train_states, env_state, last_obs, last_done, hstates, rng = (
                    runner_state
                )

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(
                    batchify(avail_actions, env.agents, config["NUM_ACTORS"])
                )
                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                ac_in = (
                    obs_batch[np.newaxis, :],
                    last_done[np.newaxis, :],
                    avail_actions,
                )
                if (
                    config["ACTOR_PARAMETER_SHARING"]
                    and not config["MATCHED_COMPARISON"]
                ):
                    ac_hstate, pi, actor_latent = actor_network.apply(
                        train_states[0].params, hstates[0], ac_in
                    )
                    action = pi.sample(seed=_rng)
                    log_prob = pi.log_prob(action)
                else:
                    actor_hstate = hstates[0].reshape(
                        (
                            env.num_agents,
                            config["NUM_ENVS"],
                            config["GRU_HIDDEN_DIM"],
                        )
                    )
                    actor_in = (
                        obs_batch.reshape((env.num_agents, config["NUM_ENVS"], -1))[
                            :, None, ...
                        ],
                        last_done.reshape((env.num_agents, config["NUM_ENVS"]))[
                            :, None, ...
                        ],
                        avail_actions.reshape((env.num_agents, config["NUM_ENVS"], -1)),
                    )
                    actor_rngs = jax.random.split(_rng, env.num_agents)

                    def apply_and_sample(params, hidden, inputs, sample_rng):
                        hidden, pi, latent = actor_network.apply(params, hidden, inputs)
                        action = pi.sample(seed=sample_rng)
                        return hidden, action, pi.log_prob(action), latent

                    parameter_axis = None if config["ACTOR_PARAMETER_SHARING"] else 0
                    ac_hstate, action, log_prob, actor_latent = jax.vmap(
                        apply_and_sample,
                        in_axes=(parameter_axis, 0, 0, 0),
                    )(
                        train_states[0].params,
                        actor_hstate,
                        actor_in,
                        actor_rngs,
                    )
                    action = action.reshape((1, config["NUM_ACTORS"]))
                    log_prob = log_prob.reshape((1, config["NUM_ACTORS"]))
                    actor_latent = actor_latent.reshape(
                        (1, config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"])
                    )
                    ac_hstate = ac_hstate.reshape(
                        (config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"])
                    )
                env_act = unbatchify(
                    action, env.agents, config["NUM_ENVS"], env.num_agents
                )
                env_act = {k: v.squeeze() for k, v in env_act.items()}

                # VALUE
                # output of wrapper is (num_envs, num_agents, world_state_size)
                # swap axes to (num_agents, num_envs, world_state_size) before reshaping to (num_actors, world_state_size)
                world_state = last_obs["world_state"].swapaxes(0, 1)
                world_state = world_state.reshape((config["NUM_ACTORS"], -1))

                cr_in = (
                    world_state[None, :],
                    last_done[np.newaxis, :],
                )
                cr_hstate, value, critic_latent = critic_network.apply(
                    train_states[1].params, hstates[1], cr_in
                )

                # Unit type belongs to the pre-step observation/latent. Capture
                # it before env.step can autoreset a completed SMACv2 episode.
                ally_unit_types = (
                    env_state.env_state.state.unit_types[:, : env.num_agents]
                    .swapaxes(0, 1)
                    .reshape((config["NUM_ACTORS"],))
                )

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)
                info = jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()
                alive_mask = jnp.sum(avail_actions, axis=-1) > 1
                transition = Transition(
                    jnp.tile(done["__all__"], env.num_agents),
                    last_done,
                    action.squeeze(),
                    value.squeeze(),
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob.squeeze(),
                    obs_batch,
                    world_state,
                    jax.lax.stop_gradient(actor_latent.squeeze(axis=0)),
                    jax.lax.stop_gradient(critic_latent.squeeze(axis=0)),
                    alive_mask,
                    alive_mask,
                    ally_unit_types,
                    info,
                    avail_actions,
                )
                runner_state = (
                    train_states,
                    env_state,
                    obsv,
                    done_batch,
                    (ac_hstate, cr_hstate),
                    rng,
                )
                return runner_state, transition

            initial_hstates = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            _, _, shuffle_audit = maybe_shuffle_targets_within_agent(
                traj_batch.actor_latent_old,
                traj_batch.alive_mask,
                False,
                config["SEED"],
                update_steps,
                env.num_agents,
                config["NUM_ENVS"],
                config["ALIGN_SHUFFLE_SEED_OFFSET"],
            )
            if config["ALIGN_TARGET_SHUFFLE"]:
                if config["ALIGN_MODE"] == "a_to_c":
                    shuffled_target, alignment_mask, shuffle_audit = (
                        maybe_shuffle_targets_within_agent(
                            traj_batch.actor_latent_old,
                            traj_batch.alive_mask,
                            True,
                            config["SEED"],
                            update_steps,
                            env.num_agents,
                            config["NUM_ENVS"],
                            config["ALIGN_SHUFFLE_SEED_OFFSET"],
                        )
                    )
                    traj_batch = traj_batch._replace(
                        actor_latent_old=shuffled_target,
                        alignment_mask=alignment_mask,
                    )
                else:
                    shuffled_target, alignment_mask, shuffle_audit = (
                        maybe_shuffle_targets_within_agent(
                            traj_batch.critic_latent_old,
                            traj_batch.alive_mask,
                            True,
                            config["SEED"],
                            update_steps,
                            env.num_agents,
                            config["NUM_ENVS"],
                            config["ALIGN_SHUFFLE_SEED_OFFSET"],
                        )
                    )
                    traj_batch = traj_batch._replace(
                        critic_latent_old=shuffled_target,
                        alignment_mask=alignment_mask,
                    )

            # CALCULATE ADVANTAGE
            train_states, env_state, last_obs, last_done, hstates, rng = runner_state

            last_world_state = last_obs["world_state"].swapaxes(0, 1)
            last_world_state = last_world_state.reshape((config["NUM_ACTORS"], -1))

            cr_in = (
                last_world_state[None, :],
                last_done[np.newaxis, :],
            )
            _, last_val, _ = critic_network.apply(
                train_states[1].params, hstates[1], cr_in
            )
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.global_done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)
            advantages = jax.lax.stop_gradient(advantages)

            if config["SCORE_RECOVERY"]:
                # The actors are still the frozen pre-update rollout actors.
                # Compute one score/Fisher target for the full rollout, then
                # carry that fixed target through every PPO epoch/minibatch.
                def agent_first(x):
                    return jnp.swapaxes(
                        x.reshape(
                            (
                                config["NUM_STEPS"],
                                env.num_agents,
                                config["NUM_ENVS"],
                                *x.shape[2:],
                            )
                        ),
                        0,
                        1,
                    )

                old_actor_params = train_states[0].params["params"]
                scores_by_agent = jax.vmap(categorical_latent_score)(
                    agent_first(traj_batch.actor_latent_old),
                    agent_first(traj_batch.action),
                    agent_first(traj_batch.avail_actions),
                    old_actor_params["Dense_1"],
                    old_actor_params["Dense_2"],
                )
                target_by_agent, recovery_target_audit = whiten_rollout_scores(
                    scores_by_agent,
                    agent_first(traj_batch.alive_mask),
                    config["SCORE_RECOVERY_FISHER_RIDGE"],
                )
                recovery_target = target_by_agent.swapaxes(0, 1).reshape(
                    (
                        config["NUM_STEPS"],
                        config["NUM_ACTORS"],
                        config["GRU_HIDDEN_DIM"],
                    )
                )
            else:
                # One dummy channel avoids carrying a full 128-D target in
                # every pre-existing (score-recovery-disabled) condition.
                recovery_target = jnp.zeros_like(
                    traj_batch.actor_latent_old[..., :1]
                )
                zero_agent = jnp.zeros(
                    (env.num_agents,), dtype=recovery_target.dtype
                )
                recovery_target_audit = {
                    "valid_count_per_agent": zero_agent,
                    "fisher_min_eigenvalue": zero_agent,
                    "fisher_max_eigenvalue": zero_agent,
                    "target_energy_per_agent": zero_agent,
                }
            recovery_target = jax.lax.stop_gradient(recovery_target)

            if config["ORACLE_LATENT_DISTORTION"]:
                # Derive an independent deterministic stream without consuming
                # the PPO runner RNG.  Matched baseline and oracle runs therefore
                # keep the same PPO sampling stream until their policies diverge.
                reference_rng = jax.random.fold_in(
                    rng,
                    jnp.asarray(
                        config["ORACLE_REFERENCE_SEED_OFFSET"], dtype=jnp.uint32
                    )
                    + update_steps.astype(jnp.uint32),
                )
                reference_initial_hstate, reference_traj = (
                    collect_oracle_reference(train_states, reference_rng)
                )
                reference_returns, reference_valid = complete_mc_return_to_go(
                    reference_traj.reward,
                    reference_traj.global_done,
                    config["GAMMA"],
                )
                if (
                    config["ORACLE_REFERENCE_BASELINE"]
                    == "crossfit_linear_critic"
                ):
                    fitted_baseline = crossfit_linear_reference_baseline(
                        reference_returns,
                        reference_traj.baseline,
                        reference_valid & reference_traj.alive_mask,
                        env.num_agents,
                        config["NUM_ENVS"],
                    )
                    reference_traj = reference_traj._replace(
                        baseline=fitted_baseline
                    )
                reference_advantages = (
                    reference_returns - reference_traj.baseline
                )
            else:
                # Static dummy data keep the ordinary MAPPO code path unchanged
                # and avoid collecting any extra environment interactions.
                reference_initial_hstate = initial_hstates[0]
                reference_traj = OracleReferenceTransition(
                    traj_batch.global_done,
                    traj_batch.done,
                    traj_batch.action,
                    traj_batch.reward,
                    jnp.zeros_like(traj_batch.value),
                    traj_batch.log_prob,
                    traj_batch.obs,
                    traj_batch.alive_mask,
                    traj_batch.avail_actions,
                )
                reference_returns = jnp.zeros_like(traj_batch.reward)
                reference_advantages = jnp.zeros_like(traj_batch.reward)
                reference_valid = jnp.zeros_like(
                    traj_batch.global_done, dtype=bool
                )

            reference_returns = jax.lax.stop_gradient(reference_returns)
            reference_advantages = jax.lax.stop_gradient(reference_advantages)
            reference_valid = jax.lax.stop_gradient(reference_valid)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_states, batch_info):
                    actor_train_state, critic_train_state = train_states
                    (
                        ac_init_hstate,
                        cr_init_hstate,
                        traj_batch,
                        advantages,
                        targets,
                        reference_initial_hstate,
                        reference_traj,
                        reference_returns,
                        reference_advantages,
                        reference_valid,
                        recovery_target,
                    ) = batch_info

                    def _actor_loss_fn(actor_params, init_hstate, traj_batch, gae):
                        # RERUN NETWORK
                        _, pi, actor_latent = actor_network.apply(
                            actor_params,
                            init_hstate.squeeze(),
                            (traj_batch.obs, traj_batch.done, traj_batch.avail_actions),
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE ACTOR LOSS
                        logratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(logratio)
                        if not config["MATCHED_COMPARISON"]:
                            gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        # debug
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])

                        actor_loss = loss_actor - config["ENT_COEF"] * entropy

                        if config["ORACLE_LATENT_DISTORTION"]:
                            policy_params = actor_params["params"]
                            actor_score = categorical_latent_score(
                                actor_latent,
                                traj_batch.action,
                                traj_batch.avail_actions,
                                policy_params["Dense_1"],
                                policy_params["Dense_2"],
                            )
                        else:
                            actor_score = jnp.zeros_like(actor_latent)

                        return actor_loss, (
                            loss_actor,
                            entropy,
                            ratio,
                            approx_kl,
                            clip_frac,
                            actor_latent,
                            actor_score,
                        )

                    def _reference_score_fn(
                        actor_params, init_hstate, reference_traj
                    ):
                        _, pi, actor_latent = actor_network.apply(
                            actor_params,
                            init_hstate.squeeze(),
                            (
                                reference_traj.obs,
                                reference_traj.done,
                                reference_traj.avail_actions,
                            ),
                        )
                        current_log_prob = pi.log_prob(reference_traj.action)
                        importance_weight = jnp.exp(
                            current_log_prob - reference_traj.log_prob
                        )
                        policy_params = actor_params["params"]
                        actor_score = categorical_latent_score(
                            actor_latent,
                            reference_traj.action,
                            reference_traj.avail_actions,
                            policy_params["Dense_1"],
                            policy_params["Dense_2"],
                        )
                        return actor_score, importance_weight

                    def _critic_loss_fn(
                        critic_params, init_hstate, traj_batch, targets,
                        recovery_target,
                    ):
                        # RERUN NETWORK
                        _, value, critic_latent = critic_network.apply(
                            critic_params,
                            init_hstate.squeeze(),
                            (traj_batch.world_state, traj_batch.done),
                        )

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )
                        recovery_loss = jnp.zeros((), dtype=value_loss.dtype)
                        if config["SCORE_RECOVERY"]:
                            prediction = recovery_head.apply(
                                {
                                    "params": critic_params["params"][
                                        "ScoreRecoveryHead_0"
                                    ]
                                },
                                critic_latent,
                                traj_batch.action,
                            )
                            valid = traj_batch.alive_mask.astype(value_loss.dtype)
                            squared_error = jnp.sum(
                                jnp.square(prediction - recovery_target), axis=-1
                            )
                            # Agent-major stratified minibatches contain the
                            # same number of environments per agent. Average
                            # valid transitions within each agent, then give
                            # each agent equal weight as in (1/N) sum_i E_i.
                            environments_per_agent = (
                                valid.shape[1] // env.num_agents
                            )
                            valid_by_agent = valid.reshape(
                                (valid.shape[0], env.num_agents, environments_per_agent)
                            )
                            error_by_agent = squared_error.reshape(
                                (valid.shape[0], env.num_agents, environments_per_agent)
                            )
                            valid_count = valid_by_agent.sum(axis=(0, 2))
                            agent_loss = (
                                error_by_agent * valid_by_agent
                            ).sum(axis=(0, 2)) / jnp.maximum(valid_count, 1.0)
                            recovery_loss = agent_loss.mean()
                        critic_loss = (
                            config["VF_COEF"] * value_loss
                            + config["SCORE_RECOVERY_COEF"] * recovery_loss
                        )
                        return critic_loss, (value_loss, critic_latent, recovery_loss)

                    def split_agents(x):
                        x = x.reshape(
                            (
                                x.shape[0],
                                env.num_agents,
                                -1,
                                *x.shape[2:],
                            )
                        )
                        return jnp.swapaxes(x, 0, 1)

                    actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                    if config["MATCHED_COMPARISON"]:
                        actor_advantages = (advantages - advantages.mean()) / (
                            advantages.std() + 1e-8
                        )
                    else:
                        actor_advantages = advantages

                    if config["MATCHED_COMPARISON"]:
                        actor_batch = jax.tree.map(split_agents, traj_batch)
                        actor_hstates = split_agents(ac_init_hstate)
                        actor_advantages = split_agents(actor_advantages)
                        reference_actor_batch = jax.tree.map(
                            split_agents, reference_traj
                        )
                        reference_actor_hstates = split_agents(
                            reference_initial_hstate
                        )
                        reference_returns_by_agent = split_agents(
                            reference_returns
                        )
                        reference_advantages_by_agent = split_agents(
                            reference_advantages
                        )
                        reference_valid_by_agent = split_agents(reference_valid)
                        parameter_axis = (
                            None if config["ACTOR_PARAMETER_SHARING"] else 0
                        )

                        def matched_total_loss(
                            actor_params,
                            critic_params,
                            distance_name=config["ALIGN_DISTANCE"],
                        ):
                            actor_losses, actor_aux = jax.vmap(
                                _actor_loss_fn,
                                in_axes=(parameter_axis, 0, 0, 0),
                            )(
                                actor_params,
                                actor_hstates,
                                actor_batch,
                                actor_advantages,
                            )
                            critic_rl_loss, critic_aux = _critic_loss_fn(
                                critic_params,
                                cr_init_hstate,
                                traj_batch,
                                targets,
                                recovery_target,
                            )

                            actor_latent = (
                                actor_aux[5]
                                .swapaxes(0, 1)
                                .reshape(traj_batch.actor_latent_old.shape)
                            )
                            critic_latent = critic_aux[1]
                            mask = traj_batch.alignment_mask
                            zero = jnp.zeros((), dtype=actor_latent.dtype)
                            containment_statistics = {
                                "similarity": zero,
                                "source_effective_rank": zero,
                                "valid_samples": zero,
                                "valid_groups": zero,
                            }
                            distance_kwargs = {
                                "distance_name": distance_name,
                                "num_agent_groups": env.num_agents,
                                "epsilon": config["ALIGN_DISTANCE_EPS"],
                                "containment_ridge_ratio": config[
                                    "ALIGN_CONTAINMENT_RIDGE_RATIO"
                                ],
                                "containment_epsilon": config[
                                    "ALIGN_CONTAINMENT_EPS"
                                ],
                                "group_labels": (
                                    traj_batch.unit_type
                                    if config[
                                        "ALIGN_CONTAINMENT_GROUP_BY_UNIT_TYPE"
                                    ]
                                    else None
                                ),
                                "num_label_groups": env.unit_type_bits,
                                "min_group_samples": config[
                                    "ALIGN_CONTAINMENT_MIN_GROUP_SAMPLES"
                                ],
                            }
                            if distance_name == "containment":
                                c_to_a_loss = zero
                                a_to_c_loss = zero
                                joint_loss = zero
                                if config["ALIGN_MODE"] in {"c_to_a", "reciprocal"}:
                                    containment_statistics = representation_distance(
                                        actor_latent,
                                        jax.lax.stop_gradient(
                                            traj_batch.critic_latent_old
                                        ),
                                        mask,
                                        return_statistics=True,
                                        **distance_kwargs,
                                    )
                                    c_to_a_loss = containment_statistics["loss"]
                                if config["ALIGN_MODE"] in {"a_to_c", "reciprocal"}:
                                    reverse_statistics = representation_distance(
                                        critic_latent,
                                        jax.lax.stop_gradient(
                                            traj_batch.actor_latent_old
                                        ),
                                        mask,
                                        return_statistics=True,
                                        **distance_kwargs,
                                    )
                                    a_to_c_loss = reverse_statistics["loss"]
                                    if config["ALIGN_MODE"] == "a_to_c":
                                        containment_statistics = reverse_statistics
                                if config["ALIGN_MODE"] == "joint":
                                    containment_statistics = representation_distance(
                                        actor_latent,
                                        critic_latent,
                                        mask,
                                        return_statistics=True,
                                        **distance_kwargs,
                                    )
                                    joint_loss = containment_statistics["loss"]
                            else:
                                c_to_a_loss = representation_distance(
                                    actor_latent,
                                    jax.lax.stop_gradient(
                                        traj_batch.critic_latent_old
                                    ),
                                    mask,
                                    **distance_kwargs,
                                )
                                a_to_c_loss = representation_distance(
                                    critic_latent,
                                    jax.lax.stop_gradient(
                                        traj_batch.actor_latent_old
                                    ),
                                    mask,
                                    **distance_kwargs,
                                )
                                joint_loss = representation_distance(
                                    actor_latent,
                                    critic_latent,
                                    mask,
                                    **distance_kwargs,
                                )
                            actor_alignment = zero
                            critic_alignment = zero
                            joint_alignment = zero
                            if config["ALIGN_MODE"] == "c_to_a":
                                actor_alignment = c_to_a_loss
                            elif config["ALIGN_MODE"] == "a_to_c":
                                critic_alignment = a_to_c_loss
                            elif config["ALIGN_MODE"] == "reciprocal":
                                actor_alignment = c_to_a_loss
                                critic_alignment = a_to_c_loss
                            elif config["ALIGN_MODE"] == "joint":
                                joint_alignment = joint_loss

                            alignment_objective = (
                                actor_alignment + critic_alignment + joint_alignment
                            )
                            oracle_zero = jnp.zeros((), dtype=actor_latent.dtype)
                            oracle_statistics = {
                                "epsilon_lat": oracle_zero,
                                "epsilon_lat_per_agent": jnp.zeros(
                                    (env.num_agents,), dtype=actor_latent.dtype
                                ),
                                "valid_samples": oracle_zero,
                                "valid_samples_per_agent": jnp.zeros(
                                    (env.num_agents,), dtype=actor_latent.dtype
                                ),
                                "reference_gradient_norm": oracle_zero,
                                "critic_gradient_norm": oracle_zero,
                                "reference_critic_gradient_cosine": oracle_zero,
                                "reference_valid_samples": oracle_zero,
                                "reference_valid_samples_per_agent": jnp.zeros(
                                    (env.num_agents,), dtype=actor_latent.dtype
                                ),
                                "critic_valid_samples": oracle_zero,
                                "critic_valid_samples_per_agent": jnp.zeros(
                                    (env.num_agents,), dtype=actor_latent.dtype
                                ),
                                "reference_to_critic_sample_ratio": oracle_zero,
                                "reference_mc_return_std": oracle_zero,
                                "reference_baselined_advantage_std": oracle_zero,
                                "reference_baseline_variance_reduction": oracle_zero,
                            }
                            if config["ORACLE_LATENT_DISTORTION"]:
                                (
                                    reference_scores,
                                    reference_importance_weight,
                                ) = jax.vmap(
                                    _reference_score_fn,
                                    in_axes=(parameter_axis, 0, 0),
                                )(
                                    actor_params,
                                    reference_actor_hstates,
                                    reference_actor_batch,
                                )
                                critic_scores = actor_aux[6]
                                critic_importance_weight = actor_aux[2]
                                critic_advantage = split_agents(advantages)
                                critic_mask = split_agents(traj_batch.alive_mask)
                                reference_mask = (
                                    reference_actor_batch.alive_mask
                                    & reference_valid_by_agent
                                )
                                oracle_statistics = oracle_latent_distortion(
                                    reference_scores,
                                    reference_advantages_by_agent,
                                    reference_returns_by_agent,
                                    reference_mask,
                                    reference_importance_weight,
                                    critic_scores,
                                    critic_advantage,
                                    critic_mask,
                                    critic_importance_weight,
                                    config["ORACLE_FISHER_RIDGE"],
                                )
                            total_loss = (
                                actor_losses.mean()
                                + critic_rl_loss
                                + config["ALIGNMENT_COEF"] * alignment_objective
                                + config["ORACLE_DISTORTION_COEF"]
                                * oracle_statistics["epsilon_lat"]
                                / env.num_agents
                            )
                            return total_loss, (
                                actor_losses,
                                actor_aux,
                                critic_rl_loss,
                                critic_aux,
                                c_to_a_loss,
                                a_to_c_loss,
                                joint_loss,
                                actor_alignment,
                                critic_alignment,
                                joint_alignment,
                                containment_statistics,
                                oracle_statistics,
                            )

                        (
                            (combined_total_loss, matched_aux),
                            (actor_grads, critic_grads),
                        ) = jax.value_and_grad(
                            matched_total_loss,
                            argnums=(0, 1),
                            has_aux=True,
                        )(
                            actor_train_state.params,
                            critic_train_state.params,
                        )

                        if (
                            (
                                config["ALIGN_MODE"] == "none"
                                or config["ALIGNMENT_COEF"] == 0
                            )
                            and not config["ORACLE_LATENT_DISTORTION"]
                            and not config["SCORE_RECOVERY"]
                        ):
                            actor_cross_grads = jax.tree.map(
                                jnp.zeros_like, actor_grads
                            )
                            critic_cross_grads = jax.tree.map(
                                jnp.zeros_like, critic_grads
                            )
                        elif config["SCORE_RECOVERY"]:
                            # This intervention has no actor-side auxiliary
                            # objective. Differentiate only the critic's
                            # recovery term; no actor parameters are even
                            # present in this computation graph.
                            actor_cross_grads = jax.tree.map(
                                jnp.zeros_like, actor_grads
                            )

                            def critic_recovery_objective(critic_params):
                                _, auxiliary = _critic_loss_fn(
                                    critic_params,
                                    cr_init_hstate,
                                    traj_batch,
                                    targets,
                                    recovery_target,
                                )
                                return config["SCORE_RECOVERY_COEF"] * auxiliary[2]

                            critic_cross_grads = jax.grad(critic_recovery_objective)(
                                critic_train_state.params
                            )
                        else:

                            def matched_cross_objective(actor_params, critic_params):
                                _, auxiliary = matched_total_loss(
                                    actor_params, critic_params
                                )
                                return (
                                    config["ALIGNMENT_COEF"]
                                    * (auxiliary[7] + auxiliary[8] + auxiliary[9])
                                    + config["ORACLE_DISTORTION_COEF"]
                                    * auxiliary[11]["epsilon_lat"]
                                    / env.num_agents
                                    + config["SCORE_RECOVERY_COEF"]
                                    * auxiliary[3][2]
                                )

                            actor_cross_grads, critic_cross_grads = jax.grad(
                                matched_cross_objective,
                                argnums=(0, 1),
                            )(
                                actor_train_state.params,
                                critic_train_state.params,
                            )
                        if config["ALIGN_GRADIENT_CALIBRATION"]:

                            def calibration_cross_grads(distance_name):
                                def objective(actor_params, critic_params):
                                    _, auxiliary = matched_total_loss(
                                        actor_params,
                                        critic_params,
                                        distance_name,
                                    )
                                    return auxiliary[7] + auxiliary[8] + auxiliary[9]

                                return jax.grad(objective, argnums=(0, 1))(
                                    actor_train_state.params,
                                    critic_train_state.params,
                                )

                            (
                                calibration_ln_actor_grads,
                                calibration_ln_critic_grads,
                            ) = calibration_cross_grads("ln_mse")
                            (
                                calibration_cka_actor_grads,
                                calibration_cka_critic_grads,
                            ) = calibration_cross_grads("linear_cka")
                            (
                                calibration_containment_actor_grads,
                                calibration_containment_critic_grads,
                            ) = calibration_cross_grads("containment")
                        if not config["ACTOR_PARAMETER_SHARING"]:
                            actor_grads = jax.tree.map(
                                lambda x: x * env.num_agents, actor_grads
                            )
                            actor_cross_grads = jax.tree.map(
                                lambda x: x * env.num_agents,
                                actor_cross_grads,
                            )
                            if config["ALIGN_GRADIENT_CALIBRATION"]:
                                calibration_ln_actor_grads = jax.tree.map(
                                    lambda x: x * env.num_agents,
                                    calibration_ln_actor_grads,
                                )
                                calibration_cka_actor_grads = jax.tree.map(
                                    lambda x: x * env.num_agents,
                                    calibration_cka_actor_grads,
                                )
                                calibration_containment_actor_grads = jax.tree.map(
                                    lambda x: x * env.num_agents,
                                    calibration_containment_actor_grads,
                                )
                        actor_rl_grads = jax.tree.map(
                            lambda total, cross: total - cross,
                            actor_grads,
                            actor_cross_grads,
                        )
                        critic_rl_grads = jax.tree.map(
                            lambda total, cross: total - cross,
                            critic_grads,
                            critic_cross_grads,
                        )
                        actor_loss = (matched_aux[0], matched_aux[1])
                        critic_loss = (matched_aux[2], matched_aux[3])
                        c_to_a_loss = matched_aux[4]
                        a_to_c_loss = matched_aux[5]
                        current_joint_loss = matched_aux[6]
                        actor_alignment_loss = matched_aux[7]
                        critic_alignment_loss = matched_aux[8]
                        joint_alignment_loss = matched_aux[9]
                        containment_statistics = matched_aux[10]
                        oracle_statistics = matched_aux[11]
                    elif config["ACTOR_PARAMETER_SHARING"]:
                        actor_loss, actor_grads = actor_grad_fn(
                            actor_train_state.params,
                            ac_init_hstate,
                            traj_batch,
                            actor_advantages,
                        )
                        critic_loss, critic_grads = jax.value_and_grad(
                            _critic_loss_fn, has_aux=True
                        )(
                            critic_train_state.params,
                            cr_init_hstate,
                            traj_batch,
                            targets,
                            recovery_target,
                        )
                        combined_total_loss = actor_loss[0] + critic_loss[0]
                        zero = jnp.zeros((), dtype=advantages.dtype)
                        c_to_a_loss = zero
                        a_to_c_loss = zero
                        current_joint_loss = zero
                        actor_alignment_loss = zero
                        critic_alignment_loss = zero
                        joint_alignment_loss = zero
                        containment_statistics = {
                            "similarity": zero,
                            "source_effective_rank": zero,
                            "valid_samples": zero,
                            "valid_groups": zero,
                        }
                        oracle_statistics = {
                            "epsilon_lat": zero,
                            "epsilon_lat_per_agent": jnp.zeros(
                                (env.num_agents,), dtype=zero.dtype
                            ),
                            "valid_samples": zero,
                            "valid_samples_per_agent": jnp.zeros(
                                (env.num_agents,), dtype=zero.dtype
                            ),
                            "reference_gradient_norm": zero,
                            "critic_gradient_norm": zero,
                            "reference_critic_gradient_cosine": zero,
                            "reference_valid_samples": zero,
                            "reference_valid_samples_per_agent": jnp.zeros(
                                (env.num_agents,), dtype=zero.dtype
                            ),
                            "critic_valid_samples": zero,
                            "critic_valid_samples_per_agent": jnp.zeros(
                                (env.num_agents,), dtype=zero.dtype
                            ),
                            "reference_to_critic_sample_ratio": zero,
                            "reference_mc_return_std": zero,
                            "reference_baselined_advantage_std": zero,
                            "reference_baseline_variance_reduction": zero,
                        }
                        actor_cross_grads = jax.tree.map(jnp.zeros_like, actor_grads)
                        critic_cross_grads = jax.tree.map(jnp.zeros_like, critic_grads)
                        actor_rl_grads = actor_grads
                        critic_rl_grads = critic_grads
                    else:
                        actor_batch = jax.tree.map(split_agents, traj_batch)
                        actor_hstates = split_agents(ac_init_hstate)
                        actor_advantages = split_agents(actor_advantages)
                        actor_loss, actor_grads = jax.vmap(actor_grad_fn)(
                            actor_train_state.params,
                            actor_hstates,
                            actor_batch,
                            actor_advantages,
                        )
                        critic_loss, critic_grads = jax.value_and_grad(
                            _critic_loss_fn, has_aux=True
                        )(
                            critic_train_state.params,
                            cr_init_hstate,
                            traj_batch,
                            targets,
                            recovery_target,
                        )
                        combined_total_loss = actor_loss[0].mean() + critic_loss[0]
                        zero = jnp.zeros((), dtype=advantages.dtype)
                        c_to_a_loss = zero
                        a_to_c_loss = zero
                        current_joint_loss = zero
                        actor_alignment_loss = zero
                        critic_alignment_loss = zero
                        joint_alignment_loss = zero
                        containment_statistics = {
                            "similarity": zero,
                            "source_effective_rank": zero,
                            "valid_samples": zero,
                            "valid_groups": zero,
                        }
                        oracle_statistics = {
                            "epsilon_lat": zero,
                            "epsilon_lat_per_agent": jnp.zeros(
                                (env.num_agents,), dtype=zero.dtype
                            ),
                            "valid_samples": zero,
                            "valid_samples_per_agent": jnp.zeros(
                                (env.num_agents,), dtype=zero.dtype
                            ),
                            "reference_gradient_norm": zero,
                            "critic_gradient_norm": zero,
                            "reference_critic_gradient_cosine": zero,
                            "reference_valid_samples": zero,
                            "reference_valid_samples_per_agent": jnp.zeros(
                                (env.num_agents,), dtype=zero.dtype
                            ),
                            "critic_valid_samples": zero,
                            "critic_valid_samples_per_agent": jnp.zeros(
                                (env.num_agents,), dtype=zero.dtype
                            ),
                            "reference_to_critic_sample_ratio": zero,
                            "reference_mc_return_std": zero,
                            "reference_baselined_advantage_std": zero,
                            "reference_baseline_variance_reduction": zero,
                        }
                        actor_cross_grads = jax.tree.map(jnp.zeros_like, actor_grads)
                        critic_cross_grads = jax.tree.map(jnp.zeros_like, critic_grads)
                        actor_rl_grads = actor_grads
                        critic_rl_grads = critic_grads

                    def actor_tree_norms(tree):
                        if config["ACTOR_PARAMETER_SHARING"]:
                            return jnp.asarray([tree_l2_norm(tree)])
                        return jax.vmap(tree_l2_norm)(tree)

                    actor_grad_norms = actor_tree_norms(actor_grads)
                    actor_rl_grad_norms = actor_tree_norms(actor_rl_grads)
                    actor_cross_grad_norms = actor_tree_norms(actor_cross_grads)
                    critic_grad_norm = tree_l2_norm(critic_grads)
                    critic_rl_grad_norm = tree_l2_norm(critic_rl_grads)
                    critic_cross_grad_norm = tree_l2_norm(critic_cross_grads)
                    if config["ALIGN_GRADIENT_CALIBRATION"]:
                        calibration_ln_actor_grad_norms = actor_tree_norms(
                            calibration_ln_actor_grads
                        )
                        calibration_cka_actor_grad_norms = actor_tree_norms(
                            calibration_cka_actor_grads
                        )
                        calibration_ln_critic_grad_norm = tree_l2_norm(
                            calibration_ln_critic_grads
                        )
                        calibration_cka_critic_grad_norm = tree_l2_norm(
                            calibration_cka_critic_grads
                        )
                        calibration_containment_actor_grad_norms = actor_tree_norms(
                            calibration_containment_actor_grads
                        )
                        calibration_containment_critic_grad_norm = tree_l2_norm(
                            calibration_containment_critic_grads
                        )

                    old_actor_params = actor_train_state.params
                    old_critic_params = critic_train_state.params
                    actor_train_state = actor_train_state.apply_gradients(
                        grads=actor_grads
                    )
                    critic_train_state = critic_train_state.apply_gradients(
                        grads=critic_grads
                    )
                    actor_param_updates = jax.tree.map(
                        lambda new, old: new - old,
                        actor_train_state.params,
                        old_actor_params,
                    )
                    critic_param_updates = jax.tree.map(
                        lambda new, old: new - old,
                        critic_train_state.params,
                        old_critic_params,
                    )
                    if config["ACTOR_PARAMETER_SHARING"]:
                        actor_update_norms = jnp.asarray(
                            [tree_l2_norm(actor_param_updates)]
                        )
                    else:
                        actor_update_norms = jax.vmap(tree_l2_norm)(actor_param_updates)
                    actor_grad_norms_after_clip = jnp.minimum(
                        actor_grad_norms, config["MAX_GRAD_NORM"]
                    )

                    mean_actor_loss = actor_loss[0].mean()
                    loss_info = {
                        "total_loss": combined_total_loss,
                        "actor_loss": mean_actor_loss,
                        "value_loss": critic_loss[1][0],
                        "score_recovery_loss": critic_loss[1][2],
                        "score_recovery_objective_weighted": config[
                            "SCORE_RECOVERY_COEF"
                        ]
                        * critic_loss[1][2],
                        "score_recovery_valid_samples": jnp.where(
                            config["SCORE_RECOVERY"],
                            traj_batch.alive_mask.sum(),
                            0,
                        ),
                        "score_recovery_target_energy_mean": recovery_target_audit[
                            "target_energy_per_agent"
                        ].mean(),
                        "score_recovery_fisher_min_eigenvalue": recovery_target_audit[
                            "fisher_min_eigenvalue"
                        ].min(),
                        "score_recovery_fisher_max_eigenvalue": recovery_target_audit[
                            "fisher_max_eigenvalue"
                        ].max(),
                        "entropy": actor_loss[1][1].mean(),
                        "ratio": actor_loss[1][2],
                        "approx_kl": actor_loss[1][3].mean(),
                        "clip_frac": actor_loss[1][4].mean(),
                        "alignment_c_to_a_distance": c_to_a_loss,
                        "alignment_a_to_c_distance": a_to_c_loss,
                        "alignment_current_joint_distance": current_joint_loss,
                        "c_to_a_loss": c_to_a_loss,
                        "a_to_c_loss": a_to_c_loss,
                        "joint_loss": current_joint_loss,
                        "alignment_actor_objective": actor_alignment_loss,
                        "alignment_critic_objective": critic_alignment_loss,
                        "alignment_joint_objective": joint_alignment_loss,
                        "alignment_objective_weighted": config["ALIGNMENT_COEF"]
                        * (
                            actor_alignment_loss
                            + critic_alignment_loss
                            + joint_alignment_loss
                        ),
                        "alignment_objective": (
                            actor_alignment_loss
                            + critic_alignment_loss
                            + joint_alignment_loss
                        ),
                        "oracle_epsilon_lat": oracle_statistics["epsilon_lat"],
                        "oracle_objective_weighted": config[
                            "ORACLE_DISTORTION_COEF"
                        ]
                        * oracle_statistics["epsilon_lat"],
                        "oracle_valid_samples": oracle_statistics["valid_samples"],
                        "oracle_reference_valid_samples": oracle_statistics[
                            "reference_valid_samples"
                        ],
                        "oracle_critic_valid_samples": oracle_statistics[
                            "critic_valid_samples"
                        ],
                        "oracle_reference_to_critic_sample_ratio": oracle_statistics[
                            "reference_to_critic_sample_ratio"
                        ],
                        "oracle_reference_gradient_norm": oracle_statistics[
                            "reference_gradient_norm"
                        ],
                        "oracle_critic_gradient_norm": oracle_statistics[
                            "critic_gradient_norm"
                        ],
                        "oracle_reference_critic_gradient_cosine": oracle_statistics[
                            "reference_critic_gradient_cosine"
                        ],
                        "oracle_reference_mc_return_std": oracle_statistics[
                            "reference_mc_return_std"
                        ],
                        "oracle_reference_baselined_advantage_std": oracle_statistics[
                            "reference_baselined_advantage_std"
                        ],
                        "oracle_reference_baseline_variance_reduction": oracle_statistics[
                            "reference_baseline_variance_reduction"
                        ],
                        "containment_similarity": containment_statistics[
                            "similarity"
                        ],
                        "containment_source_effective_rank": containment_statistics[
                            "source_effective_rank"
                        ],
                        "containment_valid_samples_mean": containment_statistics[
                            "valid_samples"
                        ],
                        "containment_valid_slot_type_groups": containment_statistics[
                            "valid_groups"
                        ],
                        "actor_rl_grad_norm_mean": actor_rl_grad_norms.mean(),
                        "actor_rl_grad_norm_max": actor_rl_grad_norms.max(),
                        "actor_cross_grad_norm_mean": (actor_cross_grad_norms.mean()),
                        "actor_cross_grad_norm_max": actor_cross_grad_norms.max(),
                        "actor_cross_to_rl_grad_ratio_mean": (
                            actor_cross_grad_norms
                            / jnp.maximum(actor_rl_grad_norms, 1e-12)
                        ).mean(),
                        "oracle_actor_grad_norm_mean": jnp.where(
                            config["ORACLE_LATENT_DISTORTION"],
                            actor_cross_grad_norms.mean(),
                            0.0,
                        ),
                        "oracle_actor_to_rl_grad_ratio_mean": jnp.where(
                            config["ORACLE_LATENT_DISTORTION"],
                            (
                                actor_cross_grad_norms
                                / jnp.maximum(actor_rl_grad_norms, 1e-12)
                            ).mean(),
                            0.0,
                        ),
                        "actor_grad_norm_mean": actor_grad_norms.mean(),
                        "actor_grad_norm_max": actor_grad_norms.max(),
                        "actor_grad_norm_after_clip_mean": (
                            actor_grad_norms_after_clip.mean()
                        ),
                        "actor_grad_norm_after_clip_max": (
                            actor_grad_norms_after_clip.max()
                        ),
                        "actor_grad_clipped_fraction": jnp.mean(
                            actor_grad_norms > config["MAX_GRAD_NORM"]
                        ),
                        "actor_update_norm_mean": actor_update_norms.mean(),
                        "actor_update_norm_max": actor_update_norms.max(),
                        "critic_grad_norm": critic_grad_norm,
                        "critic_rl_grad_norm": critic_rl_grad_norm,
                        "critic_cross_grad_norm": critic_cross_grad_norm,
                        "critic_cross_to_rl_grad_ratio": critic_cross_grad_norm
                        / jnp.maximum(critic_rl_grad_norm, 1e-12),
                        "score_recovery_critic_grad_norm": jnp.where(
                            config["SCORE_RECOVERY"], critic_cross_grad_norm, 0.0
                        ),
                        "score_recovery_critic_to_rl_grad_ratio": jnp.where(
                            config["SCORE_RECOVERY"],
                            critic_cross_grad_norm
                            / jnp.maximum(critic_rl_grad_norm, 1e-12),
                            0.0,
                        ),
                        "score_recovery_actor_grad_norm_max": jnp.where(
                            config["SCORE_RECOVERY"],
                            actor_cross_grad_norms.max(),
                            0.0,
                        ),
                        "critic_grad_norm_after_clip": jnp.minimum(
                            critic_grad_norm, config["MAX_GRAD_NORM"]
                        ),
                        "critic_grad_clipped": (
                            critic_grad_norm > config["MAX_GRAD_NORM"]
                        ),
                        "critic_update_norm": tree_l2_norm(critic_param_updates),
                        "alive_agent_fraction": traj_batch.alive_mask.mean(),
                    }

                    if config["ALIGN_GRADIENT_CALIBRATION"]:
                        loss_info.update(
                            {
                                "calibration_ln_mse_actor_cross_grad_norm_mean": calibration_ln_actor_grad_norms.mean(),
                                "calibration_linear_cka_actor_cross_grad_norm_mean": calibration_cka_actor_grad_norms.mean(),
                                "calibration_ln_mse_actor_cross_to_rl_ratio": (
                                    calibration_ln_actor_grad_norms
                                    / jnp.maximum(actor_rl_grad_norms, 1e-12)
                                ).mean(),
                                "calibration_linear_cka_actor_cross_to_rl_ratio": (
                                    calibration_cka_actor_grad_norms
                                    / jnp.maximum(actor_rl_grad_norms, 1e-12)
                                ).mean(),
                                "calibration_containment_actor_cross_grad_norm_mean": calibration_containment_actor_grad_norms.mean(),
                                "calibration_containment_actor_cross_to_rl_ratio": (
                                    calibration_containment_actor_grad_norms
                                    / jnp.maximum(actor_rl_grad_norms, 1e-12)
                                ).mean(),
                                "calibration_ln_mse_critic_cross_grad_norm": calibration_ln_critic_grad_norm,
                                "calibration_linear_cka_critic_cross_grad_norm": calibration_cka_critic_grad_norm,
                                "calibration_ln_mse_critic_cross_to_rl_ratio": calibration_ln_critic_grad_norm
                                / jnp.maximum(critic_rl_grad_norm, 1e-12),
                                "calibration_linear_cka_critic_cross_to_rl_ratio": calibration_cka_critic_grad_norm
                                / jnp.maximum(critic_rl_grad_norm, 1e-12),
                                "calibration_containment_critic_cross_grad_norm": calibration_containment_critic_grad_norm,
                                "calibration_containment_critic_cross_to_rl_ratio": calibration_containment_critic_grad_norm
                                / jnp.maximum(critic_rl_grad_norm, 1e-12),
                            }
                        )

                    if config["MATCHED_COMPARISON"]:
                        raw_advantages_by_agent = split_agents(advantages)
                        loss_info["advantage_global_mean"] = advantages.mean()
                        loss_info["advantage_global_std"] = advantages.std()
                        for agent_idx in range(env.num_agents):
                            agent_advantages = raw_advantages_by_agent[agent_idx]
                            loss_info[f"advantage_mean_agent_{agent_idx}"] = (
                                agent_advantages.mean()
                            )
                            loss_info[f"advantage_std_agent_{agent_idx}"] = (
                                agent_advantages.std()
                            )
                            loss_info[f"oracle_epsilon_lat_agent_{agent_idx}"] = (
                                oracle_statistics["epsilon_lat_per_agent"][agent_idx]
                            )
                            loss_info[f"oracle_valid_samples_agent_{agent_idx}"] = (
                                oracle_statistics["valid_samples_per_agent"][agent_idx]
                            )
                            loss_info[
                                f"oracle_critic_valid_samples_agent_{agent_idx}"
                            ] = oracle_statistics["critic_valid_samples_per_agent"][
                                agent_idx
                            ]

                    return (actor_train_state, critic_train_state), loss_info

                (
                    train_states,
                    init_hstates,
                    traj_batch,
                    advantages,
                    targets,
                    reference_initial_hstate,
                    reference_traj,
                    reference_returns,
                    reference_advantages,
                    reference_valid,
                    recovery_target,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)

                init_hstates = jax.tree.map(
                    lambda x: jnp.reshape(x, (1, config["NUM_ACTORS"], -1)),
                    init_hstates,
                )
                reference_initial_hstate = jnp.reshape(
                    reference_initial_hstate,
                    (1, config["NUM_ACTORS"], -1),
                )

                batch = (
                    init_hstates[0],
                    init_hstates[1],
                    traj_batch,
                    advantages.squeeze(),
                    targets.squeeze(),
                    reference_initial_hstate,
                    reference_traj,
                    reference_returns.squeeze(),
                    reference_advantages.squeeze(),
                    reference_valid.squeeze(),
                    recovery_target,
                )
                if (
                    config["ACTOR_PARAMETER_SHARING"]
                    and not config["MATCHED_COMPARISON"]
                ):
                    permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])
                    shuffled_batch = jax.tree.map(
                        lambda x: jnp.take(x, permutation, axis=1), batch
                    )
                    minibatches = jax.tree.map(
                        lambda x: jnp.swapaxes(
                            jnp.reshape(
                                x,
                                [x.shape[0], config["NUM_MINIBATCHES"], -1]
                                + list(x.shape[2:]),
                            ),
                            1,
                            0,
                        ),
                        shuffled_batch,
                    )
                else:
                    # Keep every agent represented in every minibatch. Matched
                    # shared and independent runs therefore give the critic the
                    # same samples in the same update order.
                    permutation = jax.random.permutation(_rng, config["NUM_ENVS"])
                    envs_per_minibatch = config["NUM_ENVS"] // config["NUM_MINIBATCHES"]

                    def make_stratified_minibatches(x):
                        x = x.reshape(
                            (
                                x.shape[0],
                                env.num_agents,
                                config["NUM_ENVS"],
                                *x.shape[2:],
                            )
                        )
                        x = jnp.take(x, permutation, axis=2)
                        x = x.reshape(
                            (
                                x.shape[0],
                                env.num_agents,
                                config["NUM_MINIBATCHES"],
                                envs_per_minibatch,
                                *x.shape[3:],
                            )
                        )
                        x = jnp.moveaxis(x, 2, 0)
                        return x.reshape(
                            (
                                config["NUM_MINIBATCHES"],
                                x.shape[1],
                                env.num_agents * envs_per_minibatch,
                                *x.shape[4:],
                            )
                        )

                    minibatches = jax.tree.map(make_stratified_minibatches, batch)

                # train_states = (actor_train_state, critic_train_state)
                train_states, loss_info = jax.lax.scan(
                    _update_minbatch, train_states, minibatches
                )
                update_state = (
                    train_states,
                    jax.tree.map(lambda x: x.squeeze(), init_hstates),
                    traj_batch,
                    advantages,
                    targets,
                    reference_initial_hstate.squeeze(axis=0),
                    reference_traj,
                    reference_returns,
                    reference_advantages,
                    reference_valid,
                    recovery_target,
                    rng,
                )
                return update_state, loss_info

            update_state = (
                train_states,
                initial_hstates,
                traj_batch,
                advantages,
                targets,
                reference_initial_hstate,
                reference_traj,
                reference_returns,
                reference_advantages,
                reference_valid,
                recovery_target,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            loss_info["ratio_0"] = loss_info["ratio"].at[0, 0].get()
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)

            train_states = update_state[0]
            metric = traj_batch.info
            metric = jax.tree.map(
                lambda x: x.reshape(
                    (config["NUM_STEPS"], config["NUM_ENVS"], env.num_agents)
                ),
                traj_batch.info,
            )
            metric["loss"] = loss_info
            metric["shuffle"] = shuffle_audit
            rng = update_state[-1]

            def callback(metric):
                # The metrics have an agent dimension, but they are identical
                # for all agents, so index into the 0th agent.
                log_data = {
                    "returns": metric["returned_episode_returns"][:, :, 0][
                        metric["returned_episode"][:, :, 0]
                    ].mean(),
                    "win_rate": metric["returned_won_episode"][:, :, 0][
                        metric["returned_episode"][:, :, 0]
                    ].mean(),
                    "env_step": (metric["update_steps"] + 1)
                    * config["NUM_ENVS"]
                    * config["NUM_STEPS"],
                    "ppo_env_step": (metric["update_steps"] + 1)
                    * config["NUM_ENVS"]
                    * config["NUM_STEPS"],
                    "oracle_env_step": (metric["update_steps"] + 1)
                    * config["NUM_ENVS"]
                    * config["ORACLE_REFERENCE_HORIZON"]
                    * int(config["ORACLE_LATENT_DISTORTION"]),
                    "total_env_step": (metric["update_steps"] + 1)
                    * config["NUM_ENVS"]
                    * (
                        config["NUM_STEPS"]
                        + config["ORACLE_REFERENCE_HORIZON"]
                        * int(config["ORACLE_LATENT_DISTORTION"])
                    ),
                    **metric["loss"],
                    **metric["shuffle"],
                }
                if progress_bar is not None:
                    completed = int(np.asarray(metric["update_steps"]).item()) + 1
                    progress_bar.update(max(0, completed - progress_bar.n))
                    postfix = {"steps": f"{int(log_data['env_step']):,}"}
                    win_rate = float(np.asarray(log_data["win_rate"]).item())
                    if np.isfinite(win_rate):
                        postfix["win"] = f"{win_rate:.3f}"
                    progress_bar.set_postfix(postfix)
                wandb.log(log_data)
                if metrics_jsonl_callback is not None:
                    metrics_output = log_data
                    if config["ALIGN_GRADIENT_CALIBRATION"]:
                        # The calibration artifact intentionally contains no
                        # performance signal; lambda is selected only from
                        # initial cross/RL gradient ratios.
                        audit_keys = {
                            "env_step",
                            "actor_rl_grad_norm_mean",
                            "critic_rl_grad_norm",
                            "alignment_c_to_a_distance",
                            "alignment_a_to_c_distance",
                        }
                        metrics_output = {
                            key: value
                            for key, value in log_data.items()
                            if key.startswith("calibration_") or key in audit_keys
                        }
                    metrics_jsonl_callback(metrics_output)

            metric["update_steps"] = update_steps
            jax.debug.callback(callback, metric, ordered=True)

            if checkpoint_callback is not None:
                rollout_env_steps = config["NUM_ENVS"] * config["NUM_STEPS"]
                completed_updates = update_steps + 1
                completed_env_steps = completed_updates * rollout_env_steps
                previous_env_steps = update_steps * rollout_env_steps
                checkpoint_interval = config["CHECKPOINT_INTERVAL_TIMESTEPS"]
                crossed_interval = (
                    completed_env_steps // checkpoint_interval
                    > previous_env_steps // checkpoint_interval
                )
                is_final = completed_updates == config["NUM_UPDATES"]
                should_save = jnp.logical_or(crossed_interval, is_final)
                nominal_env_steps = (
                    completed_env_steps // checkpoint_interval
                ) * checkpoint_interval

                def save_checkpoint(_):
                    return io_callback(
                        checkpoint_callback,
                        jax.ShapeDtypeStruct((), jnp.int32),
                        train_states[0].params,
                        train_states[1].params,
                        completed_env_steps,
                        nominal_env_steps,
                        is_final,
                        jnp.asarray(False),
                        ordered=True,
                    )

                jax.lax.cond(
                    should_save,
                    save_checkpoint,
                    lambda _: jnp.int32(0),
                    operand=None,
                )

            update_steps = update_steps + 1
            runner_state = (train_states, env_state, last_obs, last_done, hstates, rng)
            return (runner_state, update_steps), metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            (actor_train_state, critic_train_state),
            env_state,
            obsv,
            jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
            (ac_init_hstate, cr_init_hstate),
            _rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, (runner_state, 0), None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state}

    return train


@hydra.main(
    version_base=None, config_path="config", config_name="mappo_homogenous_rnn_smax"
)
def main(config):

    config = OmegaConf.to_container(config)
    # Defaults keep older external config dictionaries/checkpoints compatible.
    config.setdefault("ALIGN_DISTANCE", "ln_mse")
    config.setdefault("ALIGN_DISTANCE_EPS", 1e-8)
    config.setdefault("ALIGN_GRADIENT_CALIBRATION", False)
    config.setdefault("ORACLE_LATENT_DISTORTION", False)
    config.setdefault("ORACLE_DISTORTION_COEF", 0.0)
    config.setdefault("ORACLE_FISHER_RIDGE", 1e-3)
    config.setdefault("ORACLE_REFERENCE_MULTIPLIER", 4)
    config.setdefault("ORACLE_REFERENCE_BASELINE", "crossfit_linear_critic")
    config.setdefault("ORACLE_REFERENCE_SEED_OFFSET", 900_000)
    config.setdefault("SCORE_RECOVERY", False)
    config.setdefault("SCORE_RECOVERY_COEF", 0.0)
    config.setdefault("SCORE_RECOVERY_FISHER_RIDGE", 1e-3)
    config.setdefault("METRICS_JSONL", "")
    if not config.get("GIT_COMMIT"):
        try:
            config["GIT_COMMIT"] = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            config["GIT_COMMIT"] = "unknown"
    condition = config["ALIGN_MODE"]
    if config["ORACLE_LATENT_DISTORTION"]:
        condition = "oracle_latent_distortion"
    elif config["SCORE_RECOVERY"]:
        condition = "score_recovery"
    elif config["ALIGN_TARGET_SHUFFLE"]:
        condition = f"{condition}_shuffled"
    elif config["ALIGN_DISTANCE"] == "linear_cka" and condition != "none":
        condition = f"{condition}_cka"
    elif config["ALIGN_DISTANCE"] == "containment" and condition != "none":
        condition = f"{condition}_dsc"
    if config.get("EXPERIMENT_CONDITION"):
        if config["EXPERIMENT_CONDITION"] != condition:
            raise ValueError(
                "EXPERIMENT_CONDITION does not match ALIGN_MODE/shuffle: "
                f"{config['EXPERIMENT_CONDITION']!r} != {condition!r}"
            )
    else:
        config["EXPERIMENT_CONDITION"] = condition
    sharing_mode = (
        "shared-actor" if config["ACTOR_PARAMETER_SHARING"] else "independent-actors"
    )
    actor_mode = (
        f"matched-{sharing_mode}" if config["MATCHED_COMPARISON"] else sharing_mode
    )

    run = wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=[t for t in os.environ.get("WANDB_TAGS", "").split(",") if t],
        group=os.environ.get("WANDB_RUN_GROUP") or None,
        name=os.environ.get("WANDB_NAME")
        or (
            f"MAPPO-{actor_mode}-{condition}-"
            f"{config['MAP_NAME']}-seed{config['SEED']}"
        ),
        config=config,
        mode=config["WANDB_MODE"],
    )
    run.define_metric("env_step")
    run.define_metric("*", step_metric="env_step")
    checkpoint_callback = None
    checkpoint_run_dir = None
    if config["SAVE_CHECKPOINTS"]:
        checkpoint_interval = int(config["CHECKPOINT_INTERVAL_TIMESTEPS"])
        if checkpoint_interval <= 0:
            raise ValueError("CHECKPOINT_INTERVAL_TIMESTEPS must be positive")
        config["CHECKPOINT_INTERVAL_TIMESTEPS"] = checkpoint_interval
        checkpoint_callback, checkpoint_run_dir = make_checkpoint_callback(config, run)
        print(f"Checkpoints: {checkpoint_run_dir}", flush=True)
    metrics_jsonl_callback = make_metrics_jsonl_callback(config["METRICS_JSONL"])

    rng = jax.random.PRNGKey(config["SEED"])
    num_updates = int(
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    progress_bar = tqdm(
        total=num_updates,
        desc=(
            f"MAPPO {actor_mode} {condition} "
            f"{config['MAP_NAME']} seed={config['SEED']}"
        ),
        unit="update",
        dynamic_ncols=True,
    )
    try:
        with jax.disable_jit(False):
            train_jit = jax.jit(
                make_train(
                    config,
                    progress_bar,
                    checkpoint_callback,
                    metrics_jsonl_callback,
                )
            )
            result = train_jit(rng)
            jax.block_until_ready(result)
    finally:
        progress_bar.close()
        wandb.finish()


if __name__ == "__main__":
    main()
