"""
Based on PureJaxRL Implementation of IPPO, with changes to give a centralised critic.
"""

import functools
import json
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
    from baselines.MAPPO.score_recovery_target import whiten_rollout_scores
except ModuleNotFoundError:  # Direct execution from baselines/MAPPO.
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
    project_name = _safe_path_component(config.get("PROJECT") or run.project or "local")
    # W&B's disabled mode substitutes a random dummy ID/name even when id= is
    # passed to init(). Checkpoint identity must instead follow the frozen
    # launcher configuration in every logging mode.
    run_name = _safe_path_component(os.environ.get("WANDB_NAME") or run.name)
    run_id = _safe_path_component(config.get("WANDB_RUN_ID") or run.id)
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
        actor_recovery_params=None,
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
        model_params = {"actor": actor_params, "critic": critic_params}
        if actor_recovery_params is not None:
            model_params["actor_score_recovery_head"] = actor_recovery_params
        save_params(model_params, temporary_model_path)
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
                "alignment_coef": config["ALIGNMENT_COEF"],
                "actor_score_recovery": config["ACTOR_SCORE_RECOVERY"],
                "actor_score_recovery_coef": config["ACTOR_SCORE_RECOVERY_COEF"],
                "actor_score_recovery_fisher_ridge": config[
                    "ACTOR_SCORE_RECOVERY_FISHER_RIDGE"
                ],
                "actor_score_recovery_q_lr": config["ACTOR_SCORE_RECOVERY_Q_LR"],
                "actor_score_recovery_q_steps": config[
                    "ACTOR_SCORE_RECOVERY_Q_STEPS"
                ],
                "condition": config.get("EXPERIMENT_CONDITION", ""),
                "matrix_profile": config.get("MATRIX_PROFILE", ""),
                "protocol_version": config.get("PROTOCOL_VERSION", ""),
                "git_commit": config.get("GIT_COMMIT", ""),
                "wandb_project": project_name,
                "wandb_run_id": run_id,
                "wandb_run_name": run_name,
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
    """Separate q optimizer; no recovery gradient can update the critic."""

    action_dim: int
    latent_dim: int
    zero_output: bool = False

    @nn.compact
    def __call__(self, critic_latent, action):
        one_hot = jax.nn.one_hot(action, self.action_dim)
        features = jnp.concatenate((critic_latent, one_hot), axis=-1)
        features = nn.relu(nn.Dense(self.latent_dim, name="Dense_0")(features))
        output_init = (
            {"kernel_init": nn.initializers.zeros, "bias_init": nn.initializers.zeros}
            if self.zero_output
            else {}
        )
        return nn.Dense(self.latent_dim, name="Dense_1", **output_init)(features)


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
    info: jnp.ndarray
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


def categorical_latent_score(
    actor_latent, action, avail_actions, policy_hidden_params, policy_logits_params
):
    """Exact masked categorical score d log pi(a|z) / dz; differentiable for ARec."""
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
):
    """Dispatch to the configured alignment distance.

    Linear CKA is evaluated independently within each agent's sample pool.
    This avoids comparing coordinates emitted by different independent actor
    encoders and gives PS and NPS runs the same loss definition.
    """

    if distance_name == "ln_mse":
        return latent_distance(source, target, mask)
    if distance_name != "linear_cka":
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
    distances = jax.vmap(linear_cka_distance, in_axes=(0, 0, 0, None))(
        grouped_source,
        grouped_target,
        grouped_mask,
        epsilon,
    )
    valid_groups = grouped_mask.reshape((num_agent_groups, -1)).sum(axis=1) > 1
    valid_groups = valid_groups.astype(distances.dtype)
    return (distances * valid_groups).sum() / jnp.maximum(valid_groups.sum(), 1.0)


def make_train(
    config,
    progress_bar=None,
    checkpoint_callback=None,
    metrics_jsonl_callback=None,
):
    config.setdefault("ALIGN_DISTANCE", "ln_mse")
    config.setdefault("ALIGN_DISTANCE_EPS", 1e-8)
    config.setdefault("ACTOR_SCORE_RECOVERY", False)
    config.setdefault("ACTOR_SCORE_RECOVERY_COEF", 0.0)
    config.setdefault("ACTOR_SCORE_RECOVERY_FISHER_RIDGE", 1e-3)
    config.setdefault("ACTOR_SCORE_RECOVERY_Q_LR", 1e-3)
    config.setdefault("ACTOR_SCORE_RECOVERY_Q_STEPS", 8)
    scenario = map_name_to_scenario(config["MAP_NAME"])
    env = HeuristicEnemySMAX(scenario=scenario, **config["ENV_KWARGS"])
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    valid_align_modes = {"none", "c_to_a"}
    if config["ALIGN_MODE"] not in valid_align_modes:
        raise ValueError(
            f"ALIGN_MODE must be one of {sorted(valid_align_modes)}, got "
            f"{config['ALIGN_MODE']!r}."
        )
    valid_align_distances = {"ln_mse", "linear_cka"}
    if config["ALIGN_DISTANCE"] not in valid_align_distances:
        raise ValueError(
            f"ALIGN_DISTANCE must be one of {sorted(valid_align_distances)}, got "
            f"{config['ALIGN_DISTANCE']!r}."
        )
    if float(config["ALIGN_DISTANCE_EPS"]) <= 0:
        raise ValueError("ALIGN_DISTANCE_EPS must be positive.")
    if config["ALIGN_MODE"] != "none" and not config["MATCHED_COMPARISON"]:
        raise ValueError("Representation alignment requires MATCHED_COMPARISON=true.")
    if config["ALIGN_MODE"] == "none" and config["ALIGN_DISTANCE"] != "ln_mse":
        raise ValueError(
            "ALIGN_MODE=none must use ALIGN_DISTANCE=ln_mse so the distance-free "
            "baseline is not duplicated."
        )
    if config["ACTOR_PARAMETER_SHARING"] or not config["MATCHED_COMPARISON"]:
        raise ValueError("The four-method SMAX protocol requires matched NPS actors.")
    if config["ALIGN_MODE"] == "none" and config["ALIGNMENT_COEF"] != 0:
        raise ValueError("ALIGN_MODE=none requires ALIGNMENT_COEF=0.")
    if config["ALIGN_MODE"] == "c_to_a" and config["ALIGNMENT_COEF"] <= 0:
        raise ValueError("C→A alignment requires ALIGNMENT_COEF>0.")
    if config["ACTOR_SCORE_RECOVERY"]:
        if config["ALIGN_MODE"] != "none" or config["ALIGNMENT_COEF"] != 0:
            raise ValueError("ARec and latent alignment are mutually exclusive.")
        if config["ACTOR_SCORE_RECOVERY_COEF"] <= 0:
            raise ValueError("ACTOR_SCORE_RECOVERY_COEF must be positive.")
        if config["ACTOR_SCORE_RECOVERY_FISHER_RIDGE"] <= 0:
            raise ValueError("ACTOR_SCORE_RECOVERY_FISHER_RIDGE must be positive.")
        if config["ACTOR_SCORE_RECOVERY_Q_LR"] <= 0:
            raise ValueError("ACTOR_SCORE_RECOVERY_Q_LR must be positive.")
        if config["ACTOR_SCORE_RECOVERY_Q_STEPS"] <= 0:
            raise ValueError("ACTOR_SCORE_RECOVERY_Q_STEPS must be positive.")
    elif config["ACTOR_SCORE_RECOVERY_COEF"] != 0:
        raise ValueError("ARec coefficient must be zero when ARec is disabled.")
    if config["NUM_UPDATES"] < 1:
        raise ValueError("TOTAL_TIMESTEPS must include at least one full PPO rollout.")
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
        actor_recovery_head = ScoreRecoveryHead(
            action_dim=env.action_space(env.agents[0]).n,
            latent_dim=config["GRU_HIDDEN_DIM"],
            zero_output=True,
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
        actor_network_params = actor_network.init(
            _rng_actor, ac_init_hstate, ac_init_x
        )
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
        actor_tx = vmapped_optimizer(base_actor_tx)
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
        actor_recovery_train_state = None
        if config["ACTOR_SCORE_RECOVERY"]:
            # Fold-in preserves matched actor/critic initialization and PPO RNG.
            q_rngs = jax.random.split(
                jax.random.fold_in(_rng_critic, 71227), env.num_agents
            )
            q_params = jax.vmap(actor_recovery_head.init, in_axes=(0, None, None))(
                q_rngs,
                jnp.zeros((1, config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])),
                jnp.zeros((1, config["NUM_ENVS"]), dtype=jnp.int32),
            )
            q_tx = vmapped_optimizer(
                optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.adam(config["ACTOR_SCORE_RECOVERY_Q_LR"], eps=1e-5),
                )
            )
            actor_recovery_train_state = TrainState.create(
                apply_fn=actor_recovery_head.apply,
                params=q_params,
                tx=q_tx,
            )

        if checkpoint_callback is not None:
            checkpoint_args = (
                actor_train_state.params,
                critic_train_state.params,
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(False),
                jnp.asarray(True),
            )
            if config["ACTOR_SCORE_RECOVERY"]:
                checkpoint_args += (actor_recovery_train_state.params,)
            io_callback(
                checkpoint_callback,
                jax.ShapeDtypeStruct((), jnp.int32),
                *checkpoint_args,
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

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state

            def _env_step(runner_state, unused):
                train_states, q_state, env_state, last_obs, last_done, hstates, rng = (
                    runner_state
                )

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(
                    batchify(avail_actions, env.agents, config["NUM_ACTORS"])
                )
                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
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

                ac_hstate, action, log_prob, actor_latent = jax.vmap(
                    apply_and_sample,
                    in_axes=(0, 0, 0, 0),
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
                    info,
                    avail_actions,
                )
                runner_state = (
                    train_states,
                    q_state,
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

            # CALCULATE ADVANTAGE
            train_states, actor_recovery_train_state, env_state, last_obs, last_done, hstates, rng = runner_state

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

            if config["ACTOR_SCORE_RECOVERY"]:
                # One frozen score/Fisher target per full rollout, never per minibatch.
                def agent_first(x):
                    return jnp.swapaxes(
                        x.reshape(
                            (config["NUM_STEPS"], env.num_agents, config["NUM_ENVS"],
                             *x.shape[2:])
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
                target_by_agent, recovery_target_audit, recovery_inverse_root = (
                    whiten_rollout_scores(
                        scores_by_agent,
                        agent_first(traj_batch.alive_mask),
                        config["ACTOR_SCORE_RECOVERY_FISHER_RIDGE"],
                        return_matrix=True,
                    )
                )
                recovery_inverse_root = jax.lax.stop_gradient(recovery_inverse_root)
                q_latent = jax.lax.stop_gradient(
                    agent_first(traj_batch.critic_latent_old)
                )
                q_action = agent_first(traj_batch.action)
                q_valid = agent_first(traj_batch.alive_mask)

                def q_agent_loss(params, latent, action, target, valid):
                    prediction = actor_recovery_head.apply(params, latent, action)
                    squared = jnp.sum(jnp.square(prediction - target), axis=-1)
                    weight = valid.astype(squared.dtype)
                    return (squared * weight).sum() / jnp.maximum(weight.sum(), 1.0)

                q_grad_fn = jax.vmap(jax.value_and_grad(q_agent_loss))
                q_fit_loss_pre = jax.vmap(q_agent_loss)(
                    actor_recovery_train_state.params,
                    q_latent, q_action, target_by_agent, q_valid,
                ).mean()

                def q_update_step(q_state, unused):
                    losses, grads = q_grad_fn(
                        q_state.params, q_latent, q_action, target_by_agent, q_valid
                    )
                    return q_state.apply_gradients(grads=grads), losses.mean()

                actor_recovery_train_state, _ = jax.lax.scan(
                    q_update_step,
                    actor_recovery_train_state,
                    None,
                    config["ACTOR_SCORE_RECOVERY_Q_STEPS"],
                )
                q_prediction = jax.vmap(actor_recovery_head.apply)(
                    actor_recovery_train_state.params, q_latent, q_action
                )
                q_post_squared = jnp.sum(
                    jnp.square(q_prediction - target_by_agent), axis=-1
                )
                q_weight = q_valid.astype(q_post_squared.dtype)
                q_fit_loss_post = (
                    (q_post_squared * q_weight).sum(axis=(1, 2))
                    / jnp.maximum(q_weight.sum(axis=(1, 2)), 1.0)
                ).mean()
                actor_recovery_teacher = jax.lax.stop_gradient(
                    q_prediction.swapaxes(0, 1).reshape(traj_batch.actor_latent_old.shape)
                )
            else:
                q_fit_loss_pre = jnp.zeros((), dtype=advantages.dtype)
                q_fit_loss_post = q_fit_loss_pre
                recovery_target_audit = {
                    "target_energy_per_agent": jnp.zeros((env.num_agents,)),
                }
                recovery_inverse_root = jnp.broadcast_to(
                    jnp.eye(1, dtype=advantages.dtype), (env.num_agents, 1, 1)
                )
                actor_recovery_teacher = jnp.zeros_like(
                    traj_batch.actor_latent_old[..., :1]
                )

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_states, batch_info):
                    actor_train_state, critic_train_state = train_states
                    ac_init_hstate, cr_init_hstate, traj_batch, advantages, targets, recovery_teacher = (
                        batch_info
                    )

                    def _actor_loss_fn(
                        actor_params, init_hstate, traj_batch, gae,
                        recovery_teacher, inverse_root,
                    ):
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
                        recovery_loss = jnp.zeros((), dtype=actor_loss.dtype)
                        if config["ACTOR_SCORE_RECOVERY"]:
                            policy_params = actor_params["params"]
                            score = categorical_latent_score(
                                actor_latent,
                                traj_batch.action,
                                traj_batch.avail_actions,
                                policy_params["Dense_1"],
                                policy_params["Dense_2"],
                            )
                            normalized_score = jnp.einsum(
                                "ted,df->tef", score, inverse_root
                            )
                            squared = jnp.sum(
                                jnp.square(normalized_score - recovery_teacher),
                                axis=-1,
                            )
                            valid = traj_batch.alive_mask.astype(squared.dtype)
                            recovery_loss = (squared * valid).sum() / jnp.maximum(
                                valid.sum(), 1.0
                            )
                            actor_loss += config["ACTOR_SCORE_RECOVERY_COEF"] * recovery_loss

                        return actor_loss, (
                            loss_actor,
                            entropy,
                            ratio,
                            approx_kl,
                            clip_frac,
                            actor_latent,
                            recovery_loss,
                        )

                    def _critic_loss_fn(
                        critic_params, init_hstate, traj_batch, targets
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
                        critic_loss = config["VF_COEF"] * value_loss
                        return critic_loss, (value_loss, critic_latent)

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

                    actor_advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                    actor_batch = jax.tree.map(split_agents, traj_batch)
                    actor_hstates = split_agents(ac_init_hstate)
                    actor_advantages = split_agents(actor_advantages)
                    actor_recovery_teachers = split_agents(recovery_teacher)
                    def matched_total_loss(
                        actor_params,
                        critic_params,
                    ):
                        actor_losses, actor_aux = jax.vmap(
                            _actor_loss_fn,
                            in_axes=(0, 0, 0, 0, 0, 0),
                        )(
                            actor_params,
                            actor_hstates,
                            actor_batch,
                            actor_advantages,
                            actor_recovery_teachers,
                            recovery_inverse_root,
                        )
                        critic_rl_loss, critic_aux = _critic_loss_fn(
                            critic_params,
                            cr_init_hstate,
                            traj_batch,
                            targets,
                        )

                        actor_latent = (
                            actor_aux[5]
                            .swapaxes(0, 1)
                            .reshape(traj_batch.actor_latent_old.shape)
                        )
                        mask = traj_batch.alignment_mask
                        zero = jnp.zeros((), dtype=actor_latent.dtype)
                        c_to_a_loss = (
                            representation_distance(
                                actor_latent,
                                jax.lax.stop_gradient(traj_batch.critic_latent_old),
                                mask,
                                config["ALIGN_DISTANCE"],
                                env.num_agents,
                                config["ALIGN_DISTANCE_EPS"],
                            )
                            if config["ALIGN_MODE"] == "c_to_a"
                            else zero
                        )
                        total_loss = (
                            actor_losses.mean()
                            + critic_rl_loss
                            + config["ALIGNMENT_COEF"] * c_to_a_loss
                        )
                        return total_loss, (
                            actor_losses,
                            actor_aux,
                            critic_rl_loss,
                            critic_aux,
                            c_to_a_loss,
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

                    if config["ALIGN_MODE"] == "none" and not config["ACTOR_SCORE_RECOVERY"]:
                        actor_cross_grads = jax.tree.map(
                            jnp.zeros_like, actor_grads
                        )
                        critic_cross_grads = jax.tree.map(
                            jnp.zeros_like, critic_grads
                        )
                    elif config["ACTOR_SCORE_RECOVERY"]:
                        def per_agent_recovery_objective(
                            params, hidden, batch, gae, teacher, matrix
                        ):
                            _, auxiliary = _actor_loss_fn(
                                params, hidden, batch, gae, teacher, matrix
                            )
                            return (
                                config["ACTOR_SCORE_RECOVERY_COEF"]
                                * auxiliary[6] / env.num_agents
                            )

                        actor_cross_grads = jax.vmap(
                            jax.grad(per_agent_recovery_objective)
                        )(
                            actor_train_state.params,
                            actor_hstates,
                            actor_batch,
                            actor_advantages,
                            actor_recovery_teachers,
                            recovery_inverse_root,
                        )
                        critic_cross_grads = jax.tree.map(jnp.zeros_like, critic_grads)
                    else:

                        def matched_cross_objective(actor_params, critic_params):
                            _, auxiliary = matched_total_loss(
                                actor_params, critic_params
                            )
                            return config["ALIGNMENT_COEF"] * auxiliary[4]

                        actor_cross_grads, critic_cross_grads = jax.grad(
                            matched_cross_objective,
                            argnums=(0, 1),
                        )(
                            actor_train_state.params,
                            critic_train_state.params,
                        )
                    actor_grads = jax.tree.map(
                        lambda x: x * env.num_agents, actor_grads
                    )
                    actor_cross_grads = jax.tree.map(
                        lambda x: x * env.num_agents, actor_cross_grads
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
                    def actor_tree_norms(tree):
                        return jax.vmap(tree_l2_norm)(tree)

                    actor_grad_norms = actor_tree_norms(actor_grads)
                    actor_rl_grad_norms = actor_tree_norms(actor_rl_grads)
                    actor_cross_grad_norms = actor_tree_norms(actor_cross_grads)
                    critic_grad_norm = tree_l2_norm(critic_grads)
                    critic_rl_grad_norm = tree_l2_norm(critic_rl_grads)
                    critic_cross_grad_norm = tree_l2_norm(critic_cross_grads)
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
                    actor_update_norms = jax.vmap(tree_l2_norm)(actor_param_updates)
                    actor_grad_norms_after_clip = jnp.minimum(
                        actor_grad_norms, config["MAX_GRAD_NORM"]
                    )

                    mean_actor_loss = actor_loss[0].mean()
                    loss_info = {
                        "total_loss": combined_total_loss,
                        "actor_loss": mean_actor_loss,
                        "value_loss": critic_loss[1][0],
                        "entropy": actor_loss[1][1].mean(),
                        "ratio": actor_loss[1][2],
                        "approx_kl": actor_loss[1][3].mean(),
                        "clip_frac": actor_loss[1][4].mean(),
                        "alignment_c_to_a_distance": c_to_a_loss,
                        "alignment_objective_weighted": config["ALIGNMENT_COEF"] * c_to_a_loss,
                        "actor_score_recovery_loss": actor_loss[1][6].mean(),
                        "actor_score_recovery_objective_weighted": (
                            config["ACTOR_SCORE_RECOVERY_COEF"] * actor_loss[1][6].mean()
                        ),
                        "actor_score_recovery_q_fit_loss_pre": q_fit_loss_pre,
                        "actor_score_recovery_q_fit_loss_post": q_fit_loss_post,
                        "actor_score_recovery_q_to_zero_baseline_ratio": (
                            q_fit_loss_post
                            / jnp.maximum(
                                recovery_target_audit["target_energy_per_agent"].mean(),
                                1e-12,
                            )
                        ),
                        "actor_score_recovery_target_energy_mean": (
                            recovery_target_audit["target_energy_per_agent"].mean()
                        ),
                        "actor_score_recovery_actor_to_rl_grad_ratio_mean": (
                            (actor_cross_grad_norms
                             / jnp.maximum(actor_rl_grad_norms, 1e-12)).mean()
                            if config["ACTOR_SCORE_RECOVERY"] else 0.0
                        ),
                        "actor_score_recovery_critic_grad_norm": (
                            critic_cross_grad_norm if config["ACTOR_SCORE_RECOVERY"] else 0.0
                        ),
                        "actor_rl_grad_norm_mean": actor_rl_grad_norms.mean(),
                        "actor_cross_grad_norm_mean": actor_cross_grad_norms.mean(),
                        "actor_grad_norm_mean": actor_grad_norms.mean(),
                        "actor_grad_norm_max": actor_grad_norms.max(),
                        "actor_grad_norm_after_clip_mean": actor_grad_norms_after_clip.mean(),
                        "actor_grad_clipped_fraction": jnp.mean(
                            actor_grad_norms > config["MAX_GRAD_NORM"]
                        ),
                        "actor_update_norm_mean": actor_update_norms.mean(),
                        "critic_grad_norm": critic_grad_norm,
                        "critic_rl_grad_norm": critic_rl_grad_norm,
                        "critic_cross_grad_norm": critic_cross_grad_norm,
                        "critic_update_norm": tree_l2_norm(critic_param_updates),
                        "alive_agent_fraction": traj_batch.alive_mask.mean(),
                    }

                    return (actor_train_state, critic_train_state), loss_info

                (
                    train_states,
                    init_hstates,
                    traj_batch,
                    advantages,
                    targets,
                    recovery_teacher,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)

                init_hstates = jax.tree.map(
                    lambda x: jnp.reshape(x, (1, config["NUM_ACTORS"], -1)),
                    init_hstates,
                )

                batch = (
                    init_hstates[0],
                    init_hstates[1],
                    traj_batch,
                    advantages.squeeze(),
                    targets.squeeze(),
                    recovery_teacher,
                )
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
                    recovery_teacher,
                    rng,
                )
                return update_state, loss_info

            update_state = (
                train_states,
                initial_hstates,
                traj_batch,
                advantages,
                targets,
                actor_recovery_teacher,
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
                    **metric["loss"],
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
                    metrics_jsonl_callback(log_data)

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
                    checkpoint_args = (
                        train_states[0].params,
                        train_states[1].params,
                        completed_env_steps,
                        nominal_env_steps,
                        is_final,
                        jnp.asarray(False),
                    )
                    if config["ACTOR_SCORE_RECOVERY"]:
                        checkpoint_args += (actor_recovery_train_state.params,)
                    return io_callback(
                        checkpoint_callback,
                        jax.ShapeDtypeStruct((), jnp.int32),
                        *checkpoint_args,
                        ordered=True,
                    )

                jax.lax.cond(
                    should_save,
                    save_checkpoint,
                    lambda _: jnp.int32(0),
                    operand=None,
                )

            update_steps = update_steps + 1
            runner_state = (
                train_states, actor_recovery_train_state, env_state,
                last_obs, last_done, hstates, rng,
            )
            return (runner_state, update_steps), metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            (actor_train_state, critic_train_state),
            actor_recovery_train_state,
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
    config.setdefault("ACTOR_SCORE_RECOVERY", False)
    config.setdefault("ACTOR_SCORE_RECOVERY_COEF", 0.0)
    config.setdefault("ACTOR_SCORE_RECOVERY_FISHER_RIDGE", 1e-3)
    config.setdefault("ACTOR_SCORE_RECOVERY_Q_LR", 1e-3)
    config.setdefault("ACTOR_SCORE_RECOVERY_Q_STEPS", 8)
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
    if config["ACTOR_SCORE_RECOVERY"]:
        condition = "arec"
    elif config["ALIGN_DISTANCE"] == "linear_cka" and condition != "none":
        condition = f"{condition}_cka"
    if config.get("EXPERIMENT_CONDITION"):
        if config["EXPERIMENT_CONDITION"] != condition:
            raise ValueError(
                "EXPERIMENT_CONDITION does not match the configured method: "
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
        id=config.get("WANDB_RUN_ID") or None,
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
