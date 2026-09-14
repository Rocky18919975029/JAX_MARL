"""Evaluate a saved recurrent MAPPO policy on SMAX."""

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np
import wandb
from tqdm.auto import tqdm

from baselines.MAPPO.mappo_rnn_smax import (
    ActorRNN,
    SMAXWorldStateWrapper,
    ScannedRNN,
    batchify,
    unbatchify,
)
from baselines.MAPPO.smax_rollout import smax_rollout_horizon
from jaxmarl.environments.smax import HeuristicEnemySMAX, map_name_to_scenario
from jaxmarl.wrappers.baselines import load_params


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate an actor saved by mappo_rnn_smax.py"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint directory, run directory, or model.safetensors path",
    )
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument(
        "--policy",
        choices=("deterministic", "stochastic"),
        default="deterministic",
    )
    parser.add_argument(
        "--map-name",
        default=None,
        help="Optional map override; defaults to the checkpoint training map",
    )
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument(
        "--wandb-mode",
        choices=("disabled", "online", "offline"),
        default="disabled",
    )
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-name", default=None)
    return parser.parse_args()


def resolve_checkpoint(checkpoint_arg):
    path = Path(checkpoint_arg).expanduser().resolve()
    if path.is_file():
        model_path = path
        checkpoint_dir = path.parent
    elif (path / "model.safetensors").is_file():
        checkpoint_dir = path
        model_path = path / "model.safetensors"
    elif (path / "final" / "model.safetensors").is_file():
        checkpoint_dir = path / "final"
        model_path = checkpoint_dir / "model.safetensors"
    elif (path / "latest.json").is_file():
        with (path / "latest.json").open(encoding="utf-8") as file:
            latest = json.load(file)
        checkpoint_dir = path / latest["checkpoint"]
        model_path = checkpoint_dir / "model.safetensors"
    else:
        raise FileNotFoundError(f"No checkpoint found at {path}")

    config_path = checkpoint_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint config: {config_path}")
    return checkpoint_dir, model_path, config_path


def make_eval_batch(config, actor_params, num_envs, deterministic):
    scenario = map_name_to_scenario(config["MAP_NAME"])
    env = HeuristicEnemySMAX(scenario=scenario, **config["ENV_KWARGS"])
    env = SMAXWorldStateWrapper(env, config["OBS_WITH_AGENT_ID"])

    num_agents = env.num_agents
    num_actors = num_agents * num_envs
    hidden_dim = config["GRU_HIDDEN_DIM"]
    actor_network = ActorRNN(env.action_space(env.agents[0]).n, config=config)
    actor_parameter_sharing = config["ACTOR_PARAMETER_SHARING"]
    rollout_horizon = smax_rollout_horizon(env.max_steps)

    if not actor_parameter_sharing:
        first_leaf = jax.tree.leaves(actor_params)[0]
        if first_leaf.shape[0] != num_agents:
            raise ValueError(
                "Independent-actor checkpoint has a different agent count: "
                f"params={first_leaf.shape[0]}, eval environment={num_agents}"
            )

    def select_action(hidden, obs, done, env_state, key):
        avail_actions = jax.vmap(env.get_avail_actions)(env_state)
        avail_actions = batchify(avail_actions, env.agents, num_actors)
        obs_batch = batchify(obs, env.agents, num_actors)

        if actor_parameter_sharing:
            actor_input = (
                obs_batch[jnp.newaxis, :],
                done[jnp.newaxis, :],
                avail_actions,
            )
            hidden, pi, _ = actor_network.apply(actor_params, hidden, actor_input)
            action = pi.mode() if deterministic else pi.sample(seed=key)
        else:
            hidden_by_agent = hidden.reshape((num_agents, num_envs, hidden_dim))
            actor_input = (
                obs_batch.reshape((num_agents, num_envs, -1))[:, jnp.newaxis, ...],
                done.reshape((num_agents, num_envs))[:, jnp.newaxis, ...],
                avail_actions.reshape((num_agents, num_envs, -1)),
            )
            action_keys = jax.random.split(key, num_agents)

            def apply_policy(params, agent_hidden, agent_input, action_key):
                agent_hidden, pi, _ = actor_network.apply(
                    params, agent_hidden, agent_input
                )
                action = pi.mode() if deterministic else pi.sample(seed=action_key)
                return agent_hidden, action

            hidden, action = jax.vmap(
                apply_policy,
                in_axes=(0, 0, 0, 0),
            )(actor_params, hidden_by_agent, actor_input, action_keys)
            hidden = hidden.reshape((num_actors, hidden_dim))
            action = action.reshape((1, num_actors))

        env_action = unbatchify(action, env.agents, num_envs, num_agents)
        env_action = {agent: value.squeeze(-1) for agent, value in env_action.items()}
        return hidden, env_action

    def eval_batch(key):
        key, reset_key = jax.random.split(key)
        reset_keys = jax.random.split(reset_key, num_envs)
        obs, env_state = jax.vmap(env.reset)(reset_keys)
        hidden = ScannedRNN.initialize_carry(num_actors, hidden_dim)
        done = jnp.zeros((num_actors,), dtype=jnp.bool_)
        episode_returns = jnp.zeros((num_envs,), dtype=jnp.float32)
        episode_wins = jnp.zeros((num_envs,), dtype=jnp.float32)
        episode_lengths = jnp.zeros((num_envs,), dtype=jnp.int32)
        finished = jnp.zeros((num_envs,), dtype=jnp.bool_)

        carry = (
            obs,
            env_state,
            hidden,
            done,
            key,
            episode_returns,
            episode_wins,
            episode_lengths,
            finished,
        )

        def step_once(_, carry):
            (
                obs,
                env_state,
                hidden,
                done,
                key,
                episode_returns,
                episode_wins,
                episode_lengths,
                finished,
            ) = carry
            key, action_key, step_key = jax.random.split(key, 3)
            hidden, env_action = select_action(hidden, obs, done, env_state, action_key)
            step_keys = jax.random.split(step_key, num_envs)
            obs, env_state, reward, env_done, _ = jax.vmap(env.step)(
                step_keys, env_state, env_action
            )

            active = jnp.logical_not(finished)
            terminal = jnp.logical_and(active, env_done["__all__"])
            reference_reward = reward[env.agents[0]]
            episode_returns = episode_returns + active * reference_reward
            episode_lengths = episode_lengths + active.astype(jnp.int32)
            episode_wins = jnp.where(
                terminal,
                (reference_reward >= 1.0).astype(jnp.float32),
                episode_wins,
            )
            finished = jnp.logical_or(finished, env_done["__all__"])
            done = batchify(env_done, env.agents, num_actors).reshape((num_actors,))

            return (
                obs,
                env_state,
                hidden,
                done,
                key,
                episode_returns,
                episode_wins,
                episode_lengths,
                finished,
            )

        carry = jax.lax.fori_loop(0, rollout_horizon, step_once, carry)
        return carry[5], carry[6], carry[7], carry[8]

    return jax.jit(eval_batch), rollout_horizon


def mean_standard_error(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    if len(values) <= 1:
        return mean, 0.0, 0.0
    standard_deviation = float(values.std(ddof=1))
    standard_error = standard_deviation / math.sqrt(len(values))
    return mean, standard_deviation, standard_error


def main():
    args = parse_args()
    if args.episodes <= 0 or args.num_envs <= 0:
        raise ValueError("--episodes and --num-envs must be positive")

    checkpoint_dir, model_path, config_path = resolve_checkpoint(args.checkpoint)
    with config_path.open(encoding="utf-8") as file:
        config = json.load(file)
    metadata_path = checkpoint_dir / "metadata.json"
    checkpoint_metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.is_file()
        else {}
    )
    if args.map_name is not None:
        config["MAP_NAME"] = args.map_name

    checkpoint = load_params(model_path)
    if "actor" not in checkpoint or "critic" not in checkpoint:
        raise ValueError(
            "Checkpoint must contain both actor and critic parameter trees"
        )
    actor_params = checkpoint["actor"]

    deterministic = args.policy == "deterministic"
    eval_batch, rollout_horizon = make_eval_batch(
        config,
        actor_params,
        args.num_envs,
        deterministic,
    )

    returns = []
    wins = []
    lengths = []
    master_key = jax.random.PRNGKey(args.seed)
    num_batches = math.ceil(args.episodes / args.num_envs)
    for _ in tqdm(range(num_batches), desc="Evaluating", unit="batch"):
        master_key, batch_key = jax.random.split(master_key)
        batch_returns, batch_wins, batch_lengths, batch_finished = eval_batch(batch_key)
        batch_finished = np.asarray(batch_finished)
        if not batch_finished.all():
            unfinished = int((~batch_finished).sum())
            raise RuntimeError(
                f"{unfinished}/{len(batch_finished)} evaluation episodes did not "
                f"terminate after {rollout_horizon} transitions"
            )
        returns.append(np.asarray(batch_returns))
        wins.append(np.asarray(batch_wins))
        lengths.append(np.asarray(batch_lengths))

    returns = np.concatenate(returns)[: args.episodes]
    wins = np.concatenate(wins)[: args.episodes]
    lengths = np.concatenate(lengths)[: args.episodes]
    return_mean, return_std, return_stderr = mean_standard_error(returns)
    win_rate, win_std, win_stderr = mean_standard_error(wins)
    length_mean, length_std, length_stderr = mean_standard_error(lengths)

    result = {
        "checkpoint": str(checkpoint_dir),
        "run_id": checkpoint_metadata.get("wandb_run_id"),
        "run_name": checkpoint_metadata.get("wandb_run_name"),
        "checkpoint_env_step": config.get("CHECKPOINT_ENV_STEP"),
        "checkpoint_nominal_env_step": config.get("CHECKPOINT_NOMINAL_ENV_STEP"),
        "map_name": config["MAP_NAME"],
        "training_seed": config["SEED"],
        "eval_seed": args.seed,
        "policy": args.policy,
        "episodes": args.episodes,
        "num_envs": args.num_envs,
        "rollout_horizon": rollout_horizon,
        "actor_parameter_sharing": config["ACTOR_PARAMETER_SHARING"],
        "align_mode": config["ALIGN_MODE"],
        "align_target_shuffle": config.get("ALIGN_TARGET_SHUFFLE", False),
        "condition": config.get("EXPERIMENT_CONDITION", config["ALIGN_MODE"]),
        "matrix_profile": config.get("MATRIX_PROFILE", ""),
        "alignment_coef": config["ALIGNMENT_COEF"],
        "protocol_version": config.get("PROTOCOL_VERSION", ""),
        "git_commit": config.get("GIT_COMMIT", ""),
        "return_mean": return_mean,
        "return_std": return_std,
        "return_stderr": return_stderr,
        "return_ci95": [
            return_mean - 1.96 * return_stderr,
            return_mean + 1.96 * return_stderr,
        ],
        "win_rate": win_rate,
        "win_std": win_std,
        "win_stderr": win_stderr,
        "win_rate_ci95": [
            max(0.0, win_rate - 1.96 * win_stderr),
            min(1.0, win_rate + 1.96 * win_stderr),
        ],
        "episode_length_mean": length_mean,
        "episode_length_std": length_std,
        "episode_length_stderr": length_stderr,
        "episode_returns": returns.tolist(),
        "episode_wins": wins.astype(int).tolist(),
        "episode_lengths": lengths.astype(int).tolist(),
    }

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else checkpoint_dir
        / f"eval-{args.policy}-seed{args.seed}-n{args.episodes}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, sort_keys=True)
        file.write("\n")

    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Map: {config['MAP_NAME']}")
    print(f"Policy: {args.policy}")
    print(f"Episodes: {args.episodes}")
    print(f"Return: {return_mean:.6f} ± {return_stderr:.6f} SE")
    print(f"Win rate: {win_rate:.6f} ± {win_stderr:.6f} SE")
    print(f"Episode length: {length_mean:.3f} ± {length_stderr:.3f} SE")
    print(f"Saved evaluation: {output_path}")

    if args.wandb_mode != "disabled":
        project = args.wandb_project or f"{config.get('PROJECT') or 'smax'}-eval"
        run_name = args.wandb_name or (
            f"eval-{checkpoint_dir.parent.name}-{checkpoint_dir.name}-"
            f"{args.policy}-seed{args.seed}"
        )
        run = wandb.init(
            entity=config.get("ENTITY") or None,
            project=project,
            name=run_name,
            group=f"eval-{config['MAP_NAME']}-{args.policy}",
            mode=args.wandb_mode,
            config={
                "checkpoint": str(checkpoint_dir),
                "training_config": config,
                "eval_seed": args.seed,
                "episodes": args.episodes,
                "num_envs": args.num_envs,
                "policy": args.policy,
            },
        )
        run.log(
            {
                "eval/return_mean": return_mean,
                "eval/return_stderr": return_stderr,
                "eval/win_rate": win_rate,
                "eval/win_stderr": win_stderr,
                "eval/episode_length_mean": length_mean,
            }
        )
        run.finish()


if __name__ == "__main__":
    main()
