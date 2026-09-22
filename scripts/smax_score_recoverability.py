"""Core offline score-recoverability measurement for frozen SMAX policies.

The module never updates an RL network.  It consumes latents, sampled actions,
action masks, and exact representation-level policy scores produced by
``collect_mappo_smax_diagnostics.py``.  Fisher estimation and probe fitting use
only the episode-level fit split; the held-out split is touched only for the
reported measurement.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from scripts.h1_diagnostic_data import load_diagnostics


ARRAYS = (
    "active",
    "alive",
    "available_actions",
    "action",
    "actor_latent",
    "critic_latent",
    "actor_score",
    "diagnostic_episode_id",
)


def episode_fit_mask(episode_count: int, fit_fraction: float, seed: int) -> np.ndarray:
    """Return a deterministic episode-level fit/test assignment."""

    if episode_count < 2:
        raise ValueError("At least two episodes are required")
    if not 0.0 < fit_fraction < 1.0:
        raise ValueError("fit_fraction must lie strictly between zero and one")
    order = np.arange(episode_count, dtype=np.int32)
    np.random.default_rng(seed).shuffle(order)
    fit_count = min(episode_count - 1, max(1, int(episode_count * fit_fraction)))
    mask = np.zeros(episode_count, dtype=bool)
    mask[order[:fit_count]] = True
    return mask


def fisher_whiten(
    fit_scores: np.ndarray,
    test_scores: np.ndarray,
    ridge: float,
):
    """Whiten row-vector scores with a fit-only empirical Fisher matrix."""

    if fit_scores.ndim != 2 or test_scores.ndim != 2:
        raise ValueError("Scores must be rank-two matrices")
    if fit_scores.shape[1] != test_scores.shape[1]:
        raise ValueError("Fit/test score dimensions differ")
    if ridge <= 0 or not math.isfinite(ridge):
        raise ValueError("Fisher ridge must be finite and positive")
    score64 = np.asarray(fit_scores, dtype=np.float64)
    fisher = score64.T @ score64 / len(score64)
    fisher = 0.5 * (fisher + fisher.T)
    eigenvalues, eigenvectors = np.linalg.eigh(fisher)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    inverse_root = (eigenvectors * (eigenvalues + ridge) ** -0.5) @ eigenvectors.T
    fit_u = np.asarray(score64 @ inverse_root, dtype=np.float32)
    test_u = np.asarray(
        np.asarray(test_scores, dtype=np.float64) @ inverse_root,
        dtype=np.float32,
    )
    return fit_u, test_u, eigenvalues


def _sample_agent_split(
    arrays,
    agent: int,
    episode_mask: np.ndarray,
    count: int,
    seed: int,
):
    valid = (
        np.asarray(arrays["active"], dtype=bool)
        & np.asarray(arrays["alive"][:, :, agent], dtype=bool)
        & episode_mask[:, None]
    )
    candidates = np.argwhere(valid)
    if len(candidates) < count:
        raise RuntimeError(
            f"Agent {agent} has {len(candidates):,} valid transitions but "
            f"{count:,} are required"
        )
    rng = np.random.default_rng(seed + agent)
    chosen = candidates[rng.choice(len(candidates), size=count, replace=False)]
    episode = chosen[:, 0]
    timestep = chosen[:, 1]
    action = np.asarray(arrays["action"][episode, timestep, agent], dtype=np.int32)
    available = np.asarray(
        arrays["available_actions"][episode, timestep, agent], dtype=np.float32
    )
    if np.any(action < 0) or np.any(action >= available.shape[-1]):
        raise RuntimeError(f"Agent {agent} contains an out-of-range sampled action")
    selected_available = np.take_along_axis(available, action[:, None], axis=1)[:, 0]
    if not np.all(selected_available > 0.5):
        raise RuntimeError(
            f"Agent {agent} contains actions invalid under the stored SMAX mask"
        )
    return {
        "actor_latent": np.asarray(
            arrays["actor_latent"][episode, timestep, agent], dtype=np.float32
        ),
        "critic_latent": np.asarray(
            arrays["critic_latent"][episode, timestep, agent], dtype=np.float32
        ),
        "score": np.asarray(
            arrays["actor_score"][episode, timestep, agent], dtype=np.float32
        ),
        "action": action,
        "episode": episode,
        "timestep": timestep,
        "action_dim": available.shape[-1],
    }


def prepare_probe_data(
    arrays,
    *,
    fit_fraction: float,
    split_seed: int,
    sampling_seed: int,
    fit_samples_per_agent: int,
    test_samples_per_agent: int,
    fisher_ridge: float,
):
    """Create equal-size, per-agent fit/test probe tensors."""

    if fit_samples_per_agent <= 0 or test_samples_per_agent <= 0:
        raise ValueError("Probe sample counts must be positive")
    episodes = int(arrays["active"].shape[0])
    num_agents = int(arrays["alive"].shape[2])
    fit_episode = episode_fit_mask(episodes, fit_fraction, split_seed)
    fit_rows = []
    test_rows = []
    fisher_audit = []
    for agent in range(num_agents):
        fit = _sample_agent_split(
            arrays,
            agent,
            fit_episode,
            fit_samples_per_agent,
            sampling_seed,
        )
        test = _sample_agent_split(
            arrays,
            agent,
            ~fit_episode,
            test_samples_per_agent,
            sampling_seed + 1_000_000,
        )
        if fit["action_dim"] != test["action_dim"]:
            raise RuntimeError("Fit/test action dimensions differ")
        fit_u, test_u, eigenvalues = fisher_whiten(
            fit["score"], test["score"], fisher_ridge
        )
        mean = fit["critic_latent"].mean(axis=0)
        scale = fit["critic_latent"].std(axis=0)
        scale = np.where(scale > 1e-6, scale, 1.0).astype(np.float32)

        def probe_input(split):
            normalized = (split["critic_latent"] - mean) / scale
            action_one_hot = np.eye(split["action_dim"], dtype=np.float32)[
                split["action"]
            ]
            return np.concatenate((normalized, action_one_hot), axis=1).astype(
                np.float32
            )

        fit_rows.append((probe_input(fit), fit_u))
        test_rows.append((probe_input(test), test_u))
        positive = eigenvalues[eigenvalues > 1e-12]
        fisher_audit.append(
            {
                "agent_id": agent,
                "fisher_trace": float(eigenvalues.sum()),
                "fisher_min_eigenvalue": float(eigenvalues.min()),
                "fisher_max_eigenvalue": float(eigenvalues.max()),
                "fisher_positive_rank": int(len(positive)),
                "fisher_ridge": float(fisher_ridge),
            }
        )
    fit_x = np.stack([row[0] for row in fit_rows])
    fit_y = np.stack([row[1] for row in fit_rows])
    test_x = np.stack([row[0] for row in test_rows])
    test_y = np.stack([row[1] for row in test_rows])
    return {
        "fit_x": fit_x,
        "fit_y": fit_y,
        "test_x": test_x,
        "test_y": test_y,
        "fit_episode_mask": fit_episode,
        "fisher_audit": fisher_audit,
    }


def initialize_probe(key, input_dim: int, hidden_dim: int, output_dim: int):
    import jax
    import jax.numpy as jnp

    first, second = jax.random.split(key)
    return {
        "w1": jax.random.normal(first, (input_dim, hidden_dim))
        * math.sqrt(2.0 / input_dim),
        "b1": jnp.zeros((hidden_dim,)),
        "w2": jax.random.normal(second, (hidden_dim, output_dim))
        / math.sqrt(hidden_dim),
        "b2": jnp.zeros((output_dim,)),
    }


def probe_predict(params, inputs):
    import jax

    hidden = jax.nn.relu(inputs @ params["w1"] + params["b1"])
    return hidden @ params["w2"] + params["b2"]


def fit_independent_probes(
    fit_x: np.ndarray,
    fit_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    *,
    hidden_dim: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
):
    """Fit equal-capacity per-agent probes with one vmapped optimizer."""

    import jax
    import jax.numpy as jnp
    import optax

    if fit_x.ndim != 3 or fit_y.ndim != 3:
        raise ValueError("Probe fit arrays must have agent,sample,feature axes")
    if fit_x.shape[:2] != fit_y.shape[:2]:
        raise ValueError("Probe fit inputs and targets are misaligned")
    if test_x.shape[:2] != test_y.shape[:2]:
        raise ValueError("Probe test inputs and targets are misaligned")
    if fit_x.shape[0] != test_x.shape[0]:
        raise ValueError("Probe fit/test agent counts differ")
    if steps <= 0 or batch_size <= 0 or hidden_dim <= 0 or learning_rate <= 0:
        raise ValueError("Probe optimization hyperparameters must be positive")

    num_agents, fit_count, input_dim = fit_x.shape
    output_dim = fit_y.shape[-1]
    keys = jax.random.split(jax.random.PRNGKey(seed), num_agents)
    params = jax.vmap(
        lambda key: initialize_probe(key, input_dim, hidden_dim, output_dim)
    )(keys)
    optimizer = optax.adam(learning_rate)
    optimizer_state = optimizer.init(params)

    @jax.jit
    def update(current_params, current_optimizer_state, x, y):
        def loss_one(agent_params, agent_x, agent_y):
            prediction = probe_predict(agent_params, agent_x)
            return jnp.mean(jnp.square(prediction - agent_y))

        losses, gradients = jax.vmap(jax.value_and_grad(loss_one))(current_params, x, y)
        updates, current_optimizer_state = optimizer.update(
            gradients, current_optimizer_state, current_params
        )
        current_params = optax.apply_updates(current_params, updates)
        return current_params, current_optimizer_state, losses

    rng = np.random.default_rng(seed)
    effective_batch = min(batch_size, fit_count)
    history = []
    device_fit_x = jnp.asarray(fit_x)
    device_fit_y = jnp.asarray(fit_y)
    agent_axis = jnp.arange(num_agents)[:, None]
    for step in range(steps):
        indices = rng.choice(fit_count, size=effective_batch, replace=False)
        x = device_fit_x[agent_axis, jnp.asarray(indices)[None, :]]
        y = device_fit_y[agent_axis, jnp.asarray(indices)[None, :]]
        params, optimizer_state, losses = update(params, optimizer_state, x, y)
        if step == 0 or (step + 1) % 100 == 0 or step == steps - 1:
            history.append(
                {
                    "step": step + 1,
                    "mean_component_mse": float(np.asarray(losses).mean()),
                }
            )

    predict_all = jax.jit(jax.vmap(probe_predict))
    fit_prediction = np.asarray(predict_all(params, jnp.asarray(fit_x)))
    test_prediction = np.asarray(predict_all(params, jnp.asarray(test_x)))

    def errors(target, prediction):
        raw = np.mean(np.sum(np.square(target - prediction), axis=-1), axis=1)
        energy = np.mean(np.sum(np.square(target), axis=-1), axis=1)
        normalized = raw / np.maximum(energy, 1e-12)
        return raw, energy, normalized

    fit_raw, fit_energy, fit_normalized = errors(fit_y, fit_prediction)
    test_raw, test_energy, test_normalized = errors(test_y, test_prediction)
    return {
        "fit_raw": fit_raw,
        "fit_energy": fit_energy,
        "fit_normalized": fit_normalized,
        "test_raw": test_raw,
        "test_energy": test_energy,
        "test_normalized": test_normalized,
        "history": history,
    }


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def canonical_condition(metadata):
    condition = str(metadata["condition"])
    distance = str(metadata.get("align_distance", "ln_mse"))
    if condition == "none":
        return "none"
    if condition in {"c_to_a_mse", "c_to_a_cka"}:
        return condition
    if condition == "c_to_a" and distance == "ln_mse":
        return "c_to_a_mse"
    if condition in {"c_to_a", "c_to_a_cka"} and distance == "linear_cka":
        return "c_to_a_cka"
    raise RuntimeError(
        f"Unsupported recoverability condition: condition={condition}, "
        f"align_distance={distance}"
    )


def measure_score_recoverability(
    diagnostics_dir: Path,
    output_dir: Path,
    *,
    fit_fraction: float,
    split_seed: int,
    sampling_seed: int,
    fit_samples_per_agent: int,
    test_samples_per_agent: int,
    fisher_ridge: float,
    probe_hidden_dim: int,
    probe_steps: int,
    probe_batch_size: int,
    probe_learning_rate: float,
    probe_seed: int,
):
    metadata, arrays = load_diagnostics(diagnostics_dir, ARRAYS)
    if metadata.get("array_profile") not in (None, "score_recoverability", "full"):
        raise RuntimeError("Unsupported diagnostic array profile")
    if bool(metadata.get("actor_parameter_sharing")):
        raise RuntimeError("The preregistered score-recoverability scope is NPS only")
    data = prepare_probe_data(
        arrays,
        fit_fraction=fit_fraction,
        split_seed=split_seed,
        sampling_seed=sampling_seed,
        fit_samples_per_agent=fit_samples_per_agent,
        test_samples_per_agent=test_samples_per_agent,
        fisher_ridge=fisher_ridge,
    )
    result = fit_independent_probes(
        data["fit_x"],
        data["fit_y"],
        data["test_x"],
        data["test_y"],
        hidden_dim=probe_hidden_dim,
        steps=probe_steps,
        batch_size=probe_batch_size,
        learning_rate=probe_learning_rate,
        seed=probe_seed,
    )
    task = str(metadata["map_name"])
    condition = canonical_condition(metadata)
    seed = int(metadata["training_seed"])
    rows = []
    for agent, fisher in enumerate(data["fisher_audit"]):
        rows.append(
            {
                "task": task,
                "condition": condition,
                "seed": seed,
                "agent_id": agent,
                "epsilon_rec": float(result["test_raw"][agent]),
                "epsilon_rec_normalized": float(result["test_normalized"][agent]),
                "score_energy": float(result["test_energy"][agent]),
                "fit_epsilon_rec": float(result["fit_raw"][agent]),
                "fit_epsilon_rec_normalized": float(result["fit_normalized"][agent]),
                "fit_score_energy": float(result["fit_energy"][agent]),
                "fit_samples": fit_samples_per_agent,
                "test_samples": test_samples_per_agent,
                **fisher,
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "agent_metrics.csv", rows)
    protocol = {
        "schema_version": 1,
        "protocol": "smax-score-recoverability-v1.0",
        "task": task,
        "condition": condition,
        "training_seed": seed,
        "checkpoint": metadata["checkpoint"],
        "checkpoint_env_step": metadata.get("checkpoint_env_step"),
        "diagnostics_dir": str(Path(diagnostics_dir).resolve()),
        "diagnostic_seed": metadata["diagnostic_seed"],
        "episodes": int(metadata["episodes"]),
        "fit_fraction": fit_fraction,
        "fit_episodes": int(data["fit_episode_mask"].sum()),
        "test_episodes": int((~data["fit_episode_mask"]).sum()),
        "split_seed": split_seed,
        "sampling_seed": sampling_seed,
        "fit_samples_per_agent": fit_samples_per_agent,
        "test_samples_per_agent": test_samples_per_agent,
        "fisher_source": "fit_split_only",
        "fisher_ridge_absolute": fisher_ridge,
        "score_source": "exact_masked_categorical_d_log_prob_d_actor_latent",
        "probe_input": "standardized_critic_latent_plus_one_hot_action",
        "probe_hidden_dim": probe_hidden_dim,
        "probe_steps": probe_steps,
        "probe_batch_size": probe_batch_size,
        "probe_learning_rate": probe_learning_rate,
        "probe_seed": probe_seed,
        "probe_test_used_for_selection": False,
        "agent_aggregation": "unweighted_mean",
        "epsilon_rec": float(np.mean(result["test_raw"])),
        "epsilon_rec_normalized": float(np.mean(result["test_normalized"])),
        "num_agents": len(rows),
        "optimization_history": result["history"],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return rows, protocol
