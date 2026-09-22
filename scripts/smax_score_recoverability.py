"""Frozen-checkpoint, episode-held-out SMAX policy-score recoverability.

Each checkpoint supplies its own stochastic on-policy distribution. The RL
networks are never updated. Fisher and critic-latent normalization use fit
episodes only; validation selects probe weights and test is evaluated once.
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
    "reward",
    "diagnostic_episode_id",
)


def episode_split_masks(
    episode_count: int,
    fit_fraction: float,
    validation_fraction: float,
    seed: int,
) -> dict[str, np.ndarray]:
    """Deterministically assign complete episodes to three disjoint splits."""

    if episode_count < 7:
        raise ValueError("At least seven complete episodes are required")
    if (
        fit_fraction <= 0
        or validation_fraction <= 0
        or fit_fraction + validation_fraction >= 1
    ):
        raise ValueError("Fit, validation and test fractions must all be positive")
    order = np.random.default_rng(seed).permutation(episode_count)
    fit_count = int(episode_count * fit_fraction)
    validation_count = int(episode_count * validation_fraction)
    if (
        min(fit_count, validation_count, episode_count - fit_count - validation_count)
        < 1
    ):
        raise ValueError("Every episode split must be nonempty")
    masks = {}
    for name, indices in (
        ("fit", order[:fit_count]),
        ("validation", order[fit_count : fit_count + validation_count]),
        ("test", order[fit_count + validation_count :]),
    ):
        mask = np.zeros(episode_count, dtype=bool)
        mask[indices] = True
        masks[name] = mask
    return masks


def available_samples_by_agent(
    arrays, fit_fraction: float, validation_fraction: float, split_seed: int
) -> dict[str, np.ndarray]:
    """Count eligible transitions in each complete-episode split."""

    active = np.asarray(arrays["active"], dtype=bool)
    alive = np.asarray(arrays["alive"], dtype=bool)
    if active.ndim != 2 or alive.ndim != 3 or alive.shape[:2] != active.shape:
        raise ValueError("Expected active[episode,time] and alive[episode,time,agent]")
    masks = episode_split_masks(
        len(active), fit_fraction, validation_fraction, split_seed
    )
    valid = active[:, :, None] & alive
    return {
        name: valid[mask].sum(axis=(0, 1)).astype(np.int64)
        for name, mask in masks.items()
    }


def fisher_whiten(
    fit_scores: np.ndarray,
    validation_scores: np.ndarray,
    test_scores: np.ndarray,
    ridge: float,
):
    """Apply one fit-only inverse-Fisher square root to all three splits."""

    scores = (fit_scores, validation_scores, test_scores)
    if any(score.ndim != 2 for score in scores):
        raise ValueError("Scores must be rank-two matrices")
    if len({score.shape[1] for score in scores}) != 1 or not len(fit_scores):
        raise ValueError("Score dimensions differ or the fit split is empty")
    if ridge <= 0 or not math.isfinite(ridge):
        raise ValueError("Fisher ridge must be finite and positive")
    fit64 = np.asarray(fit_scores, dtype=np.float64)
    fisher = fit64.T @ fit64 / len(fit64)
    fisher = 0.5 * (fisher + fisher.T)
    eigenvalues, eigenvectors = np.linalg.eigh(fisher)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    inverse_root = (eigenvectors * (eigenvalues + ridge) ** -0.5) @ eigenvectors.T
    whitened = tuple(
        np.asarray(np.asarray(score, dtype=np.float64) @ inverse_root, dtype=np.float32)
        for score in scores
    )
    return (*whitened, eigenvalues)


def _sample_agent_split(
    arrays, agent: int, episode_mask: np.ndarray, count: int, seed: int
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
    episode, timestep = chosen[:, 0], chosen[:, 1]
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
        "critic_latent": np.asarray(
            arrays["critic_latent"][episode, timestep, agent], dtype=np.float32
        ),
        "score": np.asarray(
            arrays["actor_score"][episode, timestep, agent], dtype=np.float32
        ),
        "action": action,
        "episode": episode,
        "action_dim": available.shape[-1],
    }


def prepare_probe_data(
    arrays,
    *,
    fit_fraction: float,
    validation_fraction: float,
    split_seed: int,
    sampling_seed: int,
    fit_samples_per_agent: int,
    validation_samples_per_agent: int,
    test_samples_per_agent: int,
    fisher_ridge: float,
):
    """Make fixed-count per-agent arrays without episode or statistic leakage."""

    counts = {
        "fit": fit_samples_per_agent,
        "validation": validation_samples_per_agent,
        "test": test_samples_per_agent,
    }
    if any(count <= 0 for count in counts.values()):
        raise ValueError("Probe sample counts must be positive")
    episodes = int(arrays["active"].shape[0])
    agents = int(arrays["alive"].shape[2])
    masks = episode_split_masks(episodes, fit_fraction, validation_fraction, split_seed)
    rows = {name: [] for name in counts}
    fisher_audit = []
    selected_episodes = {name: [] for name in counts}
    for agent in range(agents):
        split = {
            name: _sample_agent_split(
                arrays,
                agent,
                masks[name],
                count,
                sampling_seed
                + {"fit": 0, "validation": 1_000_000, "test": 2_000_000}[name],
            )
            for name, count in counts.items()
        }
        if len({item["action_dim"] for item in split.values()}) != 1:
            raise RuntimeError("Action dimensions differ between episode splits")
        fit_u, validation_u, test_u, eigenvalues = fisher_whiten(
            split["fit"]["score"],
            split["validation"]["score"],
            split["test"]["score"],
            fisher_ridge,
        )
        targets = {"fit": fit_u, "validation": validation_u, "test": test_u}
        mean = split["fit"]["critic_latent"].mean(axis=0)
        scale = split["fit"]["critic_latent"].std(axis=0) + 1e-6
        for name in counts:
            item = split[name]
            normalized = (item["critic_latent"] - mean) / scale
            one_hot = np.eye(item["action_dim"], dtype=np.float32)[item["action"]]
            inputs = np.concatenate((normalized, one_hot), axis=1).astype(np.float32)
            rows[name].append((inputs, targets[name]))
            selected_episodes[name].append(set(item["episode"].tolist()))
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
    result = {
        f"{name}_{kind}": np.stack([item[index] for item in rows[name]])
        for name in counts
        for kind, index in (("x", 0), ("y", 1))
    }
    result.update(
        {
            "episode_masks": masks,
            "selected_episodes": selected_episodes,
            "fisher_audit": fisher_audit,
        }
    )
    return result


def initialize_probe(
    key, input_dim: int, hidden_dim: int, output_dim: int, blocks: int
):
    import jax
    import jax.numpy as jnp

    keys = iter(jax.random.split(key, 2 + 2 * blocks))
    first = {
        "w": jax.random.normal(next(keys), (input_dim, hidden_dim))
        * math.sqrt(2.0 / input_dim),
        "b": jnp.zeros((hidden_dim,)),
    }
    residual = []
    for _ in range(blocks):
        residual.append(
            {
                "w1": jax.random.normal(next(keys), (hidden_dim, hidden_dim))
                * math.sqrt(2.0 / hidden_dim),
                "b1": jnp.zeros((hidden_dim,)),
                "w2": jax.random.normal(next(keys), (hidden_dim, hidden_dim))
                / math.sqrt(hidden_dim),
                "b2": jnp.zeros((hidden_dim,)),
            }
        )
    final = {
        "w": jax.random.normal(next(keys), (hidden_dim, output_dim))
        / math.sqrt(hidden_dim),
        "b": jnp.zeros((output_dim,)),
    }
    return {"input": first, "blocks": tuple(residual), "output": final}


def probe_predict(params, inputs):
    import jax

    hidden = jax.nn.relu(inputs @ params["input"]["w"] + params["input"]["b"])
    for block in params["blocks"]:
        residual = jax.nn.relu(hidden @ block["w1"] + block["b1"])
        residual = residual @ block["w2"] + block["b2"]
        hidden = jax.nn.relu(hidden + residual)
    return hidden @ params["output"]["w"] + params["output"]["b"]


def score_errors(target: np.ndarray, prediction: np.ndarray):
    raw = np.mean(np.sum(np.square(target - prediction), axis=-1), axis=1)
    energy = np.mean(np.sum(np.square(target), axis=-1), axis=1)
    return raw, energy, raw / np.maximum(energy, 1e-12)


def fit_independent_probes(
    fit_x: np.ndarray,
    fit_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    *,
    hidden_dim: int,
    residual_blocks: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    validation_interval: int,
    patience_evaluations: int,
    seed: int,
):
    """Fit per-agent residual MLPs; select on validation and test only once."""

    import jax
    import jax.numpy as jnp
    import optax

    splits = ((fit_x, fit_y), (validation_x, validation_y), (test_x, test_y))
    agents = fit_x.shape[0]
    if any(x.ndim != 3 or y.ndim != 3 or x.shape[:2] != y.shape[:2] for x, y in splits):
        raise ValueError("Probe arrays must have matching agent/sample axes")
    if any(x.shape[0] != agents or x.shape[-1] != fit_x.shape[-1] for x, _ in splits):
        raise ValueError("Probe input dimensions or agent counts differ")
    if any(y.shape[-1] != fit_y.shape[-1] for _, y in splits):
        raise ValueError("Probe target dimensions differ")
    if (
        min(
            hidden_dim,
            residual_blocks,
            steps,
            batch_size,
            validation_interval,
            patience_evaluations,
        )
        <= 0
        or learning_rate <= 0
    ):
        raise ValueError("Probe optimization hyperparameters must be positive")

    fit_count = fit_x.shape[1]
    keys = jax.random.split(jax.random.PRNGKey(seed), agents)
    params = jax.vmap(
        lambda key: initialize_probe(
            key, fit_x.shape[-1], hidden_dim, fit_y.shape[-1], residual_blocks
        )
    )(keys)
    optimizer = optax.adam(learning_rate)
    optimizer_state = optimizer.init(params)
    best_params = params
    best_validation = np.full(agents, np.inf, dtype=np.float64)
    best_steps = np.zeros(agents, dtype=np.int32)
    stale = np.zeros(agents, dtype=np.int32)
    history = []
    random = np.random.default_rng(seed)
    device_fit_x, device_fit_y = jnp.asarray(fit_x), jnp.asarray(fit_y)
    device_validation_x = jnp.asarray(validation_x)
    device_validation_y = jnp.asarray(validation_y)
    axis = jnp.arange(agents)[:, None]

    @jax.jit
    def update(current_params, current_state, x, y):
        def one_loss(agent_params, agent_x, agent_y):
            prediction = probe_predict(agent_params, agent_x)
            return jnp.mean(jnp.sum(jnp.square(prediction - agent_y), axis=-1))

        losses, gradients = jax.vmap(jax.value_and_grad(one_loss))(current_params, x, y)
        updates, current_state = optimizer.update(
            gradients, current_state, current_params
        )
        return optax.apply_updates(current_params, updates), current_state, losses

    @jax.jit
    def validation_error(current_params):
        prediction = jax.vmap(probe_predict)(current_params, device_validation_x)
        return jnp.mean(
            jnp.sum(jnp.square(prediction - device_validation_y), axis=-1), axis=1
        )

    for step in range(1, steps + 1):
        indices = random.choice(
            fit_count, size=min(batch_size, fit_count), replace=False
        )
        x = device_fit_x[axis, jnp.asarray(indices)[None, :]]
        y = device_fit_y[axis, jnp.asarray(indices)[None, :]]
        params, optimizer_state, _ = update(params, optimizer_state, x, y)
        if step % validation_interval and step != steps:
            continue
        validation_raw = np.asarray(validation_error(params), dtype=np.float64)
        improved = validation_raw < best_validation - 1e-8
        select = jnp.asarray(improved)
        best_params = jax.tree.map(
            lambda old, new: jnp.where(
                select.reshape((agents,) + (1,) * (old.ndim - 1)), new, old
            ),
            best_params,
            params,
        )
        best_validation = np.where(improved, validation_raw, best_validation)
        best_steps = np.where(improved, step, best_steps)
        stale = np.where(improved, 0, stale + 1)
        history.append(
            {
                "step": step,
                "validation_raw_mean": float(validation_raw.mean()),
                "best_validation_raw_mean": float(best_validation.mean()),
            }
        )
        if np.all(stale >= patience_evaluations):
            break

    predict_all = jax.jit(jax.vmap(probe_predict))
    result = {"best_step": best_steps, "history": history, "steps_executed": step}
    # Test is not read by the optimizer or early stopping; its first prediction
    # happens only after the best validation-selected weights have been frozen.
    for name, x, target in (
        ("fit", fit_x, fit_y),
        ("validation", validation_x, validation_y),
        ("test", test_x, test_y),
    ):
        prediction = np.asarray(predict_all(best_params, jnp.asarray(x)))
        raw, energy, normalized = score_errors(target, prediction)
        result[f"{name}_raw"] = raw
        result[f"{name}_energy"] = energy
        result[f"{name}_normalized"] = normalized
    return result


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
    if condition in {
        "c_to_a_mse",
        "c_to_a_cka",
        "score_recovery",
        "actor_score_recovery",
    }:
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
    validation_fraction: float,
    split_seed: int,
    sampling_seed: int,
    fit_samples_per_agent: int,
    validation_samples_per_agent: int,
    test_samples_per_agent: int,
    fisher_ridge: float,
    probe_hidden_dim: int,
    probe_residual_blocks: int,
    probe_steps: int,
    probe_batch_size: int,
    probe_learning_rate: float,
    probe_validation_interval: int,
    probe_patience_evaluations: int,
    probe_seed: int,
):
    metadata, arrays = load_diagnostics(diagnostics_dir, ARRAYS)
    if metadata.get("array_profile") not in ("score_recoverability", "full"):
        raise RuntimeError("Unsupported diagnostic array profile")
    if bool(metadata.get("actor_parameter_sharing")):
        raise RuntimeError("The preregistered score-recoverability scope is NPS only")
    data = prepare_probe_data(
        arrays,
        fit_fraction=fit_fraction,
        validation_fraction=validation_fraction,
        split_seed=split_seed,
        sampling_seed=sampling_seed,
        fit_samples_per_agent=fit_samples_per_agent,
        validation_samples_per_agent=validation_samples_per_agent,
        test_samples_per_agent=test_samples_per_agent,
        fisher_ridge=fisher_ridge,
    )
    result = fit_independent_probes(
        data["fit_x"],
        data["fit_y"],
        data["validation_x"],
        data["validation_y"],
        data["test_x"],
        data["test_y"],
        hidden_dim=probe_hidden_dim,
        residual_blocks=probe_residual_blocks,
        steps=probe_steps,
        batch_size=probe_batch_size,
        learning_rate=probe_learning_rate,
        validation_interval=probe_validation_interval,
        patience_evaluations=probe_patience_evaluations,
        seed=probe_seed,
    )
    reward = np.asarray(arrays["reward"], dtype=np.float64)
    active = np.asarray(arrays["active"], dtype=bool)
    if reward.shape[:2] != active.shape or reward.ndim != 3:
        raise RuntimeError("Collected reward/active arrays are misaligned")
    episode_returns = (reward[:, :, 0] * active).sum(axis=1)
    task = str(metadata["map_name"])
    condition = canonical_condition(metadata)
    seed = int(metadata["training_seed"])
    checkpoint_step = int(
        metadata.get("checkpoint_nominal_env_step")
        or metadata.get("checkpoint_env_step")
    )
    rows = []
    for agent, fisher in enumerate(data["fisher_audit"]):
        rows.append(
            {
                "task": task,
                "condition": condition,
                "seed": seed,
                "checkpoint_env_step": checkpoint_step,
                "agent_id": agent,
                "epsilon_rec": float(result["test_raw"][agent]),
                "epsilon_rec_normalized": float(result["test_normalized"][agent]),
                "score_energy": float(result["test_energy"][agent]),
                "fit_epsilon_rec": float(result["fit_raw"][agent]),
                "fit_epsilon_rec_normalized": float(result["fit_normalized"][agent]),
                "fit_score_energy": float(result["fit_energy"][agent]),
                "validation_epsilon_rec": float(result["validation_raw"][agent]),
                "validation_epsilon_rec_normalized": float(
                    result["validation_normalized"][agent]
                ),
                "validation_score_energy": float(result["validation_energy"][agent]),
                "probe_best_step": int(result["best_step"][agent]),
                "estimator_failure_gt_one": bool(result["test_normalized"][agent] > 1),
                "fit_samples": fit_samples_per_agent,
                "validation_samples": validation_samples_per_agent,
                "test_samples": test_samples_per_agent,
                **fisher,
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "agent_metrics.csv", rows)
    masks = data["episode_masks"]
    protocol = {
        "schema_version": 2,
        "protocol": "smax-score-recoverability-resmlp-v2.0",
        "task": task,
        "condition": condition,
        "training_seed": seed,
        "checkpoint": metadata["checkpoint"],
        "checkpoint_env_step": checkpoint_step,
        "diagnostics_dir": str(Path(diagnostics_dir).resolve()),
        "diagnostic_seed": metadata["diagnostic_seed"],
        "episodes": int(metadata["episodes"]),
        "on_policy_episode_return_mean": float(episode_returns.mean()),
        "on_policy_episode_return_se": float(
            episode_returns.std(ddof=1) / math.sqrt(len(episode_returns))
        ),
        "fit_fraction": fit_fraction,
        "validation_fraction": validation_fraction,
        "test_fraction": 1.0 - fit_fraction - validation_fraction,
        "split_episodes": {name: int(mask.sum()) for name, mask in masks.items()},
        "split_seed": split_seed,
        "sampling_seed": sampling_seed,
        "fit_samples_per_agent": fit_samples_per_agent,
        "validation_samples_per_agent": validation_samples_per_agent,
        "test_samples_per_agent": test_samples_per_agent,
        "fisher_source": "fit_split_only",
        "fisher_ridge_absolute": fisher_ridge,
        "score_source": "exact_masked_categorical_d_log_prob_d_actor_latent",
        "probe_input": "fit_standardized_critic_latent_plus_one_hot_action",
        "probe_architecture": "dense_relu_three_residual_blocks_dense",
        "probe_hidden_dim": probe_hidden_dim,
        "probe_residual_blocks": probe_residual_blocks,
        "probe_max_steps": probe_steps,
        "probe_batch_size": probe_batch_size,
        "probe_learning_rate": probe_learning_rate,
        "probe_validation_interval": probe_validation_interval,
        "probe_patience_evaluations": probe_patience_evaluations,
        "probe_seed": probe_seed,
        "probe_best_steps_per_agent": result["best_step"].tolist(),
        "probe_steps_executed": result["steps_executed"],
        "probe_test_used_for_selection": False,
        "agent_aggregation": "unweighted_mean",
        "epsilon_rec": float(np.mean(result["test_raw"])),
        "epsilon_rec_normalized": float(np.mean(result["test_normalized"])),
        "fit_epsilon_rec_normalized": float(np.mean(result["fit_normalized"])),
        "validation_epsilon_rec_normalized": float(
            np.mean(result["validation_normalized"])
        ),
        "estimator_failure_agents_gt_one": int(np.sum(result["test_normalized"] > 1)),
        "num_agents": len(rows),
        "optimization_history": result["history"],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return rows, protocol
