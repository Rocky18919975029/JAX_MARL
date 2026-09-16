#!/usr/bin/env python3
"""Audit H1 latent distortion and Linear CKA with slot-by-type conditioning.

This is an offline measurement over already collected held-out diagnostics.  It
does not modify the training objective or overwrite any canonical H1 output.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from h1_diagnostic_data import TYPE_CONDITIONING_ARRAYS
    from h1_latent_distortion import (
        REFERENCE_PROTOCOL,
        fisher_metrics_from_statistics,
        load_training_protocol,
        reconstruct_training_gae,
    )
except ModuleNotFoundError:
    from scripts.h1_diagnostic_data import TYPE_CONDITIONING_ARRAYS
    from scripts.h1_latent_distortion import (
        REFERENCE_PROTOCOL,
        fisher_metrics_from_statistics,
        load_training_protocol,
        reconstruct_training_gae,
    )


AUDIT_PROTOCOL = "h1-slot-type-conditioning-v1.0"
LAYER_NORM_EPSILON = 1e-5


@dataclass
class FisherAccumulator:
    """Streaming sufficient statistics for one score-function group."""

    count: int = 0
    sum_reference: np.ndarray | None = None
    sum_critic: np.ndarray | None = None
    sum_outer: np.ndarray | None = None

    def update(self, scores, reference_signal, critic_signal):
        scores = np.asarray(scores, dtype=np.float64)
        reference_signal = np.asarray(reference_signal, dtype=np.float64)
        critic_signal = np.asarray(critic_signal, dtype=np.float64)
        if scores.ndim != 2 or not len(scores):
            return
        if len(reference_signal) != len(scores) or len(critic_signal) != len(scores):
            raise ValueError("Fisher group sample counts do not match")
        dimension = scores.shape[1]
        if self.sum_reference is None:
            self.sum_reference = np.zeros(dimension, dtype=np.float64)
            self.sum_critic = np.zeros(dimension, dtype=np.float64)
            self.sum_outer = np.zeros((dimension, dimension), dtype=np.float64)
        elif dimension != len(self.sum_reference):
            raise ValueError("Actor score dimensions changed between shards")
        self.count += len(scores)
        self.sum_reference += scores.T @ reference_signal
        self.sum_critic += scores.T @ critic_signal
        self.sum_outer += scores.T @ scores

    def statistics(self):
        if self.count <= 0 or self.sum_reference is None:
            raise ValueError("Cannot finalize an empty Fisher group")
        fisher = self.sum_outer / self.count
        fisher = (fisher + fisher.T) / 2.0
        g_reference = self.sum_reference / self.count
        g_critic = self.sum_critic / self.count
        return {
            "count": self.count,
            "dimension": len(g_reference),
            "g_reference": g_reference,
            "g_critic": g_critic,
            "delta": g_reference - g_critic,
            "fisher": fisher,
            "eigenvalues": np.linalg.eigvalsh(fisher),
        }


@dataclass
class LinearCKAAccumulator:
    """Streaming centered Linear CKA sufficient statistics."""

    count: int = 0
    sum_actor: np.ndarray | None = None
    sum_critic: np.ndarray | None = None
    actor_self: np.ndarray | None = None
    critic_self: np.ndarray | None = None
    cross: np.ndarray | None = None

    @staticmethod
    def normalize(samples):
        samples = np.asarray(samples, dtype=np.float64)
        mean = samples.mean(axis=-1, keepdims=True)
        variance = np.square(samples - mean).mean(axis=-1, keepdims=True)
        return (samples - mean) / np.sqrt(variance + LAYER_NORM_EPSILON)

    def update(self, actor, critic):
        actor = self.normalize(actor)
        critic = self.normalize(critic)
        if actor.ndim != 2 or critic.ndim != 2 or not len(actor):
            return
        if len(actor) != len(critic):
            raise ValueError("Actor and critic CKA sample counts do not match")
        if self.sum_actor is None:
            self.sum_actor = np.zeros(actor.shape[1], dtype=np.float64)
            self.sum_critic = np.zeros(critic.shape[1], dtype=np.float64)
            self.actor_self = np.zeros(
                (actor.shape[1], actor.shape[1]), dtype=np.float64
            )
            self.critic_self = np.zeros(
                (critic.shape[1], critic.shape[1]), dtype=np.float64
            )
            self.cross = np.zeros((actor.shape[1], critic.shape[1]), dtype=np.float64)
        elif actor.shape[1] != len(self.sum_actor) or critic.shape[1] != len(
            self.sum_critic
        ):
            raise ValueError("Latent dimensions changed between shards")
        self.count += len(actor)
        self.sum_actor += actor.sum(axis=0)
        self.sum_critic += critic.sum(axis=0)
        self.actor_self += actor.T @ actor
        self.critic_self += critic.T @ critic
        self.cross += actor.T @ critic

    def metrics(self, epsilon):
        if self.count <= 1 or self.sum_actor is None:
            raise ValueError("Linear CKA requires at least two samples per group")
        actor_self = (
            self.actor_self - np.outer(self.sum_actor, self.sum_actor) / self.count
        )
        critic_self = (
            self.critic_self - np.outer(self.sum_critic, self.sum_critic) / self.count
        )
        cross = self.cross - np.outer(self.sum_actor, self.sum_critic) / self.count
        numerator = float(np.square(cross).sum())
        denominator = float(
            np.sqrt(np.square(actor_self).sum()) * np.sqrt(np.square(critic_self).sum())
        )
        similarity = numerator / (denominator + epsilon)
        return {
            "count": self.count,
            "similarity": similarity,
            "distance": 1.0 - similarity,
            "numerator": numerator,
            "denominator": denominator,
        }


def nominal_step(metadata, diagnostics_dir):
    value = metadata.get("checkpoint_nominal_env_step")
    if value is not None:
        return int(value)
    name = diagnostics_dir.name
    if name == "initial":
        return 0
    if name == "final":
        config = json.loads(
            (Path(metadata["checkpoint"]) / "config.json").read_text(encoding="utf-8")
        )
        return int(config["TOTAL_TIMESTEPS"])
    if name.startswith("step_"):
        return int(name.removeprefix("step_"))
    raise RuntimeError(f"Cannot infer checkpoint step from {diagnostics_dir}")


def _group(mapping, key, factory):
    if key not in mapping:
        mapping[key] = factory()
    return mapping[key]


def _validate_type_ids(values):
    values = np.asarray(values)
    integers = values.astype(np.int64)
    if not np.array_equal(values, integers):
        raise ValueError("Unit type identifiers are not integer-valued")
    return integers


def compute_audit(diagnostics_dir, fisher_ridge_absolute, cka_epsilon=1e-8):
    diagnostics_dir = Path(diagnostics_dir).expanduser().resolve()
    metadata = json.loads(
        (diagnostics_dir / "metadata.json").read_text(encoding="utf-8")
    )
    if metadata.get("actor_parameter_sharing"):
        raise ValueError("The H1 type-conditioning audit is restricted to NPS")
    if fisher_ridge_absolute <= 0 or cka_epsilon <= 0:
        raise ValueError("Fisher ridge and CKA epsilon must be positive")
    gamma, gae_lambda, rollout_steps = load_training_protocol(metadata)
    agents = int(metadata["num_agents"])
    fisher_slot = {agent: FisherAccumulator() for agent in range(agents)}
    fisher_type = {}
    cka_slot = {agent: LinearCKAAccumulator() for agent in range(agents)}
    cka_type = {}
    episode_returns = []
    within_episode_type_changes = 0

    required = set(TYPE_CONDITIONING_ARRAYS)
    for shard in metadata.get("shards", []):
        shard_path = diagnostics_dir / shard["path"]
        with np.load(shard_path) as data:
            missing = required - set(data.files)
            if missing:
                raise RuntimeError(f"Missing {sorted(missing)} from {shard_path}")
            active = np.asarray(data["active"], dtype=bool)
            alive = np.asarray(data["alive"], dtype=bool)
            reward = np.asarray(data["reward"])
            value = np.asarray(data["value"])
            global_done = np.asarray(data["global_done"], dtype=bool)
            team_return = np.asarray(data["mc_return"][:, :, 0], dtype=np.float64)
            score = np.asarray(data["actor_score"])
            actor_latent = np.asarray(data["actor_latent"])
            critic_latent = np.asarray(data["critic_latent"])
            unit_type = _validate_type_ids(data["state_unit_types"][:, :, :agents])

            if alive.shape[:3] != unit_type.shape:
                raise ValueError(f"Alive/type axes disagree in {shard_path}")
            training_gae = reconstruct_training_gae(
                reward,
                value,
                global_done,
                active,
                gamma,
                gae_lambda,
                rollout_steps,
            )
            episode_returns.extend(
                np.sum(
                    np.where(active, reward[:, :, 0], 0.0),
                    axis=1,
                    dtype=np.float64,
                ).tolist()
            )

            # In SMACv2 a slot's type is sampled at reset and remains fixed for
            # that episode.  Count violations explicitly so a malformed
            # collection cannot masquerade as type-conditioned evidence.
            for episode in range(len(active)):
                active_steps = np.flatnonzero(active[episode])
                if not len(active_steps):
                    continue
                first = unit_type[episode, active_steps[0]]
                within_episode_type_changes += int(
                    np.any(unit_type[episode, active_steps] != first, axis=0).sum()
                )

            for agent in range(agents):
                mask = active & alive[:, :, agent]
                if not mask.any():
                    raise RuntimeError(f"No valid samples for actor slot {agent}")
                valid_score = score[:, :, agent][mask]
                valid_reference = team_return[mask]
                valid_critic = training_gae[:, :, agent][mask]
                valid_actor_latent = actor_latent[:, :, agent][mask]
                valid_critic_latent = critic_latent[:, :, agent][mask]
                valid_types = unit_type[:, :, agent][mask]

                fisher_slot[agent].update(valid_score, valid_reference, valid_critic)
                cka_slot[agent].update(valid_actor_latent, valid_critic_latent)
                for type_id in np.unique(valid_types):
                    type_mask = valid_types == type_id
                    key = (agent, int(type_id))
                    _group(fisher_type, key, FisherAccumulator).update(
                        valid_score[type_mask],
                        valid_reference[type_mask],
                        valid_critic[type_mask],
                    )
                    _group(cka_type, key, LinearCKAAccumulator).update(
                        valid_actor_latent[type_mask], valid_critic_latent[type_mask]
                    )

    if within_episode_type_changes:
        raise RuntimeError(
            f"Observed {within_episode_type_changes} within-episode slot type changes"
        )
    if not episode_returns:
        raise RuntimeError(f"No diagnostic episodes found under {diagnostics_dir}")

    per_slot = []
    per_slot_type = []
    epsilon_slot = 0.0
    epsilon_slot_type = 0.0
    cka_slot_distances = []
    cka_slot_type_distances = []
    all_types = set()
    for agent in range(agents):
        slot_fisher = fisher_metrics_from_statistics(
            fisher_slot[agent].statistics(), fisher_ridge_absolute
        )
        slot_cka = cka_slot[agent].metrics(cka_epsilon)
        epsilon_slot += slot_fisher["epsilon_lat"]
        cka_slot_distances.append(slot_cka["distance"])
        agent_type_keys = sorted(key for key in fisher_type if key[0] == agent)
        total = sum(fisher_type[key].count for key in agent_type_keys)
        weighted_epsilon = 0.0
        weighted_cka_distance = 0.0
        for key in agent_type_keys:
            if fisher_type[key].count != cka_type[key].count:
                raise RuntimeError(f"Fisher and CKA masks disagree for {key}")
            weight = fisher_type[key].count / total
            type_fisher = fisher_metrics_from_statistics(
                fisher_type[key].statistics(), fisher_ridge_absolute
            )
            type_cka = cka_type[key].metrics(cka_epsilon)
            weighted_epsilon += weight * type_fisher["epsilon_lat"]
            weighted_cka_distance += weight * type_cka["distance"]
            all_types.add(key[1])
            per_slot_type.append(
                {
                    "agent_id": agent,
                    "unit_type": key[1],
                    "within_slot_weight": weight,
                    "num_valid_samples": fisher_type[key].count,
                    "epsilon_lat": type_fisher["epsilon_lat"],
                    "gradient_cosine": type_fisher["gradient_cosine"],
                    "linear_cka_similarity": type_cka["similarity"],
                    "linear_cka_distance": type_cka["distance"],
                }
            )
        epsilon_slot_type += weighted_epsilon
        cka_slot_type_distances.append(weighted_cka_distance)
        per_slot.append(
            {
                "agent_id": agent,
                "num_valid_samples": fisher_slot[agent].count,
                "num_observed_types": len(agent_type_keys),
                "epsilon_lat_slot": slot_fisher["epsilon_lat"],
                "epsilon_lat_slot_type": weighted_epsilon,
                "linear_cka_distance_slot": slot_cka["distance"],
                "linear_cka_distance_slot_type": weighted_cka_distance,
            }
        )

    cka_distance_slot = float(np.mean(cka_slot_distances))
    cka_distance_slot_type = float(np.mean(cka_slot_type_distances))
    returns = np.asarray(episode_returns, dtype=np.float64)
    canonical_comparison = {"available": False}
    canonical_path = diagnostics_dir / "latent_summary.json"
    if canonical_path.is_file():
        canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
        if canonical.get("reference_protocol") == REFERENCE_PROTOCOL and math.isclose(
            float(canonical.get("fisher_ridge_absolute", math.nan)),
            float(fisher_ridge_absolute),
        ):
            canonical_epsilon = float(canonical["epsilon_lat"])
            absolute_error = abs(canonical_epsilon - epsilon_slot)
            if not math.isclose(
                canonical_epsilon,
                epsilon_slot,
                rel_tol=1e-8,
                abs_tol=1e-10,
            ):
                raise RuntimeError(
                    "Streaming slot-pooled epsilon_Lat does not reproduce the "
                    f"canonical diagnostic: {epsilon_slot} != {canonical_epsilon}"
                )
            canonical_comparison = {
                "available": True,
                "canonical_epsilon_lat": canonical_epsilon,
                "absolute_error": absolute_error,
            }
    result = {
        "schema_version": 1,
        "audit_protocol": AUDIT_PROTOCOL,
        "source_diagnostics_dir": str(diagnostics_dir),
        "run_id": metadata.get("run_id"),
        "run_name": metadata["run_name"],
        "task": metadata["map_name"],
        "actor_parameterization": "nps",
        "condition": metadata["condition"],
        "align_distance": metadata.get("align_distance", "ln_mse"),
        "alignment_coef": float(metadata["alignment_coef"]),
        "seed": int(metadata["training_seed"]),
        "nominal_step": nominal_step(metadata, diagnostics_dir),
        "checkpoint_step": int(
            metadata.get("checkpoint_env_step")
            or nominal_step(metadata, diagnostics_dir)
        ),
        "reference_protocol": REFERENCE_PROTOCOL,
        "fisher_ridge_absolute": float(fisher_ridge_absolute),
        "linear_cka_epsilon": float(cka_epsilon),
        "layer_norm_epsilon": LAYER_NORM_EPSILON,
        "heldout_episodes": len(returns),
        "heldout_return_mean": float(returns.mean()),
        "heldout_return_std": float(returns.std(ddof=1)),
        "epsilon_lat_slot": float(epsilon_slot),
        "epsilon_lat_slot_type": float(epsilon_slot_type),
        "epsilon_lat_type_minus_slot": float(epsilon_slot_type - epsilon_slot),
        "linear_cka_distance_slot": cka_distance_slot,
        "linear_cka_distance_slot_type": cka_distance_slot_type,
        "linear_cka_type_minus_slot": cka_distance_slot_type - cka_distance_slot,
        "observed_unit_types": sorted(all_types),
        "num_agents": agents,
        "within_episode_type_changes": within_episode_type_changes,
        "canonical_slot_reproduction": canonical_comparison,
        "aggregation": {
            "slot": "equal mean across NPS actor slots for CKA; sum across slots for epsilon_lat",
            "slot_type": "within each slot, sample-frequency weighted mean across unit types; never pool slots",
            "epsilon_lat_slot_definition": "sum_i D_i",
            "epsilon_lat_slot_type_definition": "sum_i sum_k p_ik D_ik",
            "linear_cka_slot_definition": "mean_i (1 - CKA_i)",
            "linear_cka_slot_type_definition": "mean_i sum_k p_ik (1 - CKA_ik)",
        },
        "per_slot": per_slot,
        "per_slot_type": per_slot_type,
    }
    for field in (
        "epsilon_lat_slot",
        "epsilon_lat_slot_type",
        "linear_cka_distance_slot",
        "linear_cka_distance_slot_type",
    ):
        if not math.isfinite(result[field]):
            raise RuntimeError(f"Non-finite audit metric {field}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--fisher-ridge-absolute", type=float, required=True)
    parser.add_argument("--cka-epsilon", type=float, default=1e-8)
    args = parser.parse_args()
    result = compute_audit(
        args.diagnostics_dir,
        args.fisher_ridge_absolute,
        args.cka_epsilon,
    )
    output = args.output_json.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
