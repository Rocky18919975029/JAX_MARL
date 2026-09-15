"""Read only the arrays used by a canonical H1 diagnostic stage."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


LATENT_ARRAYS = (
    "active",
    "alive",
    "diagnostic_episode_id",
    "mc_return",
    "reward",
    "value",
    "global_done",
    "actor_score",
)
DECISION_ARRAYS = (
    "active",
    "alive",
    "available_actions",
    "local_observation",
    "world_state",
    "actor_hidden_before",
    "actor_latent",
    "state_done",
    "state_step",
    "state_unit_positions",
    "state_unit_alive",
    "state_unit_teams",
    "state_unit_health",
    "state_unit_types",
    "state_unit_weapon_cooldowns",
    "state_prev_movement_actions",
    "state_prev_attack_actions",
    "enemy_default_target",
    "enemy_last_attacked_enemy",
)
BELLMAN_ARRAYS = (
    "active",
    "alive",
    "critic_latent",
    "reward",
    "global_done",
    "mc_return",
)


def load_diagnostics(directory, names):
    directory = Path(directory).expanduser().resolve()
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    arrays = {name: [] for name in names}
    for shard in metadata["shards"]:
        path = directory / shard["path"]
        with np.load(path) as data:
            absent = set(names) - set(data.files)
            if absent:
                raise RuntimeError(
                    f"Missing {sorted(absent)} from collected shard {path}"
                )
            for name in names:
                arrays[name].append(np.asarray(data[name]))
    if not metadata["shards"]:
        raise RuntimeError(f"No collected episode shards under {directory}")
    return metadata, {
        name: np.concatenate(parts, axis=0) for name, parts in arrays.items()
    }
