# Mava recurrent MAPPO + actor-side score recovery (ARec)

This experiment adds **only ARec** to Mava's recurrent MAPPO. The unmodified
`mava/systems/ppo/anakin/rec_mappo.py` is the `none` baseline. The two runs
must use the same Mava checkout, Jumanji scenario, training budget, and seeds.
The implementation is pinned to Mava commit
`9f67e612654ecb7b7d45ff8052ce9ccfc6c68d93`.

## What changes

The actor and centralised critic have the same parameter layout and ordinary
PPO losses as upstream Mava. At the end of each on-policy rollout, ARec
computes the masked representation-level policy score from the actor's GRU
output. A per-agent empirical Fisher matrix is estimated across the **whole**
update batch (all rollout shards, learner batches and devices), regularised,
and inverse-square-rooted. That whitening matrix stays
fixed for the subsequent PPO epochs.

Each agent has its own recovery head `q_i(z_critic, action)`. For each rollout,
the heads first fit the stopped Fisher-normalised score target while the actor
and critic are frozen. The heads are then stopped and the actor minimises
`PPO_actor_loss + arec.coef * mean(||u_actor - q_i||²)`. This second loss
differentiates through the actor score (including the second-order path to the
actor encoder/head) but **not** through the Fisher eigendecomposition, critic,
or recovery heads. The critic receives only the original value loss. No latent
alignment is used.

The ARec configuration adds only four settings: `arec.coef`, `arec.q_steps`,
`arec.q_lr`, and `arec.fisher_ridge`. The other defaults are Mava's original
`rec_mappo` defaults. Because Mava's default checkpoint loader cannot restore
the additional recovery-head state, this entry point explicitly rejects
`logger.checkpointing.load_model=true` rather than silently doing a partial
restore. Saving a checkpoint is supported through the full learner state.

## Install in the pinned Mava checkout

From the JaxMARL repository on the server:

```bash
cd ~/JaxMARL
git pull --ff-only
MAVA_ROOT=/home/data/zeshenghong/Mava
git -C "$MAVA_ROOT" rev-parse HEAD
install -m 0644 experiments/mava_jumanji/rec_mappo_arec.py \
  "$MAVA_ROOT/mava/systems/ppo/anakin/rec_mappo_arec.py"
install -m 0644 experiments/mava_jumanji/rec_mappo_arec.yaml \
  "$MAVA_ROOT/mava/configs/default/rec_mappo_arec.yaml"
```

Check the printed Mava commit against the pin above before copying. This does
not edit Mava's baseline script or config.

## Two-environment smoke test

From `$MAVA_ROOT` (choose currently free GPUs):

```bash
cd /home/data/zeshenghong/Mava
unset LD_LIBRARY_PATH
CUDA_VISIBLE_DEVICES=0 uv run --extra cuda12 python \
  mava/systems/ppo/anakin/rec_mappo_arec.py \
  env=lbf system.seed=1 system.num_updates=3 \
  arch.num_envs=2 arch.num_evaluation=1 arch.num_eval_episodes=2 \
  arch.absolute_metric=false system.update_batch_size=1 \
  system.rollout_length=8 system.ppo_epochs=1 system.num_minibatches=1

CUDA_VISIBLE_DEVICES=1 uv run --extra cuda12 python \
  mava/systems/ppo/anakin/rec_mappo_arec.py \
  env=rware system.seed=1 system.num_updates=3 \
  arch.num_envs=2 arch.num_evaluation=1 arch.num_eval_episodes=2 \
  arch.absolute_metric=false system.update_batch_size=1 \
  system.rollout_length=8 system.ppo_epochs=1 system.num_minibatches=1
```

The directly comparable `none` baseline is obtained by replacing
`rec_mappo_arec.py` with Mava's existing `rec_mappo.py` in these commands,
keeping every other override the same. For a full experiment, set a common
budget and multiple paired seeds. Do not infer a performance benefit from a
three-update smoke test.

We verified one compiled learner update locally on both LBF and RWARE using
the pinned Mava checkout. The server CUDA smoke and longer runs remain to be
run by the user.
