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

## Paired high-agent benchmark experiment

`run_matched_optimal.py` selects the largest-agent standard tasks in the
downloaded [Sable benchmark optimal on-policy parameter table](
https://sites.google.com/view/sable-marl): LBF `15x15-4p-5f` (4 agents) and
RWARE `large-8ag` (8 agents). The former has `fov: 15` on a 15x15 grid, so do
not describe it as a strongly partially observed LBF task. Its task-level
recurrent MAPPO settings and RWARE's are transcribed into
`optimal_rec_mappo.json`. Both conditions use the same scenario, network,
training and evaluation settings for each task and seed. The **only**
condition-specific changes are the ARec entry point and its four auxiliary
settings. The baseline is upstream Mava's original `rec_mappo.py`.

The common formal protocol is 64 environments per learner batch, 2 learner
batches, 128 rollout steps, 1220 updates (19,988,480 environment steps per
run), 122 evaluations and 32 episodes per evaluation. The remaining published
task-level settings are in `optimal_rec_mappo.json`; the launcher records the
complete resolved overrides, source hashes and Mava commit in a manifest.
The ARec coefficient `1e-4` and head settings below come from the earlier
wiring smoke, **not** from task-specific tuning. A same-seed paired comparison
controls the MAPPO configuration; it does not make ARec tuned or guarantee a
gain. The ARec run will consume more wall-clock compute for the same number
of environment transitions.

From the JaxMARL checkout on the server, after installing the two ARec files
as above, run one-seed low-cost smoke for both tasks first:

```bash
unset LD_LIBRARY_PATH
MAVA_ROOT=/home/data/zeshenghong/Mava
SMOKE_ROOT=/home/data/zeshenghong/JaxMARL/mava_high_agent_paired/smoke_v1
python experiments/mava_jumanji/run_matched_optimal.py run \
  --mava-root "$MAVA_ROOT" --run-root "$SMOKE_ROOT" \
  --seeds 1 --gpus 1 --max-runs-per-gpu 1 --smoke
python experiments/mava_jumanji/run_matched_optimal.py status --run-root "$SMOKE_ROOT"
```

The smoke reduces environments, batches, epochs and updates for a quick
integration check. It is **not** a sample of the formal optimal-protocol
learning curve. Once all four smoke jobs show `COMPLETED`, run the 16 formal
jobs (2 tasks × 2 conditions × 4 paired seeds):

```bash
RUN_ROOT=/home/data/zeshenghong/JaxMARL/mava_high_agent_paired/formal_4seed_v1
mkdir -p "$RUN_ROOT"
nohup python -u experiments/mava_jumanji/run_matched_optimal.py run \
  --mava-root "$MAVA_ROOT" --run-root "$RUN_ROOT" \
  --seeds 1-4 --gpus 0,1,2,3 --max-runs-per-gpu 1 \
  > "$RUN_ROOT/launcher.stdout" 2>&1 &
echo "Launcher PID: $!"
watch -n 5 "python experiments/mava_jumanji/run_matched_optimal.py status --run-root '$RUN_ROOT'"
```

Results stay under `$RUN_ROOT/runs/<task>--<condition>--seedN/`, each worker's
stdout/stderr under `logs/`, and per-run state under `status/`. The launcher
orders jobs by seed, skips completed runs on restart, and continues after a
failed run. Inspect that run's log; then rerun the same command with
`--retry-failed` to retry failures without duplicating successful runs. Each
worker sees one GPU, even when multiple workers are allowed per GPU.
