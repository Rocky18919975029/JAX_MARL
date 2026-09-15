# SMAX MAPPO

This fork includes the changes used for the tested SMAX MAPPO setup:

- a terminal progress bar with environment steps, win rate, and ETA;
- compatibility for Distrax/TFP with JAX 0.10.2 without editing `site-packages`;
- configurable recurrent-state width instead of a hard-coded width of 128.
- selectable shared or independent actor parameters with a centralized critic.

## Actor parameter sharing

The official parameter-sharing behavior remains the default:

```bash
ACTOR_PARAMETER_SHARING=true
```

To give every allied agent its own complete actor encoder, GRU, and action head
while retaining the shared centralized critic, use:

```bash
ACTOR_PARAMETER_SHARING=false
```

In independent-actor mode, each actor has its own parameters, Adam state,
gradient clipping, and advantage normalization. Minibatches retain a separate,
equal set of environment trajectories for every actor.

## Matched actor-sharing comparison

For a controlled shared-versus-independent actor experiment, enable:

```bash
MATCHED_COMPARISON=true
```

Run this once with `ACTOR_PARAMETER_SHARING=true` and once with
`ACTOR_PARAMETER_SHARING=false` for every seed. With the same seed, both runs
use:

- identical initial actor parameters (the independent run copies the same
  initial parameters to every actor);
- identical per-agent action-sampling keys and agent-wise actor execution;
- identical environment-stratified minibatches and critic update order;
- identical global, cross-agent advantage normalization in every minibatch.

The shared run still uses one actor optimizer, while the independent run uses
one optimizer per actor. Gradient norms, clipping rates, parameter-update norms,
and global/per-agent advantage statistics are logged to W&B to quantify the
remaining optimization differences.

## Actor-critic representation alignment

Alignment experiments require `MATCHED_COMPARISON=true`. Select one mode with:

```bash
ALIGN_MODE=none        # matched baseline
ALIGN_MODE=c_to_a      # frozen critic rollout latent teaches actor
ALIGN_MODE=a_to_c      # frozen actor rollout latent teaches critic
ALIGN_MODE=reciprocal  # both frozen-target directions
ALIGN_MODE=joint       # current actor and critic latents jointly align
ALIGNMENT_COEF=0.1
```

Both representations are the 128-dimensional post-GRU, pre-head latent. The
distance is per-sample layer-normalized MSE, averaged only over currently alive
agents. Frozen targets are saved during rollout and remain fixed across all PPO
minibatches and epochs in that update. Joint mode obtains actor and critic
gradients from one combined forward loss, so neither side observes parameters
already updated by the other side.

For the H1 semantic-correspondence controls, enable target shuffling only with
a directional frozen-target mode:

```bash
ALIGN_MODE=a_to_c ALIGN_TARGET_SHUFFLE=true
ALIGN_MODE=c_to_a ALIGN_TARGET_SHUFFLE=true
```

Targets are cyclically deranged within each agent's alive environment-by-time
pool once per PPO update. The same permutation is reused by every minibatch and
epoch, and its RNG substream does not alter rollout/action sampling. These
historical controls are outside the current NPS-only H1 protocol documented in
`H1_EXPERIMENT.md`.

## Checkpoints and post-training evaluation

Checkpointing is disabled by default for individual baseline commands. Enable
it with:

```bash
SAVE_CHECKPOINTS=true \
CHECKPOINT_INTERVAL_TIMESTEPS=1000000 \
CHECKPOINT_DIR="$HOME/JaxMARL/checkpoints/MAPPO/smax" \
python baselines/MAPPO/mappo_rnn_smax.py ...
```

The controlled-heterogeneity launcher enables checkpointing by default. Each
run gets its own directory containing an `initial` checkpoint, nominal
interval checkpoints, and a separate `final` checkpoint. Every checkpoint
contains:

- `model.safetensors`: complete actor and critic parameter trees;
- `config.json`: the full training and architecture configuration;
- `metadata.json`: step, seed, map, actor-sharing, and alignment metadata.

Set `WANDB_UPLOAD_CHECKPOINTS=true` to additionally version every checkpoint as
a W&B model artifact. Local saving remains enabled independently of artifact
upload.

Evaluate either a run directory, its `final` directory, an intermediate step
directory, or a direct `model.safetensors` path:

```bash
python baselines/MAPPO/eval_mappo_rnn_smax.py \
  --checkpoint /path/to/run/final \
  --episodes 256 \
  --num-envs 64 \
  --seed 10000 \
  --policy deterministic
```

Use `--policy stochastic` to sample from the learned policy. Evaluation writes
per-episode results plus means, standard deviations, standard errors, and 95%
confidence intervals to JSON. Add `--wandb-mode online` to log the aggregate
evaluation metrics to a separate W&B evaluation project.

## Tested environment

- Ubuntu 22.04
- Python 3.11
- JAX/JAXLIB 0.10.2 with CUDA 13
- Distrax 0.1.9
- `tfp-nightly` 0.26.0.dev20260908
- NVIDIA driver 580 or newer

Install JAX before the repository dependencies:

```bash
python -m pip install "jax[cuda13]==0.10.2"
python -m pip install -i https://pypi.org/simple \
  "tfp-nightly==0.26.0.dev20260908"
python -m pip install "distrax==0.1.9"
python -m pip install -e ".[algs]"
```

## Reproduce MAPPO on SMAX 10m_vs_11m

One seed, using the PPO settings reported for SMAX in the JaxMARL paper:

```bash
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HYDRA_FULL_ERROR=1

time python baselines/MAPPO/mappo_rnn_smax.py \
  MAP_NAME=10m_vs_11m \
  SEED=0 \
  NUM_ENVS=64 \
  NUM_STEPS=128 \
  TOTAL_TIMESTEPS=10000000 \
  LR=0.004 \
  UPDATE_EPOCHS=2 \
  NUM_MINIBATCHES=2 \
  MAX_GRAD_NORM=0.5 \
  WANDB_MODE=offline \
  PROJECT=jaxmarl-smax-repro
```

The number of completed environment steps is
`floor(TOTAL_TIMESTEPS / (NUM_ENVS * NUM_STEPS)) * NUM_ENVS * NUM_STEPS`.
For this configuration it is 9,994,240 steps over 1,220 updates.
