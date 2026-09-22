# ShadowHandOver: matched HAPPO, MAPPO, and MADPO

This extension keeps `third_party/HARL` unchanged. All three algorithms use the
official HARL `ShadowHandOver/happo/config.json` protocol with NPS actors; only
the actor update rule changes. Consequently, MAPPO and MADPO are matched-
protocol variants, not separately tuned official configurations.

## MADPO implementation notes

The implementation follows the official MADPO conditional Cauchy--Schwarz
divergence but fixes four integration hazards in the reference repository:

- the old policy is synchronized by freezing a copy immediately before update;
- reference and peer policy evaluations are stop-gradient;
- the peer is the previous agent in the sampled update order;
- only CCSD is subsampled before its quadratic Gram matrices are constructed.

PPO still sees the complete rollout minibatch. `div_max_samples` therefore
controls the divergence estimator only and does not change the shared PPO
optimization protocol.

## Server smoke test

Run from the JAX_MARL root in the existing Isaac Gym environment:

```bash
conda activate harl_dex
unset LD_LIBRARY_PATH

export DEX_SMOKE_ROOT="/home/data/zeshenghong/JaxMARL/harl_dexhands_shadowhandover/smoke"
mkdir -p "$DEX_SMOKE_ROOT"

python experiments/harl_dexhands/run_matrix.py \
  --run-root "$DEX_SMOKE_ROOT" \
  --algorithms happo,mappo,madpo \
  --seeds 1 \
  --gpus 0,1,2 \
  --max-runs-per-gpu 1 \
  --n-rollout-threads 16 \
  --num-env-steps 2400 \
  --div-max-samples 256 \
  --wandb-mode disabled
```

This is exactly two 75-step rollouts (`16 * 75 * 2`). Monitor all three runs:

```bash
watch -n 5 python experiments/harl_dexhands/monitor.py \
  --run-root "$DEX_SMOKE_ROOT" \
  --algorithms happo,mappo,madpo \
  --seeds 1 \
  --div-max-samples 256
```

## Four-seed matched matrix

Do not start this until the smoke test has completed without non-finite MADPO
metrics. The official matched budget is 50M steps with 256 Isaac Gym envs.

```bash
export DEX_FORMAL_ROOT="/home/data/zeshenghong/JaxMARL/harl_dexhands_shadowhandover/formal_4seed"
mkdir -p "$DEX_FORMAL_ROOT"

nohup python experiments/harl_dexhands/run_matrix.py \
  --run-root "$DEX_FORMAL_ROOT" \
  --algorithms happo,mappo,madpo \
  --seeds 1-4 \
  --gpus 0,1,2,3 \
  --max-runs-per-gpu 1 \
  --div-max-samples 1024 \
  --wandb-project harl-dexhands-shadowhandover \
  > "$DEX_FORMAL_ROOT/training.stdout" 2>&1 &
```

MADPO's published coefficient (`div_coef=1000`, `div_weight=0.05`,
`div_sigma=1`) is only a starting point because no official ShadowHandOver
tuning is available. It is persisted in run names, W&B config, status JSON, and
the frozen HARL config.
