# ShadowHandOver: matched HAPPO, MAPPO, MADPO, and actor-side score recovery

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

## Actor-side score recovery (ARec)

`--conditions none,arec` compares each original algorithm against the same
algorithm plus ARec. `none` keeps the old run names and update code unchanged.
The ARec condition is deliberately feed-forward/NPS/EP-only, matching the
official ShadowHandOver HAPPO config. The environment's agent count and
algorithm architecture are not changed.

On every rollout, each agent computes the score of the **joint continuous
action** under HARL's original diagonal Gaussian:
`s = ∂ log π(a|zA) / ∂zA`. Active rollout samples estimate one 256×256 Fisher
matrix per agent. Its inverse square root, with fixed ridge, is detached for
the entire PPO update. A per-agent recovery head is fitted to the frozen
whitened scores using detached critic features and the executed action vector.
Its predictions are then frozen as minibatch-aligned teacher targets. The
actor objective adds `λ_ARec ||M s_current − stop_gradient(q(zC,a))||²` to the
unmodified HAPPO/MAPPO/MADPO actor objective. This path differentiates through
the score (second order), but not through Fisher, q, or the critic. The critic
still receives only its original value loss. HAPPO's sequential factor and
MADPO's divergence/reference-policy updates remain intact.

The ARec coefficient, q fit steps/LR, and Fisher ridge are experimental
hyperparameters, **not** tuned ShadowHandOver values. The defaults
(`1e-4`, `4`, `1e-3`, `1e-3`) are smoke-test starting points. Each run logs
`actor/arec_q_loss_pre`, `actor/arec_q_loss_post`,
`actor/arec_q_to_zero_ratio`, `actor/arec_target_energy`, and
`actor/arec_weighted_loss`; q parameters are also saved with the actor and
critic checkpoints.

### Six-run ARec smoke test

First update the server checkout and activate `harl_dex`. Isaac Gym is imported
before PyTorch by the one-run trainer, and the launcher restores the Conda
runtime library path for subprocesses.

```bash
cd ~/JaxMARL
git pull --ff-only
conda activate harl_dex
unset LD_LIBRARY_PATH
export DEX_AREC_SMOKE_ROOT=/home/data/zeshenghong/JaxMARL/harl_dexhands_shadowhandover/arec_smoke_v1
python experiments/harl_dexhands/run_matrix.py --run-root "$DEX_AREC_SMOKE_ROOT" --algorithms happo,mappo,madpo --conditions none,arec --seeds 1 --gpus 0,1,2 --max-runs-per-gpu 1 --n-rollout-threads 16 --num-env-steps 2400 --div-max-samples 256 --arec-coef 0.0001 --arec-q-steps 4 --arec-q-lr 0.001 --arec-fisher-ridge 0.001 --wandb-mode disabled
```

Monitor the exact six names with the same ARec and MADPO arguments:

```bash
watch -n 5 "python experiments/harl_dexhands/monitor.py --run-root '$DEX_AREC_SMOKE_ROOT' --algorithms happo,mappo,madpo --conditions none,arec --seeds 1 --div-max-samples 256 --arec-coef 0.0001 --arec-q-steps 4 --arec-q-lr 0.001 --arec-fisher-ridge 0.001"
```

Do not launch a full-budget matrix until all six smoke runs finish with finite
metrics and the ARec rows show a nonzero weighted loss and a q/zero ratio below
one. ARec implementation has been unit-tested locally, but the Isaac Gym path
must be smoke-tested on the server.

### First four-seed λ grid: 48 runs

After the six-run smoke test, compare each of the three original algorithms
against ARec at `λ ∈ {3e-5, 1e-4, 3e-4}`. Thus each seed has three `none`
baselines and nine ARec runs. All other ARec settings are fixed. Omitting the
budget/thread overrides restores the official HAPPO protocol (50M environment
steps, 256 rollout threads); MAPPO and MADPO still inherit that matched
protocol, not their own tuned configs. This grid is expensive. Cap concurrency
at **two runs per GPU** (eight across four GPUs); do not reuse the four-runs-per-
GPU SMAX setting for Isaac Gym. The launcher works one seed cohort at a time:
it runs that seed's pending experiments, retries each failed experiment once
after the other experiments in that seed, then advances to the next seed.
A failed worker never stops dispatching the remaining matrix. On restart it
keeps already-running workers alive, reserves their GPU slots, and skips all
completed runs. Earlier failed logs and metrics are archived under
`failed_attempts/` so retries start with clean metrics.

```bash
export DEX_AREC_GRID_ROOT=/home/data/zeshenghong/JaxMARL/harl_dexhands_shadowhandover/arec_lambda_grid_4seed_v1
mkdir -p "$DEX_AREC_GRID_ROOT"
nohup python experiments/harl_dexhands/run_matrix.py --run-root "$DEX_AREC_GRID_ROOT" --algorithms happo,mappo,madpo --conditions none,arec --seeds 1-4 --arec-coefs 0.00003,0.0001,0.0003 --arec-q-steps 4 --arec-q-lr 0.001 --arec-fisher-ridge 0.001 --gpus 0,1,2,3 --max-runs-per-gpu 2 --wandb-project harl-dexhands-shadowhandover-arec-grid4seed > "$DEX_AREC_GRID_ROOT/launcher.stdout" 2>&1 &
echo "Launcher PID: $!"
```

The manifest lets the monitor show the whole grid without repeating its
selection arguments:

```bash
watch -n 5 "python experiments/harl_dexhands/monitor.py --run-root '$DEX_AREC_GRID_ROOT'"
```

If the watcher shows `--run-root ''`, export the absolute path again in that
terminal; shell variables do not cross terminal sessions. The output root is
on the data disk through `/home/data`. The λ grid is an initial candidate set,
not a claim that any coefficient is tuned or beneficial.

For online logging, check W&B in the active `harl_dex` Python before launching:

```bash
python -c 'import wandb; print(wandb.__file__); assert callable(wandb.init)'
```

The launcher performs the same import preflight before dispatching workers.
If W&B is missing in this Python 3.8 environment, install a compatible pinned
SDK (`python -m pip install 'wandb==0.21.1'`) and repeat the check. A run root
whose workers failed before training can be relaunched with the identical grid;
the manifest protects its settings and the launcher retries incomplete runs.
Do not delete failed status files by hand.

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
