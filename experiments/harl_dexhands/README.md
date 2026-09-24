# ShadowHandOver: matched HAPPO, MAPPO, MADPO, and actor-side score recovery

This extension keeps `third_party/HARL` unchanged. HAPPO and MAPPO retain the
existing official HARL `ShadowHandOver/happo/config.json` protocol with NPS
actors. In a **new run root**, MADPO defaults to the separately documented
`paper2024` profile in
`configs/madpo_shadowhandover_paper2024.json`; the old 48-run root retains its
frozen `legacy` profile and can be resumed without changing any of its runs.
The runner infers `legacy` for pre-existing manifests and rejects a changed
profile in the same run root. No MAPPO training setting is altered.

MADPO's paper Appendix Tables 3–4 report ShadowHandOver `lambda=0.2` and
Gaussian-kernel `sigma=500`, plus ReLU, gamma 0.99, 128 rollout threads,
75-step episodes, 256×3 MLP, five PPO epochs, clip 0.2, and 5e-4 actor/critic
learning rates. These override the HAPPO training/model fields **for MADPO
only**. The generic MADPO repository config supplies the 40M-step budget and
the separate divergence multiplier `div_coef=1000`; it does not publish a
task-specific JSON. We use a bounded 4000-sample CCSD estimator corresponding
to the paper's reported batch size. This is a documented integration choice,
not a claim of bit-exact reproduction: the paper writes a `1/sigma` divergence
coefficient, whereas the repository exposes both a kernel `sigma` and a
separate `div_coef`. The checked-in profile records that distinction.

Inspect the resolved, frozen run protocol with
`python experiments/harl_dexhands/monitor.py --run-root "$RUN_ROOT"`; the
manifest records both the upstream HARL config hash and the paper-profile hash.

### Replace the interrupted legacy 48-run grid

`run_paper_madpo_matrix.py` creates a new, frozen 48-run study. It references
completed HAPPO/MAPPO runs in the original grid without retraining or moving
their files; it trains only missing HAPPO/MAPPO runs plus all 16 paper-profile
MADPO runs in the new root. It preserves the old HAPPO/MAPPO settings (50M
steps), while MADPO uses the paper profile (40M steps). The new manifest records
the original manifest hash and each reused run's source. The monitor resolves
those references, so the new root shows all 48 logical runs.

On launch it acquires locks for both roots. Stop the old launcher and any
in-flight workers before starting the replacement: this launcher never sends
signals to them. A completed status is skipped. A failed status is skipped in
the seed's primary wave and retried once after its other runs. A `running`
status whose PID has exited is marked interrupted and handled as failed; a
still-live PID causes an explicit error rather than a duplicate launch. The
scheduler spreads a seed's runs across all selected GPUs before assigning a
second run to any GPU, allows at most two per GPU, completes each seed's primary
jobs before advancing, and defers failed jobs to one retry at the end of that
seed. Re-running the launcher keeps completed results and starts only incomplete
runs.

```bash
cd ~/JaxMARL
git pull --ff-only
conda activate harl_dex
unset LD_LIBRARY_PATH
export DEX_LEGACY_ROOT=/home/data/zeshenghong/JaxMARL/harl_dexhands_shadowhandover/arec_lambda_grid_4seed_v1
export DEX_PAPER48_ROOT=/home/data/zeshenghong/JaxMARL/harl_dexhands_shadowhandover/arec_paper_madpo_48run_v1
mkdir -p "$DEX_PAPER48_ROOT"
nohup python experiments/harl_dexhands/run_paper_madpo_matrix.py \
  --legacy-run-root "$DEX_LEGACY_ROOT" \
  --run-root "$DEX_PAPER48_ROOT" \
  --gpus 0,1,2,3 --max-runs-per-gpu 2 \
  --wandb-project harl-dexhands-shadowhandover-arec-grid4seed \
  > "$DEX_PAPER48_ROOT/launcher.stdout" 2>&1 &
watch -n 5 "python experiments/harl_dexhands/monitor.py --run-root '$DEX_PAPER48_ROOT'"
```

Only run this once the GPUs are available for this grid; the launcher does not
stop or manage GPU workers. If W&B is not
functional in `harl_dex`, add `--wandb-mode disabled` before the redirect; the
launcher checks W&B before stopping any legacy process.

### Separate MADPO paper-profile comparison

Use a **new** data-disk run root. Smoke-test both the MADPO baseline and its
ARec variant at one full rollout with the actual 128-thread and 4000-sample
settings before scheduling full training. Do not run this while the existing
Isaac Gym grid occupies the same GPUs.

```bash
cd ~/JaxMARL
git pull --ff-only
conda activate harl_dex
unset LD_LIBRARY_PATH
export DEX_MADPO_PAPER_SMOKE=/home/data/zeshenghong/JaxMARL/harl_dexhands_shadowhandover/madpo_paper2024_smoke_v1
python experiments/harl_dexhands/run_matrix.py --run-root "$DEX_MADPO_PAPER_SMOKE" --algorithms madpo --conditions none,arec --seeds 1 --madpo-profile paper2024 --arec-coef 0.0001 --num-env-steps 9600 --gpus 0,1 --max-runs-per-gpu 1 --wandb-mode disabled
python experiments/harl_dexhands/monitor.py --run-root "$DEX_MADPO_PAPER_SMOKE"
```

For the full four-seed comparison, use another root and omit the budget/thread
overrides: MADPO uses 40M steps and 128 rollout threads, while a mixed matrix
would keep the old HAPPO/MAPPO settings. The ARec and `none` MADPO runs use
the identical MADPO profile within each seed.

```bash
export DEX_MADPO_PAPER_ROOT=/home/data/zeshenghong/JaxMARL/harl_dexhands_shadowhandover/madpo_paper2024_arec_4seed_v1
mkdir -p "$DEX_MADPO_PAPER_ROOT"
nohup python experiments/harl_dexhands/run_matrix.py --run-root "$DEX_MADPO_PAPER_ROOT" --algorithms madpo --conditions none,arec --seeds 1-4 --madpo-profile paper2024 --arec-coefs 0.00003,0.0001,0.0003 --gpus 0,1,2,3 --max-runs-per-gpu 2 --wandb-project harl-dexhands-shadowhandover-madpo-paper2024 > "$DEX_MADPO_PAPER_ROOT/launcher.stdout" 2>&1 &
watch -n 5 "python experiments/harl_dexhands/monitor.py --run-root '$DEX_MADPO_PAPER_ROOT'"
```

## MADPO implementation notes

The implementation follows the official MADPO conditional Cauchy--Schwarz
divergence but fixes four integration hazards in the reference repository:

- the old policy is synchronized by freezing a copy immediately before update;
- reference and peer policy evaluations are stop-gradient;
- the peer is the previous agent in the sampled update order;
- only CCSD is subsampled before its quadratic Gram matrices are constructed.

PPO still sees the complete rollout minibatch. `div_max_samples` therefore
controls the divergence estimator only and does not change PPO optimization.

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
python experiments/harl_dexhands/run_matrix.py --run-root "$DEX_AREC_SMOKE_ROOT" --algorithms happo,mappo,madpo --madpo-profile legacy --conditions none,arec --seeds 1 --gpus 0,1,2 --max-runs-per-gpu 1 --n-rollout-threads 16 --num-env-steps 2400 --div-max-samples 256 --arec-coef 0.0001 --arec-q-steps 4 --arec-q-lr 0.001 --arec-fisher-ridge 0.001 --wandb-mode disabled
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

The following earlier 48-run recipe is the **legacy matched-protocol study**;
its existing run root must not be reused for the paper-profile MADPO study.
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
nohup python experiments/harl_dexhands/run_matrix.py --run-root "$DEX_AREC_GRID_ROOT" --algorithms happo,mappo,madpo --madpo-profile legacy --conditions none,arec --seeds 1-4 --arec-coefs 0.00003,0.0001,0.0003 --arec-q-steps 4 --arec-q-lr 0.001 --arec-fisher-ridge 0.001 --gpus 0,1,2,3 --max-runs-per-gpu 2 --wandb-project harl-dexhands-shadowhandover-arec-grid4seed > "$DEX_AREC_GRID_ROOT/launcher.stdout" 2>&1 &
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
