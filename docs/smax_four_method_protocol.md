# Final SMAX NPS experiment stack

The supported conditions are `none`, `mse` (C→A LN-MSE), `cka` (C→A linear
CKA), and `arec` (actor-side policy-score recovery). All conditions use the
same independent-actor MAPPO architecture, matched actor/critic initialization,
global advantage normalization, stratified per-agent minibatches, and the
checked-out SMAX scenario definition. The auxiliary objectives are mutually
exclusive. C→A stops the critic target gradient; ARec uses a separate q
optimizer and stops both q/critic and Fisher gradients during the actor update.

Use only `scripts/smax_four_method.py` for new SMAX sweeps. The old SMAX-only
launchers, oracle/critic-recovery/probe experiments, diagnostics and reports
were removed; unrelated VMAS/HARL/MPE code and the upstream SMAX environment
were not changed. Historical outputs stay on disk, but this protocol does not
mix them into a new sweep.

## Reproducibility contract

- `run` requires a clean Git checkout and an output root outside the checkout.
  It records the commit, hashes of the relevant source/config files, complete
  generated commands, Python/package/conda environment, driver/GPU inventory,
  all grid values, assignment policy, and every seed in an immutable manifest.
- A seed range is `--seed-start` through `--seed-start + --seed-count - 1`.
  The same K seeds and effective environment-step budget apply to every grid
  cell. Requested timesteps are rounded **down** to a common multiple of all
  rollout sizes in the PPO grid, and both numbers are recorded.
- Repeating an identical `run` command verifies completed runs and skips them.
  A failed/interrupted/invalid run is never silently restarted or mixed with
  old metrics. Investigate it, then use a new run root for a fresh experiment.
- Per-run identity hashes the full task, method, seed, budget, PPO, and
  method-specific hyperparameters. W&B ID also includes the absolute run root,
  so a fresh sweep cannot accidentally reuse a previous W&B run. W&B `group`
  groups exactly K training seeds for one grid cell *within that run root*;
  `method` is also a tag. Checkpoint naming uses these frozen IDs even when
  W&B is disabled (which otherwise supplies a random dummy ID).
- This is a controlled statistical protocol, not a claim of bitwise-identical
  GPU arithmetic on different CUDA/JAX/driver stacks. Use the recorded
  environment lock and GPU type when rerunning.

## Example: four methods on one task, K=4

Run these commands on the server after pulling the committed revision and
activating the existing `jaxmarl` environment:

```bash
cd ~/JaxMARL
git pull --ff-only
conda activate jaxmarl
unset LD_LIBRARY_PATH
export SMAX4_ROOT=/home/data/zeshenghong/JaxMARL/h1_smax_runs/smax4_10m_vs_11m_v1
python scripts/smax_four_method.py run --run-root "$SMAX4_ROOT" --map-name 10m_vs_11m --methods none,mse,cka,arec --seed-count 4 --seed-start 1 --mse-coefs 0.03,0.1,0.3 --cka-coefs 0.03,0.1,0.3 --arec-coefs 0.00001,0.00003,0.0001 --arec-q-steps 4,8 --arec-q-lrs 0.001 --arec-fisher-ridges 0.001 --total-timesteps 10000000 --gpus 0,1,2,3 --max-runs-per-gpu 2 --project jaxmarl-smax-four-method --dry-run
nohup python scripts/smax_four_method.py run --run-root "$SMAX4_ROOT" --map-name 10m_vs_11m --methods none,mse,cka,arec --seed-count 4 --seed-start 1 --mse-coefs 0.03,0.1,0.3 --cka-coefs 0.03,0.1,0.3 --arec-coefs 0.00001,0.00003,0.0001 --arec-q-steps 4,8 --arec-q-lrs 0.001 --arec-fisher-ridges 0.001 --total-timesteps 10000000 --gpus 0,1,2,3 --max-runs-per-gpu 2 --project jaxmarl-smax-four-method > "$SMAX4_ROOT.launcher.stdout" 2>&1 &
python scripts/smax_four_method.py status --run-root "$SMAX4_ROOT"
```

The output root itself must not contain prior experiment artifacts. The stdout
file above is deliberately a sibling, so shell redirection succeeds before
the launcher creates the run root. To watch progress:

```bash
watch -n 5 "python scripts/smax_four_method.py status --run-root '$SMAX4_ROOT'"
```

Before a formal sweep, run this four-condition smoke test in a *different*
empty root. Its final checkpoint is verified even with W&B disabled:

```bash
export SMAX4_SMOKE=/home/data/zeshenghong/JaxMARL/h1_smax_runs/smax4_smoke_v1
python scripts/smax_four_method.py run --run-root "$SMAX4_SMOKE" --map-name 10m_vs_11m --methods none,mse,cka,arec --seed-count 1 --seed-start 1 --mse-coefs 0.1 --cka-coefs 0.3 --arec-coefs 0.00003 --arec-q-steps 4 --total-timesteps 16384 --gpus 0,1,2,3 --max-runs-per-gpu 1 --wandb-mode disabled --project jaxmarl-smax-four-method-smoke
python scripts/smax_four_method.py status --run-root "$SMAX4_SMOKE"
```

The `run` subcommand verifies final checkpoints and complete JSONL metrics
before marking any run completed. This smoke budget does not contain five
checkpoints, so do not run `evaluate` on the smoke root.

## Final-five evaluation and tables

After every training run completes, evaluate the last five distinct
checkpoints per seed using the same episode count and deterministic eval seed
protocol across conditions:

```bash
python scripts/smax_four_method.py evaluate --run-root "$SMAX4_ROOT" --episodes 256 --num-envs 64 --eval-seed-base 10000 --policy deterministic --gpus 0,1,2,3 --max-runs-per-gpu 2
python scripts/smax_four_method.py analyze --run-root "$SMAX4_ROOT" --bootstrap 5000 --bootstrap-seed 20260924
```

`summary/summary.csv` reports training-return/win-rate AUC (trapezoidal,
normalized by the common measured step span), final-five held-out checkpoint
performance, and deterministic seed-bootstrap 95% CIs. `per_seed.csv` and
`curves.csv` provide the seed-level numbers and seed-aggregated trajectories
with pointwise bootstrap CIs. If `analyze` is run before `evaluate`, the final
five columns are explicitly labeled `final_source=training`; do not use those
as held-out final performance. If evaluation has started but is incomplete,
analysis fails rather than silently falling back.

When selecting a winner on these same K seeds, label it *within-cohort
hyperparameter selection*. The bootstrap interval for the selected winner is
descriptive, not an independent confirmatory CI; a fresh seed cohort is needed
for an unbiased confirmation claim.
