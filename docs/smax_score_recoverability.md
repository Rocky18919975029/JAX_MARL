# SMAX nonlinear policy-score recoverability, v2

This is an **offline measurement** of frozen NPS MAPPO checkpoints, not an
auxiliary training loss. It evaluates `none`, C→A LN-MSE and C→A Linear CKA
separately on `10m_vs_11m`, `3s5z_vs_3s6z` and `smacv2_10_units`.
Tasks are never pooled.

## Matched checkpoints

For each task, the launcher finds a complete three-condition × four-seed
training-budget cohort. It intersects the **actually saved checkpoint steps**
of all 12 runs and selects distinct common steps nearest 25%, 50%, 75% and
100% of that task's budget. It refuses a nearest step more than 12.5% of the
budget from its requested fraction. `protocol.json` records requested and
actual steps; no checkpoints are interpolated and different conditions are
never compared at different steps. The final step uses each run's `final`
checkpoint. An isolated source must have the actual `none` experiment label
and no score-recovery, oracle, or shuffled-target auxiliary objective;
`ALIGN_MODE=none` alone is not sufficient.

Each checkpoint is frozen and independently rolled out for 1,024 stochastic
episodes, using the same reset-key seed within a task. Complete episodes are
assigned 70% fit, 15% validation and 15% test by one deterministic split.
There is no episode overlap. At each task/step, all conditions/seeds use the
same number of sampled valid transitions per agent, chosen from the smallest
available split count and capped at 16,384/4,096/4,096. The census is saved
in `sample_counts.json`. The same on-policy episodes also provide the
stochastic episode return plotted next to recoverability.

## Score and probe

The collector saves actor/critic GRU latents, the sampled action, the exact
SMAX invalid-action mask, and the exact gradient of masked actor log-prob
with respect to actor latent. Per agent, only fit scores estimate the Fisher
matrix. A fixed ridge-regularized inverse square root whitens fit,
validation and test scores alike. Critic latents use fit-only mean and
standard deviation; the action is one-hot encoded.

Each agent has an independent residual MLP:

`Dense(256) → ReLU → [Dense(256) → ReLU → Dense(256) + skip → ReLU] × 3 → Dense(128)`.

All cells use Adam at `1e-3`, batch size 512 and at most 5,000 steps.
Validation error is checked every 100 steps, with patience of 10 checks.
The best validation-selected weights are saved independently for each agent;
the test split is not used for model selection. Training, validation and
test normalized errors, score energies, Fisher spectrum, best steps and
estimator-failure flags are saved for audit. **Test errors are never clipped
to 1.** Values above 1 mean the finite trained estimator did worse than the
zero predictor; they are flagged for investigation, not interpreted as the
theoretical infimum.

## Run

Use a fresh root; v1 final-only results are incompatible:

```bash
cd ~/JaxMARL
git pull --ff-only
conda activate jaxmarl
unset LD_LIBRARY_PATH

export REC_ROOT=/home/data/zeshenghong/JaxMARL/h1_smax_runs/score_recoverability_resmlp_v2
python scripts/run_smax_score_recoverability.py \
  --matrix-root /home/data/zeshenghong/JaxMARL/h1_smax_runs \
  --run-root "$REC_ROOT" \
  --gpus 0,1,2,3 \
  --max-runs-per-gpu 1
```

Monitor with:

```bash
watch -n 10 python scripts/monitor_smax_score_recoverability.py --run-root "$REC_ROOT"
```

After all cells complete, `analysis/<task>/return-vs-score-recoverability.png`
shows return and normalized test error at the matched checkpoints. Tables
`agent_level.csv`, `seed_checkpoint_level.csv` and
`task_condition_checkpoint_summary.csv` are generated both globally and
separately by task. The 95% confidence intervals bootstrap training seeds,
not agents or episodes. The protocol uses a nonlinear empirical estimator
and does not claim to attain the unrestricted theoretical infimum.
