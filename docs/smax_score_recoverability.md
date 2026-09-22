# SMAX score recoverability

This is a frozen-policy, final-checkpoint-only measurement for the NPS runs on
`10m_vs_11m`, `3s5z_vs_3s6z`, and `smacv2_10_units`.  It does not alter MAPPO
training or backpropagate into an actor or critic.

For every task, condition, seed, and agent slot, the collector saves the
128-dimensional actor and critic GRU outputs, the sampled action, the SMAX
available-action mask, and the exact gradient of the masked categorical log
probability with respect to the actor GRU output.  The measurement then:

1. makes one deterministic 75/25 split by complete episode;
2. estimates the empirical Fisher from fit episodes only;
3. uses the fit Fisher to whiten fit and test policy scores;
4. trains an independent fixed-capacity probe per agent from standardized
   critic latent plus one-hot action to the whitened score;
5. reports vector squared error divided by held-out whitened-score energy.

The frozen defaults are 512 evaluation episodes, 16,384 fit and 4,096 test
transitions per agent, `xi=1e-3`, and a one-hidden-layer 256-unit MLP trained
for 2,000 Adam steps.  All conditions within a task use the same environment
reset seed, episode split, exact sample counts, probe initialization, and
optimization schedule.  RL parameters and collected arrays are detached.

The launcher recursively discovers the largest complete final-checkpoint
`none` / `c_to_a_mse` / `c_to_a_cka` NPS cohort for each task:

```bash
python scripts/run_smax_score_recoverability.py \
  --matrix-root /path/containing/all/smax/runs \
  --run-root /path/for/score_recoverability \
  --gpus 0,1,2,3 \
  --max-runs-per-gpu 1
```

Monitor all 36 cells with:

```bash
watch -n 10 python scripts/monitor_smax_score_recoverability.py \
  --run-root /path/for/score_recoverability
```

The launcher automatically writes task-separated CSV tables and both combined
and per-task PNG/PDF figures under `RUN_ROOT/analysis`.  Tasks are never pooled.
The 95% confidence intervals bootstrap training seeds; the four-seed case uses
all `4^4` ordered ordinary-bootstrap resamples.
