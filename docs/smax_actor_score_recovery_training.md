# SMAX actor-side policy-score recovery

This is a standalone matched-NPS MAPPO intervention. It does **not** apply a
latent alignment loss and does **not** send recovery gradients into the
critic. Compare it with seed-matched isolated MAPPO and C-to-A Linear CKA;
report `10m_vs_11m` and `3s5z_vs_3s6z` separately.

For each normal PPO rollout, the sampled actions, invalid-action masks and
frozen pre-update actor latents produce the exact representation-level score
`d log pi(a|z_actor) / d z_actor`. The full rollout estimates one Fisher matrix
per agent. Its ridge-regularized inverse square root `M_i` is stop-gradient
and remains fixed through all PPO epochs in that update. The score target
`u_old = stop_gradient(score_old @ M_i)` trains `q_i`.

The recovery models are **per-agent** `Dense(128) → ReLU → Dense(128)` heads on
`[stop_gradient(z_critic_old), one_hot(action)]`. Each agent has separate
parameters and optimizer state. The last Dense layer starts at zero, so the
initial teacher is the zero predictor rather than a random vector. Every
rollout, `q_i` gets `ACTOR_SCORE_RECOVERY_Q_STEPS` full-rollout gradient steps
(default 8); its parameters then stay frozen throughout the PPO update.

Actor PPO minibatches recompute **current** actor latents and scores under the
current actor parameters. Their auxiliary objective is the active/alive
masked mean squared vector norm of

`score_current @ stop_gradient(M_i) - stop_gradient(q_i(z_critic_old, action))`.

The analytic score implementation is differentiable, so the outer actor
gradient includes the required mixed second derivative. The only auxiliary
gradient recipient is the actor. The critic still receives its ordinary
value-loss gradient. A separate `q` TrainState prevents optimizer momentum
from inadvertently moving critic parameters during `q` training.

The teacher is fitted on this rollout, not a held-out evaluation split. Its
fit loss is an **optimization diagnostic**, not an unbiased estimate of
recoverability. The W&B/JSONL fields to audit are:

- `actor_score_recovery_q_fit_loss_post` and
  `actor_score_recovery_q_to_zero_baseline_ratio` (a ratio above 1 means the
  teacher is worse than predicting zero on its own fit rollout);
- `actor_score_recovery_loss` and its weighted counterpart;
- `actor_score_recovery_actor_to_rl_grad_ratio_mean`;
- `actor_score_recovery_critic_grad_norm`, which must be zero;
- `actor_score_recovery_target_energy_mean` and Fisher eigenvalues.

The checkpoint stores `actor`, `critic`, and
`actor_score_recovery_head` parameter trees. Existing actor-only evaluators
can ignore the third tree.

## Server smoke test

Run this only after pulling the implementation. Use a fresh output directory
and a diagnostic coefficient, **not** an assumed final experimental value:

```bash
cd ~/JaxMARL
git pull --ff-only
conda activate jaxmarl
unset LD_LIBRARY_PATH
export AREC_SMOKE_ROOT=/home/data/zeshenghong/JaxMARL/h1_smax_runs/actor_score_recovery_smoke_v1
python scripts/run_smax_actor_score_recovery_training.py \
  --run-root "$AREC_SMOKE_ROOT" \
  --maps 10m_vs_11m \
  --seeds 1 \
  --actor-score-recovery-coef 0.001 \
  --total-timesteps 16384 \
  --update-epochs 1 \
  --gpus 0 \
  --wandb-mode disabled
python scripts/monitor_smax_actor_score_recovery_training.py \
  --run-root "$AREC_SMOKE_ROOT"
```

Do not reuse a run directory with a different protocol; the launcher freezes
its manifest. Select a fixed coefficient using gradient-scale checks, then
run all seeds with the same coefficient. Neither one-update loss nor the
teacher's in-sample fit is evidence of return or held-out recoverability.
