# Critic-only SMAX score-recovery condition

This condition tests whether explicitly making the critic representation
recover the actor's policy-relevant score can improve MAPPO without any
actor–critic latent alignment. It is distinct from the **offline linear
recoverability probe** and from the earlier oracle latent-distortion loss.

The training condition requires matched NPS actors, `ALIGN_MODE=none`,
`ALIGNMENT_COEF=0`, and no oracle/shuffle objective. Actor PPO loss is unchanged.
The recovery head is initialized only for this condition; the actor and value
network retain the same seed-matched initializations as the isolated baseline.

After each normal rollout, the frozen pre-update actors' existing 128-D latent
vectors, sampled actions, and stored available-action masks produce exact
`d log pi(a|z_actor) / d z_actor` using the same two-layer masked policy head.
The complete rollout's active/alive scores form one empirical Fisher **per
agent**. Whitening uses the fixed absolute ridge
`SCORE_RECOVERY_FISHER_RIDGE` (default `1e-3`). The target is immediately
stop-gradient and is held fixed for all PPO epochs/minibatches in that update.
The next rollout refreshes it.

The critic TrainState contains the old value network parameters plus a new
`ScoreRecoveryHead_0` subtree. The head takes `[critic_latent,
one_hot(action)]`, applies `Dense(128) → ReLU → Dense(128)`, and predicts the
128-D target. The masked mean **squared vector norm** is computed within each
agent's active/alive samples, then averaged equally over agents and added to
the critic objective with `SCORE_RECOVERY_COEF`. The
recovery gradient reaches the head and critic encoder, but neither the actor
nor the critic value head through this auxiliary path.

Logged audit fields include `score_recovery_loss`, the weighted objective,
the Fisher eigenvalue range, target energy, valid-sample count,
`score_recovery_critic_grad_norm`,
`score_recovery_critic_to_rl_grad_ratio`, and
`score_recovery_actor_grad_norm_max`. The latter should be numerically zero.

## Server smoke test before any formal run

```bash
cd ~/JaxMARL
git pull --ff-only
conda activate jaxmarl
unset LD_LIBRARY_PATH

export REC_SMOKE_ROOT=/home/data/zeshenghong/JaxMARL/h1_smax_runs/score_recovery_training_smoke_v1
python scripts/run_smax_score_recovery_training.py \
  --run-root "$REC_SMOKE_ROOT" \
  --maps 10m_vs_11m \
  --seeds 1 \
  --score-recovery-coef 0.1 \
  --total-timesteps 16384 \
  --update-epochs 1 \
  --gpus 0 \
  --max-runs-per-gpu 1 \
  --wandb-mode disabled

python scripts/monitor_smax_score_recovery_training.py \
  --run-root "$REC_SMOKE_ROOT"
```

The smoke target is one 128-environment × 128-step PPO update. Verify that
the run completes, `score_recovery_loss` is finite and nonzero, and
`score_recovery_actor_grad_norm_max` is near zero. A completed final
checkpoint should appear under `checkpoints/`.

The final checkpoint should also load in the existing SMAX evaluator; the
critic's additional recovery-head parameters do not change the actor policy.

Do **not** reuse the smoke directory for another coefficient or protocol:
the launcher freezes an experiment manifest. Use a new directory for a
gradient-scale pilot or coefficient check. For instance, with a learning rate
of zero and coefficient 1, `score_recovery_critic_grad_norm /
critic_rl_grad_norm` gives an initial relative gradient scale without
changing parameters. This is a scale diagnostic, not evidence that any
particular coefficient improves return.

After choosing one fixed coefficient, run the three tasks separately or in
one matrix. The default budgets are 10M steps for `10m_vs_11m`, 20M for
`3s5z_vs_3s6z`, and 10M for `smacv2_10_units`. The launcher produces only
the new condition; compare its checkpoints with the existing isolated and
Linear CKA cohorts under their own task/seed matched analysis. No task-level
returns should be pooled across maps.
