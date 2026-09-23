# Four-panel ARec return report from completed legacy sweeps

`scripts/report_smax_arec_best_returns.py` combines the completed four-seed
legacy sweep roots for `10m_vs_11m`, `3s5z_vs_3s6z`, `6s9z_vs_6s10z`, and
`smacv2_10_units`. It does not train policies. It can run held-out evaluation
for missing last-five checkpoints.

The 6s9z isolated curve is selected from its separate PPO baseline sweep by
maximum mean normalized training-return AUC across the same four seeds. The
10-units ARec curve comes from the latest completed four-seed 10-units sweep;
if that sweep omitted `none`, the latest completed four-seed sweep containing
`none` supplies the isolated curve. For all other tasks, ARec and `none` are
selected from their existing task-specific roots. ARec cells are selected by
mean paired return-AUC gain over the chosen baseline. Selection and plotting
never pool tasks. This same-seed selection makes the comparison exploratory.

Automatic discovery scans `--matrix-root` for complete, protocol-matching
manifests and prints every candidate plus its status-file timestamp. To inspect
the proposed sources before evaluation:

```bash
python scripts/report_smax_arec_best_returns.py \
  --matrix-root /home/data/zeshenghong/JaxMARL/h1_smax_runs \
  --discover-only
```

The full report needs the three unchanged task roots, `--matrix-root`, and a
fresh `--output-root`. Optional explicit `--root-6s9z-baseline`,
`--root-smacv2`, and `--root-smacv2-baseline` override discovery. Reusing old
evaluation JSON is allowed only after the checkpoint path and evaluation
protocol match exactly; mismatches are reevaluated if `--evaluate-missing` is
set. The output includes PNG/PDF/SVG, per-task and combined tables, seed-level
values, plot coordinates, caption, and a source/selection manifest.

`return_auc` is the training return integral divided by each task's own step
budget. `final_eval_return_last5_ckpt` is the average held-out return of the
last five distinct saved checkpoints, then averaged across four seeds. The
curve and its pointwise 95% bootstrap band use training returns and bootstrap
entire seeds. A tuned 6s9z isolated PPO baseline may have a different learning
rate or epoch count from the ARec runs; that panel is a tuned-method comparison,
not an auxiliary-loss-only ablation.
