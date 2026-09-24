# First four-panel settings: ten-seed return report

`scripts/report_smax_first_four_panel_10seed.py` reports the frozen `none` and
`arec` cells from the *first* four-panel figure. It does not select new
hyperparameters using the ten-seed outcomes. The collection must contain the
original selected seeds 1–4 and the six paired extension seeds 5–10 for all
four tasks (80 completed runs). Missing or mismatched runs, symlinks,
checkpoints, effective settings, or evaluation protocols cause an error.

The four task-specific ARec settings are the versioned YAML files in
`configs/smax_first_four_panel/`: λ=3e-6/q=4 for `10m_vs_11m`, λ=3e-4/q=4
for `3s5z_vs_3s6z`, λ=1e-4/q=8 for `6s9z_vs_6s10z`, and λ=3e-5/q=4 for
`smacv2_10_units`. Each is compared with its original *paired* `none` setting,
not a baseline reselected from a later sweep.

The curve is unsmoothed mean training episode return over ten seeds. Its band
is the pointwise 95% percentile bootstrap interval (resampling whole training
seeds, 20,000 draws by default). Return AUC is the trapezoidal integral over
the nominal task budget divided by that budget; the first and final observed
returns extend to the exact boundaries, matching the first four-panel report.
Final performance is the per-seed mean held-out stochastic-policy return of
the last five *distinct saved checkpoints*, followed by a ten-seed mean and
bootstrap CI. Paired differences use the same seed draws for `none` and ARec.
The evaluation default is 256 episodes per checkpoint, 128 environments, and
the same deterministic evaluation-seed rule as the original report. Existing
four-seed evaluations are reused only when checkpoint and protocol identities
match. The new seeds require up to 240 additional checkpoint evaluations.

Run this from the server checkout in the `jaxmarl` environment. The output
root is a child of the data-disk collection, not the system disk:

```bash
cd ~/JaxMARL
git pull --ff-only
conda activate jaxmarl
unset LD_LIBRARY_PATH
export MATRIX_ROOT=/home/data/zeshenghong/JaxMARL/h1_smax_runs
export COLLECTION_ROOT="$MATRIX_ROOT/first_four_panel_selected_10seed_v1"
export REPORT_ROOT="$COLLECTION_ROOT/report_10seed_v1"
python scripts/organize_smax_first_four_panel.py --matrix-root "$MATRIX_ROOT" --collection-root "$COLLECTION_ROOT" --phase refresh --require-complete --apply
mkdir -p "$REPORT_ROOT"
nohup python scripts/report_smax_first_four_panel_10seed.py --collection-root "$COLLECTION_ROOT" --output-root "$REPORT_ROOT" --reuse-evaluation-root "$MATRIX_ROOT/actor_score_recovery_best_return_report_v1" --evaluate-missing --eval-episodes 256 --eval-num-envs 128 --eval-policy stochastic --gpus 0,1,2,3 --max-runs-per-gpu 2 > "$REPORT_ROOT/report.stdout" 2>&1 &
```

The outputs are `smax-arec-selected-return-curves-10seed.{png,pdf,svg}`,
`summary_table.md`, `summary_all_tasks.csv`, per-task summary/seed/curve CSVs,
`report_manifest.json`, and `figure_caption.txt`. Follow progress with
`tail -n 30 "$REPORT_ROOT/report.stdout"` and count completed evaluation JSONs
in `$REPORT_ROOT/evaluation`. A successful run prints the final figure and
table paths at the end of the stdout log. Re-running the same command validates
existing evaluations and fills only missing ones.

Interpretation caveat: seeds 1–4 used the historical training commit that
selected these hyperparameters; seeds 5–10 use the later cleaned code with the
same frozen *effective* configurations. This is neither an independently
selected ten-seed estimate nor a bitwise homogeneous-code replay. The report
records each source root, origin, and training commit so that distinction is
visible in the seed-level table and manifest.
