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

## Recover a checkout change during the six-seed extension

If a Git pull occurred while the four extension launchers were running, later
workers may train successfully (`exit=0`) yet be marked failed solely because
their checkpoints record a newer commit than the immutable launcher manifest.
Do not delete those checkpoints or simply change their status by hand. The
reconciliation tool audits each such run: it checks that the newer commit is a
descendant, that the diff contains only explicitly allowed non-SMAX files,
that every frozen training source has the manifest's exact SHA-256 at both
commits, and that the other checkpoint/metrics validations pass. It refuses
any genuine training-code or artifact mismatch. Its default mode is read-only:

```bash
python scripts/reconcile_smax_first_four_panel_commits.py --collection-root "$COLLECTION_ROOT"
```

Review the audited run count and commits. Only if all candidates pass, apply
the reconciliation, then refresh the collection and run the report:

```bash
python scripts/reconcile_smax_first_four_panel_commits.py --collection-root "$COLLECTION_ROOT" --apply
python scripts/organize_smax_first_four_panel.py --matrix-root "$MATRIX_ROOT" --collection-root "$COLLECTION_ROOT" --phase refresh --require-complete --apply
nohup python scripts/report_smax_first_four_panel_10seed.py --collection-root "$COLLECTION_ROOT" --output-root "$REPORT_ROOT" --reuse-evaluation-root "$MATRIX_ROOT/actor_score_recovery_best_return_report_v1" --evaluate-missing --eval-episodes 256 --eval-num-envs 128 --eval-policy stochastic --gpus 0,1,2,3 --max-runs-per-gpu 2 > "$REPORT_ROOT/report.stdout" 2>&1 &
```

The original failed status JSONs remain at
`extension_6seed/reconciliation/original_failed_status/`. Reconciled statuses,
the collection index, seed-level table, and report manifest distinguish the
manifest commit from the actual checkpoint commit. Never pull the checkout
while new training runs are in progress; the script is an audit of this
specific completed-run incident, not a general permission to mix code.
