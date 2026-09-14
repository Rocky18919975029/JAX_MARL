# H1 SMAX confirmatory experiment

This repository implements the execution protocol in
`H1_SMAX_EXPERIMENT_EXECUTION_MANUAL.md` on top of the existing matched MAPPO
code.  The original full matrix is two maps, PS/NPS actors, seven conditions,
and seeds 101--110 (280 runs).  Before that larger confirmatory study, the first
formal stage is a locked reduced matrix: both maps, NPS only, the `none`
baseline plus `c_to_a`, `a_to_c`, and `joint`, with seeds 1--4 (32 runs).

## What is implemented

- Semantic and within-agent shuffled `a_to_c` / `c_to_a` targets.  The
  shuffled target is a no-fixed-point cyclic permutation of the alive
  environment-by-time pool, generated once per PPO update from an RNG substream
  independent of rollout sampling.
- Per-update shuffle checksums and valid/fixed-point statistics in W&B.
- Separate RL, cross-loss, total-gradient, clipping, parameter-update,
  advantage, alive-fraction, training-return, and rollout-win diagnostics.
- Initial, nominal 500k-step, and final checkpoints carrying exact actual
  environment step, nominal step, Git commit, and protocol version.
- Frozen-config and software-manifest tooling; resumable four-GPU launchers;
  completion/failure manifests.
- Paired-seed deterministic checkpoint evaluation and unsmoothed performance
  AUC/bootstrap analysis.
- Complete stochastic diagnostic rollout shards, actor latent score vectors,
  cross-fitted reference-advantage/Fisher distortion, counterfactual SMAX
  action branching with common random numbers, and empirical Bellman-closure
  probes.

The diagnostic scripts intentionally run after training and never mutate model
or optimizer state.

## One-time protocol preparation

On the GPU server, the project checkout lives on a space-constrained disk.  All
experiment outputs must therefore live under the data disk.  The canonical H1
output root is `/home/data/zeshenghong/JaxMARL/h1_smax_runs`; the path inside
the checkout is only a convenience symlink to that directory.  Prepare it once:

```bash
cd ~/JaxMARL
git pull --ff-only
conda activate jaxmarl
unset LD_LIBRARY_PATH

export H1_DATA_ROOT="/home/data/zeshenghong/JaxMARL"
export H1_RUN_ROOT="$H1_DATA_ROOT/h1_smax_runs"
export H1_PROJECT_LINK="$HOME/JaxMARL/h1_smax_runs"

mkdir -p "$H1_DATA_ROOT"

if [ -L "$H1_PROJECT_LINK" ]; then
  test "$H1_PROJECT_LINK" -ef "$H1_RUN_ROOT" || {
    echo "Existing symlink points somewhere else: $H1_PROJECT_LINK"
    exit 1
  }
elif [ -e "$H1_PROJECT_LINK" ]; then
  test ! -e "$H1_RUN_ROOT" || {
    echo "Both project and data-disk output directories exist; merge manually."
    exit 1
  }
  mv "$H1_PROJECT_LINK" "$H1_RUN_ROOT"
  ln -s "$H1_RUN_ROOT" "$H1_PROJECT_LINK"
else
  mkdir -p "$H1_RUN_ROOT"
  ln -s "$H1_RUN_ROOT" "$H1_PROJECT_LINK"
fi

test "$H1_PROJECT_LINK" -ef "$H1_RUN_ROOT"
echo "Resolved output root: $(readlink -f "$H1_PROJECT_LINK")"
df -h "$H1_RUN_ROOT"
```

The `mv` branch preserves an existing protocol-test or smoke-test directory by
moving it to the data disk before creating the link.  It deliberately stops if
both locations already exist, because silently merging two experiment trees is
unsafe.  In every new shell, export `H1_RUN_ROOT` again.  The launcher resolves
symlinks and puts checkpoints, stdout logs, status files, W&B local
data/cache/artifact staging, and Hydra run metadata below this data-disk root.
The printed canonical path may begin with `/mnt/sda` when `/home/data` itself is
a mount alias; `test -ef` verifies directory identity without relying on the
spelling of those two equivalent paths.

Freeze the exact W&B pilot baseline config that generated the pilot curves.
The source may be a JSON export or W&B's local `files/config.yaml`:

```bash
python scripts/h1_protocol.py freeze \
  --source /absolute/path/to/pilot/config.yaml \
  --run-root "$H1_RUN_ROOT"
```

If the pilot artifact is unavailable, the documented second-priority fallback
is the repository YAML.  Its legacy default has `MATCHED_COMPARISON=false`, so
the required protocol change must be explicit and is recorded in
`deviations.md`:

```bash
python scripts/h1_protocol.py freeze \
  --source baselines/MAPPO/config/mappo_homogenous_rnn_smax.yaml \
  --override MATCHED_COMPARISON=true \
  --run-root "$H1_RUN_ROOT"
```

Capture the machine state and execute the acceptance tests:

```bash
python scripts/h1_protocol.py manifest --run-root "$H1_RUN_ROOT"

CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
python scripts/run_h1_protocol_tests.py --run-root "$H1_RUN_ROOT"
```

The tests create:

```text
protocol/tests/shuffled_target_audit.json
protocol/tests/alignment_gradient_audit.json
protocol/tests/latent_distortion_audit.json
protocol/tests/matched_initialization_audit.json
```

Do not start confirmatory training unless both report `pass` and the frozen
Git commit matches the checked-out commit.

## Reduced first formal stage (32 runs)

This profile deliberately contains no PS actors, reciprocal, or shuffled
controls.  The `none` condition supplies the seed-paired NPS baseline for all
three selected alignment modes.
Use a separate root and W&B project so it cannot be mixed with a later full
confirmatory matrix:

```bash
export H1_REDUCED_ROOT="$H1_RUN_ROOT/reduced_nps_4condition"

python scripts/h1_protocol.py freeze \
  --source baselines/MAPPO/config/mappo_homogenous_rnn_smax.yaml \
  --override MATCHED_COMPARISON=true \
  --run-root "$H1_REDUCED_ROOT"

python scripts/h1_protocol.py manifest --run-root "$H1_REDUCED_ROOT"

CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
python scripts/run_h1_protocol_tests.py --run-root "$H1_REDUCED_ROOT"
```

Preview the exact 32-run matrix before launching:

```bash
python scripts/run_h1_smax_confirmatory.py \
  --matrix-profile reduced-nps-4condition \
  --run-root "$H1_REDUCED_ROOT" \
  --maps 10m_vs_11m,smacv2_10_units \
  --actor-variants nps \
  --conditions none,c_to_a,a_to_c,joint \
  --seeds 1-4 \
  --gpus 0,1,2,3 \
  --max-runs-per-gpu 5 \
  --dry-run
```

Then launch it online in W&B.  Twenty runs start immediately (five per GPU),
and the remaining twelve start as capacity becomes available:

```bash
nohup python scripts/run_h1_smax_confirmatory.py \
  --matrix-profile reduced-nps-4condition \
  --run-root "$H1_REDUCED_ROOT" \
  --maps 10m_vs_11m,smacv2_10_units \
  --actor-variants nps \
  --conditions none,c_to_a,a_to_c,joint \
  --seeds 1-4 \
  --gpus 0,1,2,3 \
  --max-runs-per-gpu 5 \
  > "$H1_REDUCED_ROOT/training.stdout" 2>&1 &
```

The profile rejects any different seed, actor, condition, or map selection.
Runs are named `H1-reduced-{map}-nps-{condition}-lam0p1-seed{seed}` in the
`h1-smax-reduced-nps-4condition` W&B project.  Each
`H1-reduced-{map}-nps-{condition}-lam0p1` W&B group contains exactly the four
seeds for one task/condition pair, so grouping a chart by `Group` produces eight
mean curves rather than averaging different alignment modes together.

## Full confirmatory training (deferred)

Phase 1 runs seeds 101--102.  Five concurrent runs per 24 GB GPU matches the
capacity previously validated on the four RTX 4090 server; lower it if another
process uses memory.

```bash
nohup python scripts/run_h1_smax_confirmatory.py \
  --run-root "$H1_RUN_ROOT" \
  --seeds 101-102 \
  --gpus 0,1,2,3 \
  --max-runs-per-gpu 5 \
  > "$H1_RUN_ROOT/phase1.stdout" 2>&1 &
```

After checking all phase-1 status files, checkpoints, W&B groups, and a small
evaluation/diagnostic smoke test, launch the locked second phase:

```bash
nohup python scripts/run_h1_smax_confirmatory.py \
  --run-root "$H1_RUN_ROOT" \
  --seeds 103-110 \
  --gpus 0,1,2,3 \
  --max-runs-per-gpu 5 \
  > "$H1_RUN_ROOT/phase2.stdout" 2>&1 &
```

Monitor without attaching to child processes:

```bash
tail -f "$H1_RUN_ROOT/launcher.log"
watch -n 2 nvidia-smi
```

Runs are named exactly
`H1-{map}-{ps_or_nps}-{condition}-lam0p1-seed{seed}`.  The launcher is
resumable: a rerun skips status records already marked `completed` and never
silently substitutes a seed.

## Deterministic performance evaluation

After all selected training runs finish:

For the reduced stage, use `H1_REDUCED_ROOT` in place of `H1_RUN_ROOT` in all
evaluation, diagnostic, merge, analysis, and plotting commands below.  Its
four seeds remain the independent statistical units.

```bash
nohup python scripts/eval_h1_checkpoints.py \
  --run-root "$H1_RUN_ROOT" \
  --gpus 0,1,2,3 \
  --max-runs-per-gpu 1 \
  --episodes 256 \
  --num-envs 128 \
  > "$H1_RUN_ROOT/evaluation.stdout" 2>&1 &
```

This evaluates `initial`, 500k, 1M, 2M, 4M, 6M, 8M, and `final` using paired
evaluation seeds.  Aggregate the raw JSON files with:

```bash
python scripts/analyze_h1_performance.py --run-root "$H1_RUN_ROOT"
```

The resulting CSV files and unsmoothed PNG/PDF figures live under `analysis/`.
Bootstrap and shaded intervals use training seeds as the independent units.

## Mechanism diagnostics

The full preregistered mechanism suite is computationally and storage
intensive: it collects 512 complete stochastic episodes per checkpoint, runs
32 continuations for every candidate action at 256 anchor states, and fits 32
Bellman source heads.  Smoke-test one run/checkpoint first:

```bash
python scripts/run_h1_diagnostics.py \
  --run-root "$H1_REDUCED_ROOT" \
  --run-name-glob 'H1-reduced-10m_vs_11m-nps-none-lam0p1-seed1' \
  --checkpoint-name-glob final \
  --gpus 0 \
  --output-tree diagnostics_smoke \
  --stages collect,latent,decision,bellman \
  --episodes 16 \
  --batch-size 4 \
  --anchors 6 \
  --continuations 2 \
  --bellman-heads 2 \
  --allow-missing
```

The smoke output is isolated under `diagnostics_smoke/`; the formal run uses
the default `diagnostics_raw/` tree. Run the formal suite with:

```bash
nohup env JAX_ENABLE_X64=true \
python scripts/run_h1_diagnostics.py \
  --run-root "$H1_RUN_ROOT" \
  --gpus 0,1,2,3 \
  --max-runs-per-gpu 1 \
  --episodes 512 \
  --batch-size 64 \
  --anchors 256 \
  --continuations 32 \
  --bellman-heads 32 \
  > "$H1_RUN_ROOT/diagnostics.stdout" 2>&1 &
```

Stages can be scheduled separately with, for example,
`--stages collect,latent`.  Merge immutable per-checkpoint CSV files after all
workers finish:

```bash
python scripts/merge_h1_diagnostics.py --run-root "$H1_RUN_ROOT"
python scripts/analyze_h1_mechanisms.py --run-root "$H1_RUN_ROOT"
python scripts/plot_h1_mechanisms.py --run-root "$H1_RUN_ROOT"
```

Raw arrays remain under `diagnostics_raw/`; merged tables and the diagnostic
completion manifest are under `diagnostics_summary/`.

## Important interpretation details

- `returns` and `win_rate` logged during training are rollout metrics, not the
  formal deterministic evaluation curves.
- `smacv2_10_units` here is the SMACv2-style task implemented in SMAX, not the
  original PySC2 SMACv2 environment.
- `c_to_a` means critic target to actor recipient; `a_to_c` means actor target
  to critic recipient.
- The actual number of environment transitions is quantized by
  `NUM_ENVS * NUM_STEPS`.  Checkpoint directories use the preregistered nominal
  boundary, while `metadata.json` records the exact update-complete step.
- W&B smoothing is for viewing only.  All reported statistics are regenerated
  from local per-episode evaluation JSON and per-seed diagnostic files.
