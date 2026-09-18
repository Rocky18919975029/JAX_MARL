# BenchMARL/VMAS NPS alignment — phase 1

This protocol runs exactly 36 MAPPO experiments:

- tasks: `discovery_5`, `passage_5`, `football_5v5_heuristic`;
- actor parameterization: non-parameter-sharing only;
- conditions: isolated, C→A LN-MSE, C→A Linear CKA;
- training seeds: 1, 2, 3, 4.

The implementation uses the official BenchMARL VMAS fine-tuned MAPPO settings:
10M frames, 600 vectorized environments, 60k frames per rollout batch, 45
minibatch passes, minibatches of 4096, a 256×256 Tanh MLP, and learning rate
5e-5. The critic is centralized and parameter-shared. All three conditions use
the same architecture; isolated training only sets the auxiliary coefficient to
zero.

For Football, five blue policies learn against five VMAS heuristic red players.
`physically_different=false` and `randomise_formation_indices=false` avoid adding
role/type randomization to this first phase.

## Existing server environment

Use the already configured environment; do not create another conda environment:

```bash
cd ~/JaxMARL
conda activate benchmarl-vmas
unset LD_LIBRARY_PATH

python - <<'PY'
import importlib.metadata as metadata
for package in ("benchmarl", "torch", "torchrl", "tensordict", "vmas"):
    print(package, metadata.version(package))
PY
```

The integration targets BenchMARL 1.5.x, TorchRL 0.10–0.11, and VMAS >=1.3.4.

## 1. Task-specific gradient-scale calibration

The CKA coefficient is selected without returns. One independent pilot seed uses
the same initial rollout on all three tasks. For every task, eight minibatches
are drawn through the exact replay-buffer sampling path used by training. Their
RMS gradients determine one task-specific CKA coefficient that matches LN-MSE
at coefficient 0.1. No coefficient is shared across tasks.

```bash
export VMAS_ROOT="/home/data/zeshenghong/JaxMARL/benchmarl_vmas_nps_phase1"
export VMAS_CAL_ROOT="$VMAS_ROOT/cka_gradient_calibration"

mkdir -p "$VMAS_CAL_ROOT"

nohup python experiments/benchmarl_vmas/calibrate_cka.py \
  --output-root "$VMAS_CAL_ROOT" \
  --pilot-seed 9001 \
  --minibatches 8 \
  --gpus 0,1,2,3 \
  > "$VMAS_CAL_ROOT/calibration.stdout" 2>&1 &

tail --retry -F "$VMAS_CAL_ROOT/calibration.stdout"
```

## 2. Smoke test

This launches all three conditions on all three tasks for seed 1 and 120k frames.
It exercises both auxiliary losses and task construction before formal training.

```bash
export VMAS_CKA_CAL="$VMAS_CAL_ROOT/cka_gradient_calibration.json"
export VMAS_SMOKE_ROOT="$VMAS_ROOT/smoke"

python experiments/benchmarl_vmas/run_matrix.py \
  --run-root "$VMAS_SMOKE_ROOT" \
  --cka-calibration "$VMAS_CKA_CAL" \
  --seeds 1 \
  --gpus 0,1,2,3 \
  --max-runs-per-gpu 1 \
  --max-frames 120000 \
  --wandb-mode disabled
```

Verify `9 / 9` completed status files and no failed status:

```bash
python - "$VMAS_SMOKE_ROOT" <<'PY'
import json, sys
from collections import Counter
from pathlib import Path
root = Path(sys.argv[1])
states = [json.loads(p.read_text())["status"] for p in (root / "status").glob("*.json")]
print(Counter(states))
assert Counter(states) == {"completed": 9}
PY
```

## 3. Formal 36-run matrix

```bash
export VMAS_FORMAL_ROOT="$VMAS_ROOT/formal_4seed"
mkdir -p "$VMAS_FORMAL_ROOT"

nohup python experiments/benchmarl_vmas/run_matrix.py \
  --run-root "$VMAS_FORMAL_ROOT" \
  --cka-calibration "$VMAS_CKA_CAL" \
  --tasks discovery_5,passage_5,football_5v5_heuristic \
  --seeds 1-4 \
  --gpus 0,1,2,3 \
  --max-runs-per-gpu 4 \
  --wandb-project benchmarl-vmas-nps-alignment-v2 \
  > "$VMAS_FORMAL_ROOT/training.stdout" 2>&1 &

tail --retry -F "$VMAS_FORMAL_ROOT/launcher.log"
```

The launcher is restart-safe: completed status files are skipped. Increase
`--max-runs-per-gpu` only after checking GPU memory and simulator throughput.

Besides the usual return and loss curves, every rollout batch records a
first-minibatch actor-gradient audit in W&B:

- `train/agents/alignment_rl_only_gradient_norm`;
- `train/agents/alignment_aux_only_gradient_norm`;
- `train/agents/alignment_combined_gradient_norm`;
- `train/agents/alignment_aux_to_rl_gradient_ratio`.

These curves make gradient-scale drift after the initial calibration visible;
they are diagnostics only and do not modify the optimizer update.

## 4. Progress

```bash
watch -n 5 python experiments/benchmarl_vmas/monitor.py \
  --run-root "$VMAS_FORMAL_ROOT"
```

## 5. Task-separated reports

```bash
python experiments/benchmarl_vmas/analyze.py \
  --run-root "$VMAS_FORMAL_ROOT"
```

Results are deliberately written into three independent directories:

```text
analysis/discovery_5/
analysis/passage_5/
analysis/football_5v5_heuristic/
```

Each contains `checkpoint_returns.csv`, `curve_summary.csv`,
`endpoint_summary.csv`, and PNG/PDF learning curves. There is no pooled or
cross-task summary statistic.
