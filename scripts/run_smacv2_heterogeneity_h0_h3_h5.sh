#!/usr/bin/env bash
set -uo pipefail

REPO_DIR="${REPO_DIR:-$HOME/JaxMARL}"
PROJECT="${PROJECT:-jaxmarl-smax-heterogeneity}"
ALIGNMENT_COEF="${ALIGNMENT_COEF:-0.1}"
RUN_DIR="${RUN_DIR:-$REPO_DIR/logs/heterogeneity-h0-h3-h5-$(date +%Y%m%d_%H%M%S)}"
MAX_RUNS_PER_GPU="${MAX_RUNS_PER_GPU:-5}"

cd "$REPO_DIR" || exit 1
mkdir -p "$RUN_DIR"

# This environment uses JAX's pip-provided CUDA libraries. Avoid selecting the
# user's older CUDA 12.2 libraries from shell startup files.
unset LD_LIBRARY_PATH

maps=(
    smacv2_10_units_hetero_h0
    smacv2_10_units_hetero_h3
    smacv2_10_units_hetero_h5
)
sharing_values=(true false)
align_modes=(none c_to_a a_to_c reciprocal joint)
seeds=(1 2 3 4)

for gpu in 0 1 2 3; do
    : > "$RUN_DIR/tasks-gpu${gpu}.txt"
done

task_index=0
for map_name in "${maps[@]}"; do
    heterogeneity="${map_name##*_}"
    for sharing in "${sharing_values[@]}"; do
        if [[ "$sharing" == true ]]; then
            actor_label="shared"
        else
            actor_label="independent"
        fi
        for align_mode in "${align_modes[@]}"; do
            for seed in "${seeds[@]}"; do
                gpu=$((task_index % 4))
                printf '%s|%s|%s|%s|%s|%s\n' \
                    "$map_name" "$heterogeneity" "$sharing" "$actor_label" \
                    "$align_mode" "$seed" >> "$RUN_DIR/tasks-gpu${gpu}.txt"
                task_index=$((task_index + 1))
            done
        done
    done
done

run_one() {
    local gpu="$1"
    local task="$2"
    local map_name heterogeneity sharing actor_label align_mode seed
    IFS='|' read -r map_name heterogeneity sharing actor_label align_mode seed <<< "$task"

    local lambda_label="lambda${ALIGNMENT_COEF}"
    local run_name="${heterogeneity}-${actor_label}-${align_mode}-${lambda_label}-seed${seed}"
    local run_group="${heterogeneity}-${actor_label}-${lambda_label}"
    local log_file="$RUN_DIR/${run_name}.log"

    printf '[%(%F %T)T] GPU %s START %s\n' -1 "$gpu" "$run_name"
    CUDA_VISIBLE_DEVICES="$gpu" \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    HYDRA_FULL_ERROR=1 \
    WANDB_NAME="$run_name" \
    WANDB_RUN_GROUP="$run_group" \
    WANDB_TAGS="controlled-heterogeneity,${heterogeneity},${actor_label},${align_mode},${lambda_label}" \
    python baselines/MAPPO/mappo_rnn_smax.py \
        MAP_NAME="$map_name" \
        SEED="$seed" \
        ACTOR_PARAMETER_SHARING="$sharing" \
        MATCHED_COMPARISON=true \
        ALIGN_MODE="$align_mode" \
        ALIGNMENT_COEF="$ALIGNMENT_COEF" \
        WANDB_MODE=online \
        PROJECT="$PROJECT" \
        > "$log_file" 2>&1
    local status=$?
    printf '[%(%F %T)T] GPU %s END   %s status=%s\n' \
        -1 "$gpu" "$run_name" "$status"
    return "$status"
}

run_gpu_queue() {
    local gpu="$1"
    local task_file="$RUN_DIR/tasks-gpu${gpu}.txt"
    local failures=0

    while IFS= read -r task; do
        while (( $(jobs -pr | wc -l) >= MAX_RUNS_PER_GPU )); do
            wait -n || failures=$((failures + 1))
        done
        run_one "$gpu" "$task" &
    done < "$task_file"

    while (( $(jobs -pr | wc -l) > 0 )); do
        wait -n || failures=$((failures + 1))
    done
    return "$failures"
}

echo "Repository: $REPO_DIR"
echo "Run directory: $RUN_DIR"
echo "W&B project: $PROJECT"
echo "Alignment coefficient: $ALIGNMENT_COEF"
echo "Launching $task_index runs ($MAX_RUNS_PER_GPU concurrent runs per GPU)"

manager_pids=()
for gpu in 0 1 2 3; do
    run_gpu_queue "$gpu" &
    manager_pids+=("$!")
done

manager_failures=0
for pid in "${manager_pids[@]}"; do
    wait "$pid" || manager_failures=$((manager_failures + 1))
done

echo "All runs finished; GPU queues with failures: $manager_failures"
exit "$manager_failures"
