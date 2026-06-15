#!/bin/bash
# Launch multi-GPU per group DDP training.
#
# Partitions physical GPUs into groups, one group per process/rank.
# Each rank sees its own set of GPUs via CUDA_VISIBLE_DEVICES.
#
# Usage:
#   GPUS_PER_GROUP=2 ./launch_multi_gpu.sh ddp_qwen3_multi_gpu_example.py --model ... --scheme W4A16
#   GPUS_PER_GROUP=2 ./launch_multi_gpu.sh ddp_qwen3_multi_gpu_example.py --model /storage/yiliu7/Qwen/Qwen3-30B-A3B-Instruct-2507  --scheme W4A16
#
# This spawns 2 ranks, each with 2 GPUs (4 GPUs total).
# The Python script no longer needs to override CUDA_VISIBLE_DEVICES.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPUS_PER_GROUP=${GPUS_PER_GROUP:-${GPUS_PER_RANK:-2}}
NPROC=${NPROC:-2}  # number of ranks
PYTHON=${PYTHON:-/home/yiliu7/workspace/venvs/ar/bin/python}
MASTER_PORT=${MASTER_PORT:-29600}
MASTER_ADDR=${MASTER_ADDR:-localhost}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}

SCRIPT="$1"
shift

echo "Launching $NPROC ranks, $GPUS_PER_GROUP GPUs each"
echo "Python: $PYTHON"
echo "Script: $SCRIPT"

VISIBLE_GPUS_ENV=${CUDA_VISIBLE_DEVICES:-}
if [[ -n "$VISIBLE_GPUS_ENV" ]]; then
    IFS=',' read -r -a VISIBLE_GPUS <<< "$VISIBLE_GPUS_ENV"
else
    VISIBLE_GPUS=()
fi

TOTAL_GPUS_NEEDED=$((NPROC * GPUS_PER_GROUP))
if [[ ${#VISIBLE_GPUS[@]} -gt 0 && ${#VISIBLE_GPUS[@]} -ne $TOTAL_GPUS_NEEDED ]]; then
    echo "Expected $TOTAL_GPUS_NEEDED GPUs in CUDA_VISIBLE_DEVICES, got ${#VISIBLE_GPUS[@]}: $VISIBLE_GPUS_ENV" >&2
    exit 1
fi

pids=()
for RANK in $(seq 0 $((NPROC - 1))); do
    if [[ ${#VISIBLE_GPUS[@]} -gt 0 ]]; then
        GPU_OFFSET=$((RANK * GPUS_PER_GROUP))
        GPU_LIST=$(IFS=,; echo "${VISIBLE_GPUS[*]:$GPU_OFFSET:$GPUS_PER_GROUP}")
    else
        GPU_START=$((NODE_RANK * NPROC * GPUS_PER_GROUP + RANK * GPUS_PER_GROUP))
        GPU_END=$((GPU_START + GPUS_PER_GROUP - 1))
        GPU_LIST=$(seq -s, $GPU_START $GPU_END)
    fi
    echo "  Rank $RANK -> GPUs $GPU_LIST"

    CUDA_VISIBLE_DEVICES="$GPU_LIST" \
    AR_DISABLE_DATASET_SUBPROCESS=1 \
    LOCAL_RANK=0 \
    RANK=$((NODE_RANK * NPROC + RANK)) \
    WORLD_SIZE=$((NNODES * NPROC)) \
    MASTER_ADDR="$MASTER_ADDR" \
    MASTER_PORT="$MASTER_PORT" \
    TORCHELASTIC_RUN_ID="multi_gpu_$(date +%s)_$$" \
    GPUS_PER_GROUP="$GPUS_PER_GROUP" \
    "$PYTHON" -u "$SCRIPT_DIR/$SCRIPT" "$@" &

    pids+=($!)
    # Small delay so workers don't race for port binding
    sleep 0.5
done

# Wait for all processes
status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done
exit $status
