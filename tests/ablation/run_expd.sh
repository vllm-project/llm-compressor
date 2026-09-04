#!/bin/bash
# Experiment D: two-pass observer with gs_prior scale ablation
# 4 configs × 6 models, sequential on GPUs 4-7
#
# Phase 1 (parallel, small models):
#   GPU 4:   Llama-8B        (~16GB)
#   GPU 5:   Qwen3-8B        (~16GB)
#   GPU 6:   Qwen3-14B       (~28GB)
#   GPU 7:   Qwen3-32B       (~64GB)
#
# Phase 2 (sequential, large models):
#   GPU 4,5: Llama-70B       (~140GB, offloads to CPU)
#   GPU 6,7: Qwen3-30B-A3B   (MoE, ~60GB)
#
# Requires HF_TOKEN to be set in the environment.

set -euo pipefail

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN must be set" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="${VENV:-$(dirname "$(which python)")}"
CONFIGS="twopass-gsp-1.0 twopass-gsp-1.25 twopass-gsp-1.5 twopass-gsp-1.75"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

echo "=== Experiment D: two-pass observer gs_prior ablation ==="
echo "Started at $(date)"
echo "Configs: $CONFIGS"
echo "GPUs: 4,5,6,7"
echo ""

# ── Phase 1: small/medium models in parallel (1 GPU each) ──────────

echo "--- Phase 1: small/medium models (parallel) ---"

CUDA_VISIBLE_DEVICES=4 $VENV/torchrun --nproc_per_node=1 --master_port=29500 \
    "$SCRIPT_DIR/compress_all.py" \
    --models meta-llama/Meta-Llama-3-8B-Instruct \
    --configs $CONFIGS \
    > "$LOG_DIR/expd-llama8b.log" 2>&1 &
PID_LLAMA8B=$!
echo "Launched Llama-8B       (GPU 4)     PID=$PID_LLAMA8B"

CUDA_VISIBLE_DEVICES=5 $VENV/torchrun --nproc_per_node=1 --master_port=29501 \
    "$SCRIPT_DIR/compress_all.py" \
    --models Qwen/Qwen3-8B \
    --configs $CONFIGS \
    > "$LOG_DIR/expd-qwen8b.log" 2>&1 &
PID_QWEN8B=$!
echo "Launched Qwen3-8B       (GPU 5)     PID=$PID_QWEN8B"

CUDA_VISIBLE_DEVICES=6 $VENV/torchrun --nproc_per_node=1 --master_port=29502 \
    "$SCRIPT_DIR/compress_all.py" \
    --models Qwen/Qwen3-14B \
    --configs $CONFIGS \
    > "$LOG_DIR/expd-qwen14b.log" 2>&1 &
PID_QWEN14B=$!
echo "Launched Qwen3-14B      (GPU 6)     PID=$PID_QWEN14B"

CUDA_VISIBLE_DEVICES=7 $VENV/torchrun --nproc_per_node=1 --master_port=29503 \
    "$SCRIPT_DIR/compress_all.py" \
    --models Qwen/Qwen3-32B \
    --configs $CONFIGS \
    > "$LOG_DIR/expd-qwen32b.log" 2>&1 &
PID_QWEN32B=$!
echo "Launched Qwen3-32B      (GPU 7)     PID=$PID_QWEN32B"

FAILED=0
for name_pid in "Llama-8B:$PID_LLAMA8B" "Qwen3-8B:$PID_QWEN8B" \
                "Qwen3-14B:$PID_QWEN14B" "Qwen3-32B:$PID_QWEN32B"; do
    name="${name_pid%%:*}"
    pid="${name_pid##*:}"
    if wait "$pid"; then
        echo "✓ $name complete"
    else
        echo "✗ $name FAILED (exit $?)"
        FAILED=$((FAILED + 1))
    fi
done

# ── Phase 2: large models in parallel (2 GPUs each) ────────────────

echo ""
echo "--- Phase 2: large models (parallel) ---"

CUDA_VISIBLE_DEVICES=4,5 $VENV/torchrun --nproc_per_node=1 --master_port=29504 \
    "$SCRIPT_DIR/compress_all.py" \
    --models meta-llama/Meta-Llama-3-70B-Instruct \
    --configs $CONFIGS \
    > "$LOG_DIR/expd-llama70b.log" 2>&1 &
PID_LLAMA70B=$!
echo "Launched Llama-70B      (GPU 4,5)   PID=$PID_LLAMA70B"

CUDA_VISIBLE_DEVICES=6,7 $VENV/python "$SCRIPT_DIR/compress_moe.py" \
    --model Qwen/Qwen3-30B-A3B \
    --configs $CONFIGS \
    > "$LOG_DIR/expd-moe30b.log" 2>&1 &
PID_MOE=$!
echo "Launched Qwen3-30B-A3B  (GPU 6,7)   PID=$PID_MOE"

for name_pid in "Llama-70B:$PID_LLAMA70B" "MoE-30B:$PID_MOE"; do
    name="${name_pid%%:*}"
    pid="${name_pid##*:}"
    if wait "$pid"; then
        echo "✓ $name complete"
    else
        echo "✗ $name FAILED (exit $?)"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "Finished at $(date)"
if [ $FAILED -eq 0 ]; then
    echo "All compressions succeeded."
else
    echo "$FAILED job(s) failed. Check logs in $LOG_DIR/"
fi
