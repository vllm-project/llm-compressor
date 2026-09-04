#!/bin/bash
# Experiment F: imatrix knob tuning (nvfp4_expanded_imatrix, all nofp8)
#
# 6 configs × 6 models = 36 runs, all need calibration data (ultrachat-200k)
# Runs one config at a time, all 6 models in parallel:
#   GPU 0:   Llama-8B
#   GPU 1:   Qwen3-8B
#   GPU 2:   Qwen3-14B
#   GPU 3:   Qwen3-32B
#   GPU 4,5: Llama-70B
#   GPU 6,7: Qwen3-30B-A3B (MoE)

set -euo pipefail

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN must be set" >&2
    exit 1
fi

SCRIPT_DIR="/home/Roderick-Wu/observer-regression"
VENV="/home/Roderick-Wu/compress/.venv/bin"
OUTPUT_DIR="/home/Roderick-Wu/compressed-models"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

DATASET="HuggingFaceH4/ultrachat_200k"
NUM_SAMPLES=512

CONFIGS=(
    imatrix-norm2.0
    imatrix-norm2.4
    imatrix-norm3.0
    imatrix-temp0.5
    imatrix-temp2.0
    imatrix-clip5
)

echo "=== Experiment F: imatrix knob tuning ==="
echo "Configs: ${CONFIGS[*]}"
echo "Started at $(date)"
echo ""

TOTAL_FAILED=0

for cfg in "${CONFIGS[@]}"; do
    echo "--- Config: $cfg ---"
    echo "Started at $(date)"

    CUDA_VISIBLE_DEVICES=0 $VENV/torchrun --nproc_per_node=1 --master_port=29500 \
        "$SCRIPT_DIR/compress_all.py" \
        --models meta-llama/Meta-Llama-3-8B-Instruct \
        --configs "$cfg" \
        --dataset "$DATASET" --num-samples $NUM_SAMPLES \
        --output-dir "$OUTPUT_DIR" \
        > "$LOG_DIR/expf-${cfg}-llama8b.log" 2>&1 &
    PID_L8B=$!

    CUDA_VISIBLE_DEVICES=1 $VENV/torchrun --nproc_per_node=1 --master_port=29501 \
        "$SCRIPT_DIR/compress_all.py" \
        --models Qwen/Qwen3-8B \
        --configs "$cfg" \
        --dataset "$DATASET" --num-samples $NUM_SAMPLES \
        --output-dir "$OUTPUT_DIR" \
        > "$LOG_DIR/expf-${cfg}-qwen8b.log" 2>&1 &
    PID_Q8B=$!

    CUDA_VISIBLE_DEVICES=2 $VENV/torchrun --nproc_per_node=1 --master_port=29502 \
        "$SCRIPT_DIR/compress_all.py" \
        --models Qwen/Qwen3-14B \
        --configs "$cfg" \
        --dataset "$DATASET" --num-samples $NUM_SAMPLES \
        --output-dir "$OUTPUT_DIR" \
        > "$LOG_DIR/expf-${cfg}-qwen14b.log" 2>&1 &
    PID_Q14B=$!

    CUDA_VISIBLE_DEVICES=3 $VENV/torchrun --nproc_per_node=1 --master_port=29503 \
        "$SCRIPT_DIR/compress_all.py" \
        --models Qwen/Qwen3-32B \
        --configs "$cfg" \
        --dataset "$DATASET" --num-samples $NUM_SAMPLES \
        --output-dir "$OUTPUT_DIR" \
        > "$LOG_DIR/expf-${cfg}-qwen32b.log" 2>&1 &
    PID_Q32B=$!

    CUDA_VISIBLE_DEVICES=4,5 $VENV/torchrun --nproc_per_node=1 --master_port=29504 \
        "$SCRIPT_DIR/compress_all.py" \
        --models meta-llama/Meta-Llama-3-70B-Instruct \
        --configs "$cfg" \
        --dataset "$DATASET" --num-samples $NUM_SAMPLES \
        --output-dir "$OUTPUT_DIR" \
        > "$LOG_DIR/expf-${cfg}-llama70b.log" 2>&1 &
    PID_L70B=$!

    CUDA_VISIBLE_DEVICES=6,7 $VENV/python "$SCRIPT_DIR/compress_moe.py" \
        --model Qwen/Qwen3-30B-A3B \
        --configs "$cfg" \
        --dataset "$DATASET" --num-samples $NUM_SAMPLES \
        --output-dir "$OUTPUT_DIR" \
        > "$LOG_DIR/expf-${cfg}-moe30b.log" 2>&1 &
    PID_MOE=$!

    echo "  Launched 6 models on GPUs 0-7"

    FAILED=0
    for name_pid in "Llama-8B:$PID_L8B" "Qwen3-8B:$PID_Q8B" \
                    "Qwen3-14B:$PID_Q14B" "Qwen3-32B:$PID_Q32B" \
                    "Llama-70B:$PID_L70B" "MoE-30B:$PID_MOE"; do
        name="${name_pid%%:*}"
        pid="${name_pid##*:}"
        if wait "$pid"; then
            echo "  ✓ $cfg $name complete"
        else
            echo "  ✗ $cfg $name FAILED (exit $?)"
            FAILED=$((FAILED + 1))
        fi
    done
    TOTAL_FAILED=$((TOTAL_FAILED + FAILED))
    echo ""
done

echo "Finished at $(date)"
if [ $TOTAL_FAILED -eq 0 ]; then
    echo "All compressions succeeded."
else
    echo "$TOTAL_FAILED job(s) failed. Check logs in $LOG_DIR/"
fi
