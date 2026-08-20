#!/bin/bash
# Experiment E: twopass-nofp8 + imatrix-nofp8-expanded
#
# Part 1: twopass-nofp8 (data-free, 1 config × 6 models)
#   GPU 0:   Llama-8B
#   GPU 1:   Qwen3-8B
#   GPU 2:   Qwen3-14B
#   GPU 3:   Qwen3-32B
#   GPU 4,5: Llama-70B
#   GPU 6,7: Qwen3-30B-A3B (MoE)
#
# Part 2: imatrix-nofp8-expanded (needs calibration data, 1 config × 6 models)
#   Same GPU layout as Part 1
#
# Requires HF_TOKEN to be set in the environment.

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

echo "=== Experiment E: twopass-nofp8 + imatrix ==="
echo "Started at $(date)"
echo ""

# ── Part 1: twopass-nofp8 (data-free) ──────────────────────────────
echo "--- Part 1: twopass-nofp8 (data-free) ---"

CUDA_VISIBLE_DEVICES=0 $VENV/torchrun --nproc_per_node=1 --master_port=29500 \
    "$SCRIPT_DIR/compress_all.py" \
    --models meta-llama/Meta-Llama-3-8B-Instruct \
    --configs twopass-nofp8 \
    --output-dir "$OUTPUT_DIR" \
    > "$LOG_DIR/expe-twopass-llama8b.log" 2>&1 &
PID_TP_L8B=$!
echo "Launched Llama-8B       (GPU 0)     PID=$PID_TP_L8B"

CUDA_VISIBLE_DEVICES=1 $VENV/torchrun --nproc_per_node=1 --master_port=29501 \
    "$SCRIPT_DIR/compress_all.py" \
    --models Qwen/Qwen3-8B \
    --configs twopass-nofp8 \
    --output-dir "$OUTPUT_DIR" \
    > "$LOG_DIR/expe-twopass-qwen8b.log" 2>&1 &
PID_TP_Q8B=$!
echo "Launched Qwen3-8B       (GPU 1)     PID=$PID_TP_Q8B"

CUDA_VISIBLE_DEVICES=2 $VENV/torchrun --nproc_per_node=1 --master_port=29502 \
    "$SCRIPT_DIR/compress_all.py" \
    --models Qwen/Qwen3-14B \
    --configs twopass-nofp8 \
    --output-dir "$OUTPUT_DIR" \
    > "$LOG_DIR/expe-twopass-qwen14b.log" 2>&1 &
PID_TP_Q14B=$!
echo "Launched Qwen3-14B      (GPU 2)     PID=$PID_TP_Q14B"

CUDA_VISIBLE_DEVICES=3 $VENV/torchrun --nproc_per_node=1 --master_port=29503 \
    "$SCRIPT_DIR/compress_all.py" \
    --models Qwen/Qwen3-32B \
    --configs twopass-nofp8 \
    --output-dir "$OUTPUT_DIR" \
    > "$LOG_DIR/expe-twopass-qwen32b.log" 2>&1 &
PID_TP_Q32B=$!
echo "Launched Qwen3-32B      (GPU 3)     PID=$PID_TP_Q32B"

CUDA_VISIBLE_DEVICES=4,5 $VENV/torchrun --nproc_per_node=1 --master_port=29504 \
    "$SCRIPT_DIR/compress_all.py" \
    --models meta-llama/Meta-Llama-3-70B-Instruct \
    --configs twopass-nofp8 \
    --output-dir "$OUTPUT_DIR" \
    > "$LOG_DIR/expe-twopass-llama70b.log" 2>&1 &
PID_TP_L70B=$!
echo "Launched Llama-70B      (GPU 4,5)   PID=$PID_TP_L70B"

CUDA_VISIBLE_DEVICES=6,7 $VENV/python "$SCRIPT_DIR/compress_moe.py" \
    --model Qwen/Qwen3-30B-A3B \
    --configs twopass-nofp8 \
    --output-dir "$OUTPUT_DIR" \
    > "$LOG_DIR/expe-twopass-moe30b.log" 2>&1 &
PID_TP_MOE=$!
echo "Launched MoE-30B        (GPU 6,7)   PID=$PID_TP_MOE"

FAILED=0
for name_pid in "Llama-8B:$PID_TP_L8B" "Qwen3-8B:$PID_TP_Q8B" \
                "Qwen3-14B:$PID_TP_Q14B" "Qwen3-32B:$PID_TP_Q32B" \
                "Llama-70B:$PID_TP_L70B" "MoE-30B:$PID_TP_MOE"; do
    name="${name_pid%%:*}"
    pid="${name_pid##*:}"
    if wait "$pid"; then
        echo "✓ twopass $name complete"
    else
        echo "✗ twopass $name FAILED (exit $?)"
        FAILED=$((FAILED + 1))
    fi
done

# ── Part 2: imatrix-nofp8-expanded (needs calibration data) ────────
echo ""
echo "--- Part 2: imatrix-nofp8-expanded (with calibration) ---"

CUDA_VISIBLE_DEVICES=0 $VENV/torchrun --nproc_per_node=1 --master_port=29500 \
    "$SCRIPT_DIR/compress_all.py" \
    --models meta-llama/Meta-Llama-3-8B-Instruct \
    --configs imatrix-nofp8-expanded \
    --dataset HuggingFaceH4/ultrachat_200k --num-samples 512 \
    --output-dir "$OUTPUT_DIR" \
    > "$LOG_DIR/expe-imatrix-llama8b.log" 2>&1 &
PID_IM_L8B=$!
echo "Launched Llama-8B       (GPU 0)     PID=$PID_IM_L8B"

CUDA_VISIBLE_DEVICES=1 $VENV/torchrun --nproc_per_node=1 --master_port=29501 \
    "$SCRIPT_DIR/compress_all.py" \
    --models Qwen/Qwen3-8B \
    --configs imatrix-nofp8-expanded \
    --dataset HuggingFaceH4/ultrachat_200k --num-samples 512 \
    --output-dir "$OUTPUT_DIR" \
    > "$LOG_DIR/expe-imatrix-qwen8b.log" 2>&1 &
PID_IM_Q8B=$!
echo "Launched Qwen3-8B       (GPU 1)     PID=$PID_IM_Q8B"

CUDA_VISIBLE_DEVICES=2 $VENV/torchrun --nproc_per_node=1 --master_port=29502 \
    "$SCRIPT_DIR/compress_all.py" \
    --models Qwen/Qwen3-14B \
    --configs imatrix-nofp8-expanded \
    --dataset HuggingFaceH4/ultrachat_200k --num-samples 512 \
    --output-dir "$OUTPUT_DIR" \
    > "$LOG_DIR/expe-imatrix-qwen14b.log" 2>&1 &
PID_IM_Q14B=$!
echo "Launched Qwen3-14B      (GPU 2)     PID=$PID_IM_Q14B"

CUDA_VISIBLE_DEVICES=3 $VENV/torchrun --nproc_per_node=1 --master_port=29503 \
    "$SCRIPT_DIR/compress_all.py" \
    --models Qwen/Qwen3-32B \
    --configs imatrix-nofp8-expanded \
    --dataset HuggingFaceH4/ultrachat_200k --num-samples 512 \
    --output-dir "$OUTPUT_DIR" \
    > "$LOG_DIR/expe-imatrix-qwen32b.log" 2>&1 &
PID_IM_Q32B=$!
echo "Launched Qwen3-32B      (GPU 3)     PID=$PID_IM_Q32B"

CUDA_VISIBLE_DEVICES=4,5 $VENV/torchrun --nproc_per_node=1 --master_port=29504 \
    "$SCRIPT_DIR/compress_all.py" \
    --models meta-llama/Meta-Llama-3-70B-Instruct \
    --configs imatrix-nofp8-expanded \
    --dataset HuggingFaceH4/ultrachat_200k --num-samples 512 \
    --output-dir "$OUTPUT_DIR" \
    > "$LOG_DIR/expe-imatrix-llama70b.log" 2>&1 &
PID_IM_L70B=$!
echo "Launched Llama-70B      (GPU 4,5)   PID=$PID_IM_L70B"

CUDA_VISIBLE_DEVICES=6,7 $VENV/python "$SCRIPT_DIR/compress_moe.py" \
    --model Qwen/Qwen3-30B-A3B \
    --configs imatrix-nofp8-expanded \
    --dataset HuggingFaceH4/ultrachat_200k --num-samples 512 \
    --output-dir "$OUTPUT_DIR" \
    > "$LOG_DIR/expe-imatrix-moe30b.log" 2>&1 &
PID_IM_MOE=$!
echo "Launched MoE-30B        (GPU 6,7)   PID=$PID_IM_MOE"

for name_pid in "Llama-8B:$PID_IM_L8B" "Qwen3-8B:$PID_IM_Q8B" \
                "Qwen3-14B:$PID_IM_Q14B" "Qwen3-32B:$PID_IM_Q32B" \
                "Llama-70B:$PID_IM_L70B" "MoE-30B:$PID_IM_MOE"; do
    name="${name_pid%%:*}"
    pid="${name_pid##*:}"
    if wait "$pid"; then
        echo "✓ imatrix $name complete"
    else
        echo "✗ imatrix $name FAILED (exit $?)"
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
