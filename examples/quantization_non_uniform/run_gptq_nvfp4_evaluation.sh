#!/bin/bash
set -e

# GPTQ + nvfp4a16 Scale Rounding Evaluation Script
# This script:
# 1. Quantizes a model twice (with and without scale rounding) using llmcompressor
# 2. Evaluates both models using lm_eval with vllm backend
# 3. Compares the results

# Configuration
# Set default models if not specified
if [ -z "$1" ]; then
    MODELS=("TinyLlama/TinyLlama-1.1B-Chat-v1.0" "meta-llama/Llama-3.2-1B")
    MODEL_ID="${MODELS[0]}"  # For backward compatibility
else
    MODEL_ID="$1"
    MODELS=("$MODEL_ID")
fi

EVAL_TASKS="${2:-wikitext}"
EVAL_NUM_FEWSHOT="${3:-0}"
EVAL_LIMIT="${4:-}"  # Leave empty for full eval, or set to number like 100 for testing

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RHDEV_VENV="$HOME/rhdev"
QUANTIZE_SCRIPT="$SCRIPT_DIR/gptq_nvfp4_quantize_both.py"

# Derived paths
MODEL_NAME="$(basename $MODEL_ID)"
BASELINE_DIR="${MODEL_NAME}-GPTQ-NVFP4A16-NoRounding"
IMPROVED_DIR="${MODEL_NAME}-GPTQ-NVFP4A16-WithRounding"

echo "=============================================================================="
echo "GPTQ + nvfp4a16 Scale Rounding Evaluation"
echo "=============================================================================="
echo "Model: $MODEL_ID"
echo "Eval tasks: $EVAL_TASKS"
echo "Few-shot: $EVAL_NUM_FEWSHOT"
echo "Limit: ${EVAL_LIMIT:-full dataset}"
echo "=============================================================================="

# Step 1: Quantize both models using llmcompressor
echo ""
echo "=============================================================================="
echo "STEP 1: Quantizing models (with and without scale rounding)"
echo "=============================================================================="
echo "Using venv: $RHDEV_VENV"

cd "$SCRIPT_DIR"

# Activate rhdev venv and run quantization
source "$RHDEV_VENV/bin/activate"

echo "Running quantization script..."
python "$QUANTIZE_SCRIPT" "$MODEL_ID"

deactivate

echo ""
echo "Quantization complete!"
echo "  Baseline: $BASELINE_DIR"
echo "  Improved: $IMPROVED_DIR"

# Step 2: Evaluate baseline model (without scale rounding)
echo ""
echo "=============================================================================="
echo "STEP 2: Evaluating BASELINE model (round_scales=False)"
echo "=============================================================================="

BASELINE_OUTPUT="results_${MODEL_NAME}_baseline.json"
BASELINE_CMD="lm_eval --model vllm \
    --model_args pretrained=$BASELINE_DIR,tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.8,max_model_len=2048 \
    --tasks $EVAL_TASKS \
    --num_fewshot $EVAL_NUM_FEWSHOT \
    --batch_size auto \
    --output_path $BASELINE_OUTPUT"

if [ -n "$EVAL_LIMIT" ]; then
    BASELINE_CMD="$BASELINE_CMD --limit $EVAL_LIMIT"
fi

echo "Command: $BASELINE_CMD"
eval $BASELINE_CMD

# Step 3: Evaluate improved model (with scale rounding)
echo ""
echo "=============================================================================="
echo "STEP 3: Evaluating IMPROVED model (round_scales=True)"
echo "=============================================================================="

IMPROVED_OUTPUT="results_${MODEL_NAME}_improved.json"
IMPROVED_CMD="lm_eval --model vllm \
    --model_args pretrained=$IMPROVED_DIR,tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.8,max_model_len=2048 \
    --tasks $EVAL_TASKS \
    --num_fewshot $EVAL_NUM_FEWSHOT \
    --batch_size auto \
    --output_path $IMPROVED_OUTPUT"

if [ -n "$EVAL_LIMIT" ]; then
    IMPROVED_CMD="$IMPROVED_CMD --limit $EVAL_LIMIT"
fi

echo "Command: $IMPROVED_CMD"
eval $IMPROVED_CMD

# Step 4: Compare results
echo ""
echo "=============================================================================="
echo "RESULTS SUMMARY"
echo "=============================================================================="
echo "Baseline results: $BASELINE_OUTPUT"
echo "Improved results: $IMPROVED_OUTPUT"
echo ""
echo "Baseline model (round_scales=False):"
python -c "import json; data=json.load(open('$BASELINE_OUTPUT')); print(json.dumps(data.get('results', {}), indent=2))"
echo ""
echo "Improved model (round_scales=True):"
python -c "import json; data=json.load(open('$IMPROVED_OUTPUT')); print(json.dumps(data.get('results', {}), indent=2))"
echo ""
echo "=============================================================================="
echo "EVALUATION COMPLETE"
echo "=============================================================================="
echo ""
echo "To compare specific metrics, check the JSON files:"
echo "  - $BASELINE_OUTPUT"
echo "  - $IMPROVED_OUTPUT"
