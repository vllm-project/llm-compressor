#!/bin/bash
# Evaluate HIGGS mixed-precision checkpoints on gsm8k, wikitext, MMLU
# Uses vllm venv with lm_eval
#
# Usage:
#   bash eval_higgs.sh <model_path> [tp_size] [max_model_len]
#
# Examples:
#   bash eval_higgs.sh ~/hf_hub/Llama-3.1-8B-Instruct-HIGGS-NVFP4+FP8-W6.0avg 1 4096
#   bash eval_higgs.sh ~/hf_hub/Qwen3-30B-A3B-HIGGS-NVFP4+FP8-W6.0avg 4 4096

set -e

MODEL_PATH="${1:?Usage: eval_higgs.sh <model_path> [tp_size] [max_model_len]}"
TP="${2:-1}"
MAX_MODEL_LEN="${3:-4096}"

VLLM_PYTHON="$HOME/vllm/bin/python"
LM_EVAL="$HOME/vllm/bin/lm_eval"

echo "=========================================="
echo "Model:         $MODEL_PATH"
echo "TP:            $TP"
echo "Max model len: $MAX_MODEL_LEN"
echo "=========================================="

VLLM_ARGS="pretrained=${MODEL_PATH},dtype=auto,max_model_len=${MAX_MODEL_LEN},tensor_parallel_size=${TP},gpu_memory_utilization=0.85,trust_remote_code=True"

# Results will be saved per-model
RESULTS_DIR="${MODEL_PATH}/eval_results"
mkdir -p "$RESULTS_DIR"

echo ""
echo ">>> Evaluating wikitext..."
$LM_EVAL --model vllm \
    --model_args "${VLLM_ARGS},add_bos_token=True" \
    --tasks wikitext \
    --batch_size auto \
    --output_path "${RESULTS_DIR}/wikitext" \
    2>&1 | tee "${RESULTS_DIR}/wikitext.log"

echo ""
echo ">>> Evaluating gsm8k..."
$LM_EVAL --model vllm \
    --model_args "${VLLM_ARGS}" \
    --tasks gsm8k \
    --batch_size auto \
    --num_fewshot 5 \
    --output_path "${RESULTS_DIR}/gsm8k" \
    2>&1 | tee "${RESULTS_DIR}/gsm8k.log"

echo ""
echo ">>> Evaluating MMLU..."
$LM_EVAL --model vllm \
    --model_args "${VLLM_ARGS}" \
    --tasks mmlu \
    --batch_size auto \
    --num_fewshot 5 \
    --output_path "${RESULTS_DIR}/mmlu" \
    2>&1 | tee "${RESULTS_DIR}/mmlu.log"

echo ""
echo "=========================================="
echo "All evaluations complete for: $MODEL_PATH"
echo "Results saved to: $RESULTS_DIR"
echo "=========================================="
