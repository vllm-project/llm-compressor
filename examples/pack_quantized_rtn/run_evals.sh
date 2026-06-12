#!/bin/bash
# Run lm_eval on one or more pack-quantized checkpoints.
#
# Usage:
#   bash run_evals.sh <model_dir> [model_dir ...]
#
# Example:
#   bash run_evals.sh \
#       Meta-Llama-3-8B-Instruct-W7A16-RTN \
#       Meta-Llama-3-8B-Instruct-W5A8-RTN \
#       Meta-Llama-3-8B-Instruct-W3A4-RTN

source /home/HDCharles/vllm/bin/activate

MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
TASKS="${TASKS:-wikitext}"
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-eval_results}"

MODEL_BASE="${MODEL_BASE:-Meta-Llama-3-8B-Instruct}"

if [ $# -eq 0 ]; then
    # default: eval everything produced by run_all.sh
    set -- \
        "${MODEL_BASE}-W2A16-RTN" \
        "${MODEL_BASE}-W3A16-RTN" \
        "${MODEL_BASE}-W5A16-RTN" \
        "${MODEL_BASE}-W7A16-RTN" \
        "${MODEL_BASE}-W2A4-RTN"  \
        "${MODEL_BASE}-W3A4-RTN"  \
        "${MODEL_BASE}-W5A8-RTN"  \
        "${MODEL_BASE}-W7A8-RTN"
fi

mkdir -p "$OUTPUT_DIR"

for model_path in "$@"; do
    model_name="$(basename "$model_path")"
    out_file="$OUTPUT_DIR/${model_name}.json"

    if [ ! -d "$model_path" ]; then
        echo "=== Skipping $model_name (not found) ==="
        continue
    fi

    echo "=== Evaluating: $model_name ==="
    echo "    tasks=${TASKS}  fewshot=${NUM_FEWSHOT}  max_len=${MAX_MODEL_LEN}"
    echo "    output -> $out_file"

    lm_eval --model vllm \
        --model_args "pretrained=${model_path},dtype=auto,max_model_len=${MAX_MODEL_LEN},add_bos_token=True,gpu_memory_utilization=0.85" \
        --tasks "$TASKS" \
        --num_fewshot "$NUM_FEWSHOT" \
        --apply_chat_template \
        --batch_size auto \
        --output_path "$out_file"

    echo ""
done

echo "=== Results summary ==="
for model_path in "$@"; do
    model_name="$(basename "$model_path")"
    out_file="$OUTPUT_DIR/${model_name}.json"
    if [ -f "$out_file" ]; then
        echo -n "  $model_name: "
        python3 -c "
import json, sys
d = json.load(open('$out_file'))
results = d.get('results', {})
for task, metrics in results.items():
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f'{task}/{k}={v:.4f}', end='  ')
print()
" 2>/dev/null || echo "(parse failed, see $out_file)"
    fi
done
