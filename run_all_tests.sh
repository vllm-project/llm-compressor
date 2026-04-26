#!/bin/bash
# AWQ DDP Regression Test Suite
# Compares pre-DDP vs post-DDP AWQ quality across models, formats, and benchmarks.
#
# Usage:
#   ./run_all_tests.sh 2>&1 | tee regression_results.log
#   python extract_log_summary.py regression_results.log
#
# Models are saved to disk and NOT cleaned up for follow-up evaluation.

set -o pipefail

# ── Configuration ────────────────────────────────────────────────────────────

PRE_DDP_COMMIT="2ab02443"
POST_DDP_COMMIT="0bc916e5"

SCRIPTS=(
    "examples/awq/regression_tests/llama3_awq.py"
    "examples/awq/regression_tests/qwen3_vl_awq.py"
    "examples/awq/regression_tests/llama4_scout_awq.py"
    "examples/awq/regression_tests/qwen25_32b_awq.py"
    "examples/awq/regression_tests/mixtral_awq.py"
)

MODEL_SHORT_NAMES=(
    "Meta-Llama-3-8B-Instruct"
    "Qwen3-VL-8B-Instruct"
    "Llama-4-Scout-17B-16E-Instruct"
    "Qwen2.5-32B-Instruct"
    "Mixtral-8x7B-Instruct-v0.1"
)

# vLLM eval settings per model: max_model_len,tensor_parallel_size,num_gpus_quant
MODEL_VLLM_ARGS=(
    "2048,1,1"
    "4096,1,1"
    "4096,2,1"
    "4096,2,1"
    "2048,2,1"
)

SCHEMES=("W4A16_ASYM" "W8A8" "W8A16")

EVAL_TASKS=("gsm8k" "wikitext" "mmlu")
EVAL_FEWSHOT=("5" "0" "5")

CODE_STATES=("pre-ddp" "post-ddp")

EVAL_BASE_DIR="./eval_results"
MODEL_BASE_DIR="./regression_models"
RESULTS_CSV="regression_results.csv"

mkdir -p "$EVAL_BASE_DIR" "$MODEL_BASE_DIR"

# ── Helper: activate environment ─────────────────────────────────────────────

activate_env() {
    source /home/HDCharles/rhdev/bin/activate
}

# ── Helper: run vLLM evaluation with fallback chain ──────────────────────────

run_vllm_eval() {
    local save_dir=$1
    local task=$2
    local num_fewshot=$3
    local max_model_len=$4
    local tp_size=$5
    local eval_output_dir=$6

    mkdir -p "$eval_output_dir"

    echo "  EVAL: $task (fewshot=$num_fewshot, tp=$tp_size, max_len=$max_model_len)"

    # Build common eval flags
    local chat_args="--apply_chat_template"
    if [ "$num_fewshot" -gt 0 ]; then
        chat_args="$chat_args --fewshot_as_multiturn"
    fi

    # Try with tensor_parallel
    if [ "$tp_size" -gt 1 ]; then
        lm_eval \
            --model vllm \
            --model_args "pretrained=$save_dir,dtype=auto,max_model_len=$max_model_len,add_bos_token=True,tensor_parallel_size=$tp_size,gpu_memory_utilization=0.85" \
            --tasks "$task" \
            --num_fewshot "$num_fewshot" \
            --batch_size auto \
            $chat_args \
            --output_path "$eval_output_dir" 2>&1
        if [ $? -eq 0 ]; then return 0; fi

        echo "  TP=$tp_size failed, trying expert_parallel..."
        lm_eval \
            --model vllm \
            --model_args "pretrained=$save_dir,dtype=auto,max_model_len=$max_model_len,add_bos_token=True,enable_expert_parallel=True,gpu_memory_utilization=0.85" \
            --tasks "$task" \
            --num_fewshot "$num_fewshot" \
            --batch_size auto \
            $chat_args \
            --output_path "$eval_output_dir" 2>&1
        if [ $? -eq 0 ]; then return 0; fi
    fi

    echo "  Trying TP=1..."
    lm_eval \
        --model vllm \
        --model_args "pretrained=$save_dir,dtype=auto,max_model_len=$max_model_len,add_bos_token=True,gpu_memory_utilization=0.85" \
        --tasks "$task" \
        --num_fewshot "$num_fewshot" \
        --batch_size auto \
        $chat_args \
        --output_path "$eval_output_dir" 2>&1
    if [ $? -eq 0 ]; then return 0; fi

    echo "  Trying enforce_eager..."
    lm_eval \
        --model vllm \
        --model_args "pretrained=$save_dir,dtype=auto,max_model_len=$max_model_len,add_bos_token=True,enforce_eager=True,gpu_memory_utilization=0.85" \
        --tasks "$task" \
        --num_fewshot "$num_fewshot" \
        --batch_size auto \
        $chat_args \
        --output_path "$eval_output_dir" 2>&1
    if [ $? -eq 0 ]; then return 0; fi

    echo "  Trying hf backend as last resort..."
    lm_eval \
        --model hf \
        --model_args "pretrained=$save_dir,dtype=auto,add_bos_token=True" \
        --tasks "$task" \
        --num_fewshot "$num_fewshot" \
        --batch_size auto \
        $chat_args \
        --output_path "$eval_output_dir" 2>&1
    return $?
}

# ── Helper: extract metric from lm_eval JSON results ────────────────────────

extract_metric() {
    local eval_output_dir=$1
    local task=$2

    # Find the most recent results JSON in the eval output dir
    local results_json
    results_json=$(find "$eval_output_dir" -name "results_*.json" -type f 2>/dev/null | sort | tail -1)

    if [ -z "$results_json" ]; then
        echo "N/A"
        return
    fi

    python3 -c "
import json, sys
with open('$results_json') as f:
    data = json.load(f)
results = data.get('results', {})
task = '$task'

# Handle task name variations (e.g., wikitext vs wikitext2)
task_results = None
for key in results:
    if task in key:
        task_results = results[key]
        break

if task_results is None:
    print('N/A')
    sys.exit()

# Extract the primary metric for each task
if 'gsm8k' in task:
    val = task_results.get('exact_match,flexible-extract')
    if val is not None:
        print(f'{val*100:.2f}%')
    else:
        print('N/A')
elif 'wikitext' in task:
    val = task_results.get('word_perplexity,none')
    if val is not None:
        print(f'{val:.2f}')
    else:
        print('N/A')
elif 'mmlu' in task:
    val = task_results.get('acc,none')
    if val is not None:
        print(f'{val*100:.2f}%')
    else:
        print('N/A')
else:
    # Generic: grab first non-stderr, non-alias metric
    for k, v in task_results.items():
        if 'stderr' not in k and k != 'alias' and isinstance(v, (int, float)):
            print(f'{v:.4f}')
            sys.exit()
    print('N/A')
" 2>/dev/null || echo "N/A"
}

# ── Helper: switch code state ────────────────────────────────────────────────

switch_code_state() {
    local state=$1

    if [ "$state" == "pre-ddp" ]; then
        echo "Switching to pre-DDP code ($PRE_DDP_COMMIT)..."
        git checkout "$PRE_DDP_COMMIT" 2>&1
        pip install -e . --quiet 2>&1
    elif [ "$state" == "post-ddp" ]; then
        echo "Switching to post-DDP code ($POST_DDP_COMMIT)..."
        git checkout "$POST_DDP_COMMIT" 2>&1
        pip install -e . --quiet 2>&1
    fi
}

# ── Helper: print current results summary ────────────────────────────────────

print_summary() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗"
    echo "║  RESULTS SUMMARY (so far)                                                                         ║"
    echo "╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝"
    echo ""
    if [ -f "$RESULTS_CSV" ]; then
        # Print header + all rows as a formatted table
        column -t -s',' < "$RESULTS_CSV"
    else
        echo "(no results yet)"
    fi
    echo ""
    echo "════════════════════════════════════════════════════════════════════════════════════════════════════════"
    echo ""
}

# ── Initialize results CSV ───────────────────────────────────────────────────

echo "model,scheme,code_state,task,metric,status,save_dir" > "$RESULTS_CSV"

# ── Main loop ────────────────────────────────────────────────────────────────

TOTAL=0
PASSED=0
FAILED=0

activate_env

for model_idx in "${!SCRIPTS[@]}"; do
    script="${SCRIPTS[$model_idx]}"
    model_name="${MODEL_SHORT_NAMES[$model_idx]}"
    IFS=',' read -r max_model_len tp_size num_gpus_quant <<< "${MODEL_VLLM_ARGS[$model_idx]}"

    for scheme in "${SCHEMES[@]}"; do
        for code_state in "${CODE_STATES[@]}"; do

            save_dir="$MODEL_BASE_DIR/${model_name}-${scheme}-${code_state}"

            echo ""
            echo "╔══════════════════════════════════════════════════════════════════════════════════════════╗"
            echo "║  MODEL: $model_name"
            echo "║  SCHEME: $scheme"
            echo "║  CODE STATE: $code_state"
            echo "╚══════════════════════════════════════════════════════════════════════════════════════════╝"
            echo ""

            # ── Quantize (skip if model already exists) ────────────
            if [ -d "$save_dir" ] && [ -f "$save_dir/config.json" ]; then
                echo "Quantized model already exists at $save_dir, skipping quantization."
            else
                # Switch code state
                switch_code_state "$code_state"

                echo "============================================"
                echo "Running: $script --scheme $scheme"
                echo "============================================"

                if [ "$num_gpus_quant" -gt 1 ]; then
                    torchrun --nproc_per_node="$num_gpus_quant" "$script" \
                        --scheme "$scheme" --save-dir "$save_dir" 2>&1
                else
                    python "$script" --scheme "$scheme" --save-dir "$save_dir" 2>&1
                fi

                quant_status=$?
                if [ $quant_status -ne 0 ]; then
                    echo "QUANTIZATION FAILED for $model_name / $scheme / $code_state"
                    for task in "${EVAL_TASKS[@]}"; do
                        echo "$model_name,$scheme,$code_state,$task,N/A,QUANT_FAILED,$save_dir" >> "$RESULTS_CSV"
                    done
                    FAILED=$((FAILED + ${#EVAL_TASKS[@]}))
                    TOTAL=$((TOTAL + ${#EVAL_TASKS[@]}))
                    print_summary
                    continue
                fi
            fi

            # ── Clear GPU memory before eval ─────────────────────────
            python3 -c "import torch; torch.cuda.empty_cache(); [torch.cuda.reset_peak_memory_stats(i) for i in range(torch.cuda.device_count())]" 2>/dev/null

            # ── Evaluate ─────────────────────────────────────────────
            for eval_idx in "${!EVAL_TASKS[@]}"; do
                task="${EVAL_TASKS[$eval_idx]}"
                fewshot="${EVAL_FEWSHOT[$eval_idx]}"
                eval_dir="$EVAL_BASE_DIR/${model_name}-${scheme}-${code_state}/${task}"

                TOTAL=$((TOTAL + 1))

                # Skip eval if results already exist
                existing_result=$(find "$eval_dir" -name "results_*.json" -type f 2>/dev/null | sort | tail -1)
                if [ -n "$existing_result" ]; then
                    metric_val=$(extract_metric "$eval_dir" "$task")
                    echo "  EVAL: $task — skipping, previous result found: $metric_val"
                    echo "$model_name,$scheme,$code_state,$task,$metric_val,PASSED,$save_dir" >> "$RESULTS_CSV"
                    PASSED=$((PASSED + 1))
                    continue
                fi

                run_vllm_eval "$save_dir" "$task" "$fewshot" "$max_model_len" "$tp_size" "$eval_dir"
                eval_status=$?

                if [ $eval_status -eq 0 ]; then
                    metric_val=$(extract_metric "$eval_dir" "$task")
                    echo "$model_name,$scheme,$code_state,$task,$metric_val,PASSED,$save_dir" >> "$RESULTS_CSV"
                    PASSED=$((PASSED + 1))
                else
                    echo "$model_name,$scheme,$code_state,$task,N/A,FAILED,$save_dir" >> "$RESULTS_CSV"
                    FAILED=$((FAILED + 1))
                fi
            done

            # ── Clean up model to free disk space ────────────────────
            if [ -d "$save_dir" ]; then
                echo "Removing quantized model at $save_dir to free disk space."
                rm -rf "$save_dir"
            fi

            print_summary

        done  # code_state
    done  # scheme
done  # model

# ── Final Summary ────────────────────────────────────────────────────────────

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════════════════╗"
echo "║  FINAL SUMMARY: $PASSED passed, $FAILED failed out of $TOTAL total evaluations         ║"
echo "╚══════════════════════════════════════════════════════════════════════════════════════════╝"
echo ""
print_summary
echo "Results CSV: $RESULTS_CSV"
echo "Saved models: $MODEL_BASE_DIR/"
echo "Eval outputs: $EVAL_BASE_DIR/"
echo ""
echo "To extract detailed metrics from the log:"
echo "  python extract_log_summary.py regression_results.log"
