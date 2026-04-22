#!/bin/bash
# GPTQ actorder Regression Test Suite
# Compares without-actorder vs with-actorder (actorder=weight) for GPTQ
# quantization across models, schemes, and benchmarks.
#
# Usage:
#   ./run_all_tests.sh 2>&1 | tee regression_results.log
#   python extract_log_summary.py regression_results.log
#
# Models are saved to disk and NOT cleaned up for follow-up evaluation.

set -o pipefail

# Avoid permission errors on shared HF cache files
export HF_DATASETS_CACHE="/tmp/hf_datasets_cache"
mkdir -p "$HF_DATASETS_CACHE"

# ── Configuration ────────────────────────────────────────────────────────────
# Each entry defines one (script, model, scheme, vllm_args) test configuration.
# The scheme is tied to the script — no cross-product.

SCRIPTS=(
    # "testing/llama3_fp8_block.py"
    # "testing/qwen25_32b_fp8_block.py"
    # "testing/qwen3_vl_fp8_block.py"
    # "testing/llama4_scout_fp8_block.py"
    # "testing/mixtral_fp8_block.py"
    "testing/llama3_w4a16_gptq.py"
    "testing/llama3_w8a16_gptq.py"
)

MODEL_SHORT_NAMES=(
    # "Meta-Llama-3-8B-Instruct"
    # "Qwen2.5-32B-Instruct"
    # "Qwen3-VL-8B-Instruct"
    # "Llama-4-Scout-17B-16E-Instruct"
    # "Mixtral-8x7B-Instruct-v0.1"
    "Meta-Llama-3-8B-Instruct"
    "Meta-Llama-3-8B-Instruct"
)

# Scheme label per entry (used for naming and CSV output)
MODEL_SCHEMES=(
    # "FP8_BLOCK"
    # "FP8_BLOCK"
    # "FP8_BLOCK"
    # "FP8_BLOCK"
    # "FP8_BLOCK"
    "W4A16"
    "W8A16"
)

# vLLM eval settings per entry: max_model_len,tensor_parallel_size,num_gpus_quant
MODEL_VLLM_ARGS=(
    # "2048,1,1"
    # "4096,2,1"
    # "4096,1,1"
    # "4096,2,1"
    # "2048,2,1"
    "2048,1,1"
    "2048,1,1"
)

# without-actorder:       no actorder flag (standard GPTQ)
# with-actorder:          actorder=weight
# with-group-actorder:    actorder=group
ACTORDER_STATES=("without-actorder" "with-actorder" "with-group-actorder")

# eval_name      lm_eval_task    fewshot  backend
# gsm8k          gsm8k           5        vllm
# gsm8k_platinum gsm8k_platinum  5        vllm
# wikitext       wikitext        0        vllm
# mmlu           mmlu            5        vllm
EVAL_NAMES=("gsm8k" "gsm8k_platinum" "wikitext" "mmlu")
EVAL_LM_TASKS=("gsm8k" "gsm8k_platinum" "wikitext" "mmlu")
EVAL_FEWSHOT=("5" "5" "0" "5")
EVAL_BACKENDS=("vllm" "vllm" "vllm" "vllm")

EVAL_BASE_DIR="./eval_results"
MODEL_BASE_DIR="./regression_models"
RESULTS_CSV="regression_results.csv"

mkdir -p "$EVAL_BASE_DIR" "$MODEL_BASE_DIR"

# ── Helper: activate environments ────────────────────────────────────────────

activate_quant_env() {
    source /home/HDCharles/rhdev/bin/activate
}

activate_eval_env() {
    source /home/HDCharles/vllm/bin/activate
}

# ── Helper: run vLLM evaluation with fallback chain ──────────────────────────

EVAL_BACKEND=""  # set by run_vllm_eval to indicate which backend succeeded

run_vllm_eval() {
    local save_dir=$1
    local task=$2
    local num_fewshot=$3
    local max_model_len=$4
    local tp_size=$5
    local eval_output_dir=$6

    mkdir -p "$eval_output_dir"
    EVAL_BACKEND="FAILED"

    activate_eval_env

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
        if [ $? -eq 0 ]; then EVAL_BACKEND="vllm_tp${tp_size}"; return 0; fi

        echo "  TP=$tp_size failed, trying expert_parallel..."
        lm_eval \
            --model vllm \
            --model_args "pretrained=$save_dir,dtype=auto,max_model_len=$max_model_len,add_bos_token=True,enable_expert_parallel=True,gpu_memory_utilization=0.85" \
            --tasks "$task" \
            --num_fewshot "$num_fewshot" \
            --batch_size auto \
            $chat_args \
            --output_path "$eval_output_dir" 2>&1
        if [ $? -eq 0 ]; then EVAL_BACKEND="vllm_expert_parallel"; return 0; fi
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
    if [ $? -eq 0 ]; then EVAL_BACKEND="vllm_tp1"; return 0; fi

    echo "  Trying enforce_eager..."
    lm_eval \
        --model vllm \
        --model_args "pretrained=$save_dir,dtype=auto,max_model_len=$max_model_len,add_bos_token=True,enforce_eager=True,gpu_memory_utilization=0.85" \
        --tasks "$task" \
        --num_fewshot "$num_fewshot" \
        --batch_size auto \
        $chat_args \
        --output_path "$eval_output_dir" 2>&1
    if [ $? -eq 0 ]; then EVAL_BACKEND="vllm_eager"; return 0; fi

    echo "  Trying hf backend as last resort..."
    lm_eval \
        --model hf \
        --model_args "pretrained=$save_dir,dtype=auto,add_bos_token=True" \
        --tasks "$task" \
        --num_fewshot "$num_fewshot" \
        --batch_size auto \
        $chat_args \
        --output_path "$eval_output_dir" 2>&1
    if [ $? -eq 0 ]; then EVAL_BACKEND="hf"; return 0; fi

    EVAL_BACKEND="FAILED"
    return 1
}

# ── Helper: run HF-only evaluation ─────────────────────────────────────────

run_hf_eval() {
    local save_dir=$1
    local task=$2
    local num_fewshot=$3
    local eval_output_dir=$4

    mkdir -p "$eval_output_dir"
    EVAL_BACKEND="FAILED"

    activate_eval_env

    echo "  EVAL: $task (fewshot=$num_fewshot, backend=hf)"

    # Build common eval flags
    local chat_args="--apply_chat_template"
    if [ "$num_fewshot" -gt 0 ]; then
        chat_args="$chat_args --fewshot_as_multiturn"
    fi

    lm_eval \
        --model hf \
        --model_args "pretrained=$save_dir,dtype=auto,add_bos_token=True" \
        --tasks "$task" \
        --num_fewshot "$num_fewshot" \
        --batch_size auto \
        $chat_args \
        --output_path "$eval_output_dir" 2>&1
    if [ $? -eq 0 ]; then EVAL_BACKEND="hf"; return 0; fi

    EVAL_BACKEND="FAILED"
    return 1
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
    val = task_results.get('exact_match,strict-match')
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

# ── Helper: print actorder comparison table ─────────────────────────────────

print_comparison() {
    if [ ! -f "$RESULTS_CSV" ]; then
        return
    fi

    python3 - "$RESULTS_CSV" <<'PYEOF'
import csv, sys

csv_path = sys.argv[1]

rows = []
with open(csv_path) as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

if not rows:
    sys.exit()

actorder_keys = ["without-actorder", "with-actorder", "with-group-actorder"]

# Build lookup: (model, scheme, task) -> {actorder_state: metric}
lookup = {}
for r in rows:
    key = (r["model"], r["scheme"], r["task"])
    lookup.setdefault(key, {})
    lookup[key][r["actorder"]] = r["metric"]

# Only print if we have at least one row with baseline + one other
entries = [(k, v) for k, v in lookup.items()
           if "without-actorder" in v and
           any(s in v for s in actorder_keys[1:])]
if not entries:
    sys.exit()

def parse_metric(s):
    s = s.strip()
    if s.endswith("%"):
        return float(s[:-1]), True
    try:
        return float(s), False
    except ValueError:
        return None, False

def calc_improvement(baseline_str, compare_str, task):
    b_val, _ = parse_metric(baseline_str)
    c_val, _ = parse_metric(compare_str)
    if b_val is None or c_val is None or b_val == 0:
        return "N/A"
    if "wikitext" in task:
        pct = (b_val - c_val) / b_val * 100
    else:
        pct = (c_val - b_val) / b_val * 100
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.2f}%"

print("")
print("╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
print("║  ACTORDER COMPARISON (vs without-actorder baseline)                                                                                 ║")
print("╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")
print("")

header = (f"{'model':<36} {'scheme':<12} {'task':<18} "
          f"{'no-actorder':>14} "
          f"{'weight':>14} {'wt vs base':>12} "
          f"{'group':>14} {'grp vs base':>12}")
print(header)
print("-" * len(header))

for (model, scheme, task), metrics in sorted(entries):
    wo = metrics.get("without-actorder", "")
    wi = metrics.get("with-actorder", "")
    wg = metrics.get("with-group-actorder", "")

    wi_imp = calc_improvement(wo, wi, task) if wo and wi else ""
    wg_imp = calc_improvement(wo, wg, task) if wo and wg else ""

    print(f"{model:<36} {scheme:<12} {task:<18} "
          f"{wo:>14} "
          f"{wi:>14} {wi_imp:>12} "
          f"{wg:>14} {wg_imp:>12}")

print("")
PYEOF
}

# ── Initialize results CSV (preserve previous results) ──────────────────────

if [ -f "$RESULTS_CSV" ]; then
    cp "$RESULTS_CSV" "${RESULTS_CSV}.bak"
fi
echo "model,scheme,actorder,task,metric,status,eval_backend,save_dir" > "$RESULTS_CSV"

# ── Main loop ────────────────────────────────────────────────────────────────

TOTAL=0
PASSED=0
FAILED=0

for model_idx in "${!SCRIPTS[@]}"; do
    script="${SCRIPTS[$model_idx]}"
    model_name="${MODEL_SHORT_NAMES[$model_idx]}"
    scheme="${MODEL_SCHEMES[$model_idx]}"
    IFS=',' read -r max_model_len tp_size num_gpus_quant <<< "${MODEL_VLLM_ARGS[$model_idx]}"

        for actorder_state in "${ACTORDER_STATES[@]}"; do

            save_dir="$MODEL_BASE_DIR/${model_name}-${scheme}-${actorder_state}"

            echo ""
            echo "╔══════════════════════════════════════════════════════════════════════════════════════════╗"
            echo "║  MODEL: $model_name"
            echo "║  SCHEME: $scheme"
            echo "║  ACTORDER: $actorder_state"
            echo "╚══════════════════════════════════════════════════════════════════════════════════════════╝"
            echo ""

            # ── Skip entirely if all evals already have results ────
            all_evals_cached=true
            for eval_idx in "${!EVAL_NAMES[@]}"; do
                eval_dir="$EVAL_BASE_DIR/${model_name}-${scheme}-${actorder_state}/${EVAL_NAMES[$eval_idx]}"
                if ! find "$eval_dir" -name "results_*.json" -type f 2>/dev/null | grep -q .; then
                    all_evals_cached=false
                    break
                fi
            done
            if [ "$all_evals_cached" = true ]; then
                echo "All evals already cached, skipping quantization and eval."
                for eval_idx in "${!EVAL_NAMES[@]}"; do
                    eval_name="${EVAL_NAMES[$eval_idx]}"
                    lm_task="${EVAL_LM_TASKS[$eval_idx]}"
                    eval_dir="$EVAL_BASE_DIR/${model_name}-${scheme}-${actorder_state}/${eval_name}"
                    metric_val=$(extract_metric "$eval_dir" "$lm_task")
                    echo "  $eval_name: $metric_val"
                    echo "$model_name,$scheme,$actorder_state,$eval_name,$metric_val,PASSED,cached,$save_dir" >> "$RESULTS_CSV"
                    PASSED=$((PASSED + 1))
                    TOTAL=$((TOTAL + 1))
                done
                print_summary
                print_comparison
                continue
            fi

            # ── Quantize (skip if model already exists) ────────────
            if [ -d "$save_dir" ] && [ -f "$save_dir/config.json" ]; then
                echo "Quantized model already exists at $save_dir, skipping quantization."
            else
                activate_quant_env

                echo "============================================"
                echo "Running: $script (actorder_state=$actorder_state)"
                echo "============================================"

                # Build actorder argument
                actorder_arg=""
                if [ "$actorder_state" == "with-actorder" ]; then
                    actorder_arg="--actorder weight"
                elif [ "$actorder_state" == "with-group-actorder" ]; then
                    actorder_arg="--actorder group"
                fi

                if [ "$num_gpus_quant" -gt 1 ]; then
                    torchrun --nproc_per_node="$num_gpus_quant" "$script" \
                        $actorder_arg --save-dir "$save_dir" 2>&1
                else
                    python "$script" $actorder_arg --save-dir "$save_dir" 2>&1
                fi

                quant_status=$?
                if [ $quant_status -ne 0 ]; then
                    echo "QUANTIZATION FAILED for $model_name / $scheme / $actorder_state"
                    for eval_name in "${EVAL_NAMES[@]}"; do
                        echo "$model_name,$scheme,$actorder_state,$eval_name,N/A,QUANT_FAILED,N/A,$save_dir" >> "$RESULTS_CSV"
                    done
                    FAILED=$((FAILED + ${#EVAL_NAMES[@]}))
                    TOTAL=$((TOTAL + ${#EVAL_NAMES[@]}))
                    print_summary
                    print_comparison
                    continue
                fi
            fi

            # ── Clear GPU memory before eval ─────────────────────────
            python3 -c "import torch; torch.cuda.empty_cache(); [torch.cuda.reset_peak_memory_stats(i) for i in range(torch.cuda.device_count())]" 2>/dev/null

            # ── Evaluate ─────────────────────────────────────────────
            for eval_idx in "${!EVAL_NAMES[@]}"; do
                eval_name="${EVAL_NAMES[$eval_idx]}"
                lm_task="${EVAL_LM_TASKS[$eval_idx]}"
                fewshot="${EVAL_FEWSHOT[$eval_idx]}"
                backend="${EVAL_BACKENDS[$eval_idx]}"
                eval_dir="$EVAL_BASE_DIR/${model_name}-${scheme}-${actorder_state}/${eval_name}"

                TOTAL=$((TOTAL + 1))

                # Skip eval if results already exist
                existing_result=$(find "$eval_dir" -name "results_*.json" -type f 2>/dev/null | sort | tail -1)
                if [ -n "$existing_result" ]; then
                    metric_val=$(extract_metric "$eval_dir" "$lm_task")
                    echo "  EVAL: $eval_name — skipping, previous result found: $metric_val"
                    echo "$model_name,$scheme,$actorder_state,$eval_name,$metric_val,PASSED,cached,$save_dir" >> "$RESULTS_CSV"
                    PASSED=$((PASSED + 1))
                    continue
                fi

                if [ "$backend" == "hf" ]; then
                    run_hf_eval "$save_dir" "$lm_task" "$fewshot" "$eval_dir"
                else
                    run_vllm_eval "$save_dir" "$lm_task" "$fewshot" "$max_model_len" "$tp_size" "$eval_dir"
                fi
                eval_status=$?

                if [ $eval_status -eq 0 ]; then
                    metric_val=$(extract_metric "$eval_dir" "$lm_task")
                    echo "$model_name,$scheme,$actorder_state,$eval_name,$metric_val,PASSED,$EVAL_BACKEND,$save_dir" >> "$RESULTS_CSV"
                    PASSED=$((PASSED + 1))
                else
                    echo "$model_name,$scheme,$actorder_state,$eval_name,N/A,FAILED,$EVAL_BACKEND,$save_dir" >> "$RESULTS_CSV"
                    FAILED=$((FAILED + 1))
                fi
            done

            # ── Clean up model to free disk space ────────────────────
            if [ -d "$save_dir" ]; then
                echo "Removing quantized model at $save_dir to free disk space."
                rm -rf "$save_dir"
            fi

            print_summary
            print_comparison

        done  # actorder_state
done  # model

# ── Final Summary ────────────────────────────────────────────────────────────

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════════════════╗"
echo "║  FINAL SUMMARY: $PASSED passed, $FAILED failed out of $TOTAL total evaluations         ║"
echo "╚══════════════════════════════════════════════════════════════════════════════════════════╝"
echo ""
print_summary
print_comparison
echo "Results CSV: $RESULTS_CSV"
echo "Saved models: $MODEL_BASE_DIR/"
echo "Eval outputs: $EVAL_BASE_DIR/"
echo ""
echo "To extract detailed metrics from the log:"
echo "  python extract_log_summary.py regression_results.log"
