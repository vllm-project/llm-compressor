#!/bin/bash
set -e

# GPTQ + nvfp4a16 Scale Rounding Multi-Model Evaluation Script
# This script evaluates GPTQ scale rounding across multiple models

# Models to evaluate
MODELS=(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    "meta-llama/Llama-3.2-1B"
)

# Evaluation configuration
EVAL_TASKS="${1:-wikitext}"
EVAL_NUM_FEWSHOT="${2:-0}"
EVAL_LIMIT="${3:-}"  # Leave empty for full eval, or set to number like 100 for testing

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RHDEV_VENV="$HOME/rhdev"
QUANTIZE_SCRIPT="$SCRIPT_DIR/gptq_nvfp4_quantize_both.py"

echo "=============================================================================="
echo "GPTQ + nvfp4a16 Scale Rounding Multi-Model Evaluation"
echo "=============================================================================="
echo "Models: ${MODELS[@]}"
echo "Eval tasks: $EVAL_TASKS"
echo "Few-shot: $EVAL_NUM_FEWSHOT"
echo "Limit: ${EVAL_LIMIT:-full dataset}"
echo "=============================================================================="

# Create results directory
RESULTS_DIR="$SCRIPT_DIR/evaluation_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"
echo "Results will be saved to: $RESULTS_DIR"

# Summary file
SUMMARY_FILE="$RESULTS_DIR/summary.txt"
echo "GPTQ + nvfp4a16 Scale Rounding Evaluation Summary" > "$SUMMARY_FILE"
echo "Date: $(date)" >> "$SUMMARY_FILE"
echo "Tasks: $EVAL_TASKS" >> "$SUMMARY_FILE"
echo "Few-shot: $EVAL_NUM_FEWSHOT" >> "$SUMMARY_FILE"
echo "Limit: ${EVAL_LIMIT:-full dataset}" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# Arrays to collect results
declare -a MODEL_NAMES
declare -a BASELINE_FILES
declare -a IMPROVED_FILES

# Function to evaluate a single model
evaluate_model() {
    local MODEL_ID="$1"
    local MODEL_NAME="$(basename $MODEL_ID)"

    echo ""
    echo "=============================================================================="
    echo "Evaluating: $MODEL_ID"
    echo "=============================================================================="

    # Step 1: Quantize both models
    echo ""
    echo "STEP 1: Quantizing $MODEL_NAME (with and without scale rounding)"
    echo "=============================================================================="

    cd "$SCRIPT_DIR"
    source "$RHDEV_VENV/bin/activate"

    python "$QUANTIZE_SCRIPT" "$MODEL_ID"

    deactivate

    # Derived paths
    BASELINE_DIR="${MODEL_NAME}-GPTQ-NVFP4A16-NoRounding"
    IMPROVED_DIR="${MODEL_NAME}-GPTQ-NVFP4A16-WithRounding"

    echo ""
    echo "Quantization complete!"
    echo "  Baseline: $BASELINE_DIR"
    echo "  Improved: $IMPROVED_DIR"

    # Step 2: Evaluate baseline model
    echo ""
    echo "=============================================================================="
    echo "STEP 2: Evaluating BASELINE model (round_scales=False)"
    echo "=============================================================================="

    BASELINE_OUTPUT="$RESULTS_DIR/results_${MODEL_NAME}_baseline.json"
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

    # Step 3: Evaluate improved model
    echo ""
    echo "=============================================================================="
    echo "STEP 3: Evaluating IMPROVED model (round_scales=True)"
    echo "=============================================================================="

    IMPROVED_OUTPUT="$RESULTS_DIR/results_${MODEL_NAME}_improved.json"
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

    # Step 4: Save results for later comparison
    echo ""
    echo "=============================================================================="
    echo "Evaluation complete for $MODEL_NAME"
    echo "=============================================================================="
    echo "Results saved:"
    echo "  Baseline: $BASELINE_OUTPUT"
    echo "  Improved: $IMPROVED_OUTPUT"

    # Collect results
    MODEL_NAMES+=("$MODEL_NAME")
    BASELINE_FILES+=("$BASELINE_OUTPUT")
    IMPROVED_FILES+=("$IMPROVED_OUTPUT")

    # Cleanup quantized models to save space (optional)
    # rm -rf "$BASELINE_DIR" "$IMPROVED_DIR"
}

# Evaluate all models
for MODEL_ID in "${MODELS[@]}"; do
    evaluate_model "$MODEL_ID"
done

# Generate comprehensive comparison
echo ""
echo "=============================================================================="
echo "GENERATING FINAL COMPARISON"
echo "=============================================================================="

# Python script to generate comparison
python3 << 'EOF'
import json
import sys
from pathlib import Path

# Read model info from environment
import os
models = os.environ.get('MODEL_NAMES', '').split()
baseline_files = os.environ.get('BASELINE_FILES', '').split()
improved_files = os.environ.get('IMPROVED_FILES', '').split()

if not models:
    print("No results to compare")
    sys.exit(0)

print("\n" + "=" * 80)
print("GPTQ + nvfp4a16 SCALE ROUNDING COMPARISON")
print("=" * 80)
print()

# Collect all results
results = {}
for i, model_name in enumerate(models):
    baseline_file = baseline_files[i]
    improved_file = improved_files[i]

    with open(baseline_file) as f:
        baseline_data = json.load(f)
    with open(improved_file) as f:
        improved_data = json.load(f)

    results[model_name] = {
        'baseline': baseline_data.get('results', {}),
        'improved': improved_data.get('results', {})
    }

# Print detailed results for each model
for model_name, data in results.items():
    print(f"\n{'-' * 80}")
    print(f"Model: {model_name}")
    print(f"{'-' * 80}")

    baseline = data['baseline']
    improved = data['improved']

    # Get all tasks
    tasks = set(baseline.keys()) | set(improved.keys())

    for task in sorted(tasks):
        print(f"\n  Task: {task}")

        if task in baseline and task in improved:
            baseline_metrics = baseline[task]
            improved_metrics = improved[task]

            # Get all metrics
            all_metrics = set(baseline_metrics.keys()) | set(improved_metrics.keys())

            # Print each metric
            for metric in sorted(all_metrics):
                if metric in baseline_metrics and metric in improved_metrics:
                    baseline_val = baseline_metrics[metric]
                    improved_val = improved_metrics[metric]

                    # Only print numeric metrics
                    if isinstance(baseline_val, (int, float)) and isinstance(improved_val, (int, float)):
                        diff = improved_val - baseline_val

                        # Determine if higher is better (most metrics) or lower is better (perplexity, loss)
                        lower_is_better = 'perplexity' in metric.lower() or 'loss' in metric.lower()

                        if lower_is_better:
                            pct_change = -100 * diff / baseline_val if baseline_val != 0 else 0
                            symbol = "↓" if diff < 0 else "↑"
                            better = "✓" if diff < 0 else "✗"
                        else:
                            pct_change = 100 * diff / baseline_val if baseline_val != 0 else 0
                            symbol = "↑" if diff > 0 else "↓"
                            better = "✓" if diff > 0 else "✗"

                        print(f"    {metric:25s}: ", end="")
                        print(f"Baseline={baseline_val:.4f}  Improved={improved_val:.4f}  ", end="")
                        print(f"Δ={diff:+.4f} ({pct_change:+.2f}%) {symbol} {better}")

# Print summary table
print("\n" + "=" * 80)
print("SUMMARY TABLE")
print("=" * 80)

# Collect key metrics for summary
summary_data = []
for model_name, data in results.items():
    baseline = data['baseline']
    improved = data['improved']

    for task in sorted(set(baseline.keys()) & set(improved.keys())):
        baseline_metrics = baseline[task]
        improved_metrics = improved[task]

        # Common important metrics
        important_metrics = ['word_perplexity', 'byte_perplexity', 'bits_per_byte', 'acc', 'acc_norm']

        for metric in important_metrics:
            if metric in baseline_metrics and metric in improved_metrics:
                baseline_val = baseline_metrics[metric]
                improved_val = improved_metrics[metric]

                if isinstance(baseline_val, (int, float)) and isinstance(improved_val, (int, float)):
                    diff = improved_val - baseline_val
                    lower_is_better = 'perplexity' in metric.lower() or 'loss' in metric.lower()

                    if lower_is_better:
                        pct_change = -100 * diff / baseline_val if baseline_val != 0 else 0
                    else:
                        pct_change = 100 * diff / baseline_val if baseline_val != 0 else 0

                    summary_data.append({
                        'model': model_name,
                        'task': task,
                        'metric': metric,
                        'baseline': baseline_val,
                        'improved': improved_val,
                        'improvement': pct_change
                    })

if summary_data:
    print()
    print(f"{'Model':<30} {'Task':<20} {'Metric':<20} {'Baseline':<12} {'Improved':<12} {'Improvement':>12}")
    print("-" * 120)
    for row in summary_data:
        print(f"{row['model']:<30} {row['task']:<20} {row['metric']:<20} "
              f"{row['baseline']:<12.4f} {row['improved']:<12.4f} {row['improvement']:>+11.2f}%")

print("\n" + "=" * 80)
print("Key:")
print("  ✓ = Improved (better with round_scales=True)")
print("  ✗ = Degraded (worse with round_scales=True)")
print("  ↑ = Value increased")
print("  ↓ = Value decreased")
print("=" * 80)
print()

EOF

# Export variables for Python script
export MODEL_NAMES="${MODEL_NAMES[*]}"
export BASELINE_FILES="${BASELINE_FILES[*]}"
export IMPROVED_FILES="${IMPROVED_FILES[*]}"

python3 << 'EOFPY' | tee -a "$SUMMARY_FILE"
import json
import sys
from pathlib import Path
import os

models = os.environ.get('MODEL_NAMES', '').split()
baseline_files = os.environ.get('BASELINE_FILES', '').split()
improved_files = os.environ.get('IMPROVED_FILES', '').split()

if not models:
    print("No results to compare")
    sys.exit(0)

print("\n" + "=" * 80)
print("GPTQ + nvfp4a16 SCALE ROUNDING COMPARISON")
print("=" * 80)
print()

results = {}
for i, model_name in enumerate(models):
    baseline_file = baseline_files[i]
    improved_file = improved_files[i]

    with open(baseline_file) as f:
        baseline_data = json.load(f)
    with open(improved_file) as f:
        improved_data = json.load(f)

    results[model_name] = {
        'baseline': baseline_data.get('results', {}),
        'improved': improved_data.get('results', {})
    }

for model_name, data in results.items():
    print(f"\n{'-' * 80}")
    print(f"Model: {model_name}")
    print(f"{'-' * 80}")

    baseline = data['baseline']
    improved = data['improved']

    tasks = set(baseline.keys()) | set(improved.keys())

    for task in sorted(tasks):
        print(f"\n  Task: {task}")

        if task in baseline and task in improved:
            baseline_metrics = baseline[task]
            improved_metrics = improved[task]

            all_metrics = set(baseline_metrics.keys()) | set(improved_metrics.keys())

            for metric in sorted(all_metrics):
                if metric in baseline_metrics and metric in improved_metrics:
                    baseline_val = baseline_metrics[metric]
                    improved_val = improved_metrics[metric]

                    if isinstance(baseline_val, (int, float)) and isinstance(improved_val, (int, float)):
                        diff = improved_val - baseline_val
                        lower_is_better = 'perplexity' in metric.lower() or 'loss' in metric.lower()

                        if lower_is_better:
                            pct_change = -100 * diff / baseline_val if baseline_val != 0 else 0
                            symbol = "↓" if diff < 0 else "↑"
                            better = "✓" if diff < 0 else "✗"
                        else:
                            pct_change = 100 * diff / baseline_val if baseline_val != 0 else 0
                            symbol = "↑" if diff > 0 else "↓"
                            better = "✓" if diff > 0 else "✗"

                        print(f"    {metric:25s}: ", end="")
                        print(f"Baseline={baseline_val:.4f}  Improved={improved_val:.4f}  ", end="")
                        print(f"Δ={diff:+.4f} ({pct_change:+.2f}%) {symbol} {better}")

print("\n" + "=" * 80)
print("SUMMARY TABLE")
print("=" * 80)

summary_data = []
for model_name, data in results.items():
    baseline = data['baseline']
    improved = data['improved']

    for task in sorted(set(baseline.keys()) & set(improved.keys())):
        baseline_metrics = baseline[task]
        improved_metrics = improved[task]

        important_metrics = ['word_perplexity', 'byte_perplexity', 'bits_per_byte', 'acc', 'acc_norm']

        for metric in important_metrics:
            if metric in baseline_metrics and metric in improved_metrics:
                baseline_val = baseline_metrics[metric]
                improved_val = improved_metrics[metric]

                if isinstance(baseline_val, (int, float)) and isinstance(improved_val, (int, float)):
                    diff = improved_val - baseline_val
                    lower_is_better = 'perplexity' in metric.lower() or 'loss' in metric.lower()

                    if lower_is_better:
                        pct_change = -100 * diff / baseline_val if baseline_val != 0 else 0
                    else:
                        pct_change = 100 * diff / baseline_val if baseline_val != 0 else 0

                    summary_data.append({
                        'model': model_name,
                        'task': task,
                        'metric': metric,
                        'baseline': baseline_val,
                        'improved': improved_val,
                        'improvement': pct_change
                    })

if summary_data:
    print()
    print(f"{'Model':<30} {'Task':<20} {'Metric':<20} {'Baseline':<12} {'Improved':<12} {'Improvement':>12}")
    print("-" * 120)
    for row in summary_data:
        print(f"{row['model']:<30} {row['task']:<20} {row['metric']:<20} "
              f"{row['baseline']:<12.4f} {row['improved']:<12.4f} {row['improvement']:>+11.2f}%")

print("\n" + "=" * 80)
print("Key:")
print("  ✓ = Improved (better with round_scales=True)")
print("  ✗ = Degraded (worse with round_scales=True)")
print("  ↑ = Value increased")
print("  ↓ = Value decreased")
print("=" * 80)
print()
EOFPY

echo ""
echo "=============================================================================="
echo "ALL EVALUATIONS COMPLETE"
echo "=============================================================================="
echo "Results saved to: $RESULTS_DIR"
echo "Summary file: $SUMMARY_FILE"
echo ""
