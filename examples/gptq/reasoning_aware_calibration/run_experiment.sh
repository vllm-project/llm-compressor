#!/usr/bin/env bash
# Generate on-policy MATH-500 traces and calibrate all requested GPTQ arms.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$SCRIPT_DIR"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B}"
MODEL_PATH="${MODEL_PATH:-$MODEL_ID}"
OUT="${OUT:-${XDG_CACHE_HOME:-/tmp}/llm-compressor-rac-gptq-qwen3-8b}"
GEN_PYTHON="${GEN_PYTHON:-python}"
QUANT_PYTHON="${QUANT_PYTHON:-python}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
GEN_TP="${GEN_TP:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-}"
CONDITIONS="${CONDITIONS:-FP8_DYNAMIC/full FP8_DYNAMIC/dataset_masked FP8_DYNAMIC/onpolicy_masked MXFP4XFP8/full MXFP4XFP8/dataset_masked MXFP4XFP8/onpolicy_masked W4A16/full W4A16/dataset_masked W4A16/onpolicy_masked}"

IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
GPU_COUNT="${#GPU_IDS[@]}"
if [[ -z "$MAX_PARALLEL" ]]; then
  MAX_PARALLEL="$GPU_COUNT"
fi
if (( MAX_PARALLEL < 1 || GPU_COUNT < 1 )); then
  echo "CUDA_VISIBLE_DEVICES and MAX_PARALLEL must name at least one GPU" >&2
  exit 2
fi
if (( MAX_PARALLEL > GPU_COUNT )); then
  MAX_PARALLEL="$GPU_COUNT"
fi

mkdir -p "$OUT/data" "$OUT/models" "$OUT/logs"
export FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

check_generation_metadata() {
  "$GEN_PYTHON" - "$1" "$MODEL_PATH" "$MAX_NEW_TOKENS" <<'PY'
import json
import sys

path, model, max_new_tokens = sys.argv[1:]
with open(path, encoding="utf-8") as file:
    metadata = json.load(file)
expected = {"model": model, "max_new_tokens": int(max_new_tokens), "num_samples": 500}
mismatches = {
    key: (metadata.get(key), value)
    for key, value in expected.items()
    if metadata.get(key) != value
}
if mismatches:
    raise SystemExit(f"Existing trace metadata mismatch (found, expected): {mismatches}")
PY
}

check_calibration_metadata() {
  "$QUANT_PYTHON" - "$1" "$MODEL_PATH" "$2" "$3" <<'PY'
import json
import sys

path, model, scheme, arm = sys.argv[1:]
with open(path, encoding="utf-8") as file:
    metadata = json.load(file)
expected = {"model": model, "scheme": scheme, "arm": arm}
mismatches = {
    key: (metadata.get(key), value)
    for key, value in expected.items()
    if metadata.get(key) != value
}
if mismatches:
    raise SystemExit(f"Existing checkpoint metadata mismatch (found, expected): {mismatches}")
PY
}

if [[ ! -s "$OUT/data/generation_complete.json" || ! -s "$OUT/data/traces.jsonl" || ! -s "$OUT/data/prompts.jsonl" ]]; then
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$GEN_PYTHON" generate_on_policy_traces.py \
    --model "$MODEL_PATH" \
    --output-dir "$OUT/data" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --tensor-parallel-size "$GEN_TP" \
    2>&1 | tee "$OUT/logs/generate.log"
else
  check_generation_metadata "$OUT/data/generation_complete.json"
  echo "Reusing existing traces in $OUT/data"
fi

read -r -a CONDITIONS_ARRAY <<< "$CONDITIONS"
echo "Calibrating ${#CONDITIONS_ARRAY[@]} conditions; up to $MAX_PARALLEL jobs on GPUs $CUDA_VISIBLE_DEVICES"
for ((batch_start = 0; batch_start < ${#CONDITIONS_ARRAY[@]}; batch_start += MAX_PARALLEL)); do
  batch_pids=()
  batch_labels=()
  batch_end=$((batch_start + MAX_PARALLEL))
  if (( batch_end > ${#CONDITIONS_ARRAY[@]} )); then
    batch_end="${#CONDITIONS_ARRAY[@]}"
  fi

  for ((condition_index = batch_start; condition_index < batch_end; condition_index++)); do
    condition="${CONDITIONS_ARRAY[$condition_index]}"
    scheme="${condition%%/*}"
    arm="${condition#*/}"
    save_dir="$OUT/models/$scheme/$arm"
    if [[ -s "$save_dir/config.json" && -s "$save_dir/calibration_stats.json" ]]; then
      check_calibration_metadata "$save_dir/calibration_stats.json" "$scheme" "$arm"
      echo "Skipping existing $scheme/$arm at $save_dir"
      continue
    fi

    gpu_index=$((condition_index - batch_start))
    gpu_id="${GPU_IDS[$gpu_index]}"
    echo "Starting $scheme/$arm on GPU $gpu_id"
    (
      export CUDA_VISIBLE_DEVICES="$gpu_id"
      export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
      "$QUANT_PYTHON" calibrate_gptq.py \
        --model "$MODEL_PATH" \
        --prompts "$OUT/data/prompts.jsonl" \
        --traces "$OUT/data/traces.jsonl" \
        --arm "$arm" \
        --scheme "$scheme" \
        --save-dir "$save_dir"
    ) > "$OUT/logs/${scheme}-${arm}.log" 2>&1 &
    batch_pids+=("$!")
    batch_labels+=("$scheme/$arm")
  done

  batch_status=0
  for batch_index in "${!batch_pids[@]}"; do
    if wait "${batch_pids[$batch_index]}"; then
      echo "Completed ${batch_labels[$batch_index]}"
    else
      log_label="${batch_labels[$batch_index]//\//-}"
      echo "Failed ${batch_labels[$batch_index]}; see $OUT/logs/$log_label.log" >&2
      batch_status=1
    fi
  done
  if (( batch_status != 0 )); then
    exit "$batch_status"
  fi
done
