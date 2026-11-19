#!/usr/bin/env bash
# Serve and evaluate BF16 plus every completed GPTQ checkpoint in sequence.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B}"
OUT="${OUT:-${XDG_CACHE_HOME:-/tmp}/llm-compressor-rac-gptq-qwen3-8b}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-${MODEL_PATH:-$MODEL_ID}}"
VLLM_BIN="${VLLM_BIN:-vllm}"
INSPECT="${INSPECT:-inspect}"
INSPECT_PYTHON="${INSPECT_PYTHON:-python}"
GPU_IDS="${CUDA_VISIBLE_DEVICES:-0}"
PORT="${PORT:-8000}"
TP="${TP:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-1800}"
EVAL_LIMIT="${EVAL_LIMIT:-}"
FORCE="${FORCE:-0}"
CONDITIONS="${CONDITIONS:-bf16 FP8_DYNAMIC-full FP8_DYNAMIC-dataset_masked FP8_DYNAMIC-onpolicy_masked MXFP4A8-full MXFP4A8-dataset_masked MXFP4A8-onpolicy_masked W4A16-full W4A16-dataset_masked W4A16-onpolicy_masked}"

SERVER_PID=""
stop_server() {
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
  fi
}
trap stop_server EXIT INT TERM

start_and_eval() {
  local condition="$1" model_path="$2" model_label="$3"
  local log_dir="$OUT/evaluations${EVAL_LIMIT:+-limit$EVAL_LIMIT}/$condition"
  local waited=0
  local eval_args=()
  mkdir -p "$log_dir"

  if [[ "$FORCE" != 1 && -n "$(find "$log_dir" -maxdepth 1 -name '*.math_verify.json' -print -quit)" ]]; then
    echo "Skipping existing scored evaluation for $condition"
    return
  fi

  echo "Serving $condition from $model_path on GPUs $GPU_IDS"
  CUDA_VISIBLE_DEVICES="$GPU_IDS" "$VLLM_BIN" serve "$model_path" \
    --served-model-name "$model_label" \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking": true}' \
    --host 127.0.0.1 --port "$PORT" \
    > "$log_dir/serve.log" 2>&1 &
  SERVER_PID="$!"

  until curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "vLLM exited while loading $condition; see $log_dir/serve.log" >&2
      tail -n 80 "$log_dir/serve.log" >&2 || true
      stop_server
      return 1
    fi
    if (( waited >= VLLM_ENGINE_READY_TIMEOUT_S )); then
      echo "Timed out waiting for vLLM to load $condition; see $log_dir/serve.log" >&2
      stop_server
      return 1
    fi
    sleep 5
    waited=$((waited + 5))
  done

  if [[ -n "$EVAL_LIMIT" ]]; then
    eval_args=(--limit "$EVAL_LIMIT")
  fi
  MODEL="$model_label" PORT="$PORT" LOG_DIR="$log_dir" \
    INSPECT="$INSPECT" INSPECT_PYTHON="$INSPECT_PYTHON" \
    "$SCRIPT_DIR/run_eval.sh" "${eval_args[@]}"
  stop_server
  sleep 3
}

read -r -a CONDITIONS_ARRAY <<< "$CONDITIONS"
for condition in "${CONDITIONS_ARRAY[@]}"; do
  if [[ "$condition" == bf16 ]]; then
    model_path="$BASE_MODEL_PATH"
  else
    scheme="${condition%%-*}"
    arm="${condition#*-}"
    case "$scheme" in
      FP8_DYNAMIC) scheme_dir="FP8_DYNAMIC" ;;
      MXFP4A8) scheme_dir="MXFP4XFP8" ;;
      W4A16) scheme_dir="W4A16" ;;
      *) echo "Unknown evaluation condition: $condition" >&2; exit 2 ;;
    esac
    model_path="$OUT/models/$scheme_dir/$arm"
    if [[ ! -s "$model_path/config.json" ]]; then
      echo "Missing checkpoint for $condition: $model_path" >&2
      exit 2
    fi
  fi
  start_and_eval "$condition" "$model_path" "Qwen3-8B-$condition"
done
