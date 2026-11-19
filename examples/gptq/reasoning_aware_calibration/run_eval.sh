#!/usr/bin/env bash
# Run the held-out Inspect MATH task against an already-running vLLM server.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL="${MODEL:-Qwen3-8B}"
PORT="${PORT:-8000}"
EPOCHS="${EPOCHS:-1}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
MAX_CONNECTIONS="${MAX_CONNECTIONS:-32}"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/out/eval-logs/$MODEL}"
INSPECT="${INSPECT:-inspect}"
INSPECT_PYTHON="${INSPECT_PYTHON:-python}"

calibration_ids="$SCRIPT_DIR/assets/calibration_ids.txt"
eval_ids="$SCRIPT_DIR/assets/eval_sample_ids.txt"
calibration_count="$(wc -l < "$calibration_ids")"
eval_count="$(wc -l < "$eval_ids")"
if [[ "$calibration_count" -ne 500 || "$eval_count" -ne 4500 ]]; then
  echo "Expected 500 calibration and 4,500 evaluation IDs; got $calibration_count and $eval_count" >&2
  exit 1
fi
if comm -12 <(sort "$calibration_ids") <(sort "$eval_ids") | grep -q .; then
  echo "Calibration and evaluation sample IDs overlap" >&2
  exit 1
fi

sample_limit=""
extra_args=()
while (($#)); do
  case "$1" in
    --limit)
      if (($# < 2)); then
        echo "--limit requires a positive integer" >&2
        exit 2
      fi
      sample_limit="$2"
      shift 2
      ;;
    --limit=*)
      sample_limit="${1#--limit=}"
      shift
      ;;
    *)
      extra_args+=("$1")
      shift
      ;;
  esac
done

if [[ -n "$sample_limit" ]]; then
  if [[ ! "$sample_limit" =~ ^[1-9][0-9]*$ ]] || (( sample_limit > eval_count )); then
    echo "--limit must be between 1 and $eval_count" >&2
    exit 2
  fi
  sample_ids="$(head -n "$sample_limit" "$eval_ids" | paste -sd, -)"
else
  sample_ids="$(paste -sd, - < "$eval_ids")"
fi

export VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:${PORT}/v1}"
export VLLM_API_KEY="${VLLM_API_KEY:-local}"
mkdir -p "$LOG_DIR"

"$INSPECT" eval inspect_evals/math \
  --model "vllm/${MODEL}" \
  --epochs "$EPOCHS" --epochs-reducer mean \
  --temperature 0.6 --top-p 0.95 \
  --max-tokens "$MAX_TOKENS" \
  --max-connections "$MAX_CONNECTIONS" \
  --sample-id "$sample_ids" \
  --generate-config "$SCRIPT_DIR/assets/generate_config.json" \
  -M client_timeout=7200 \
  --log-dir "$LOG_DIR" \
  "${extra_args[@]}"

"$INSPECT_PYTHON" "$SCRIPT_DIR/score_math_verify.py" "$LOG_DIR" \
  --json-out "$LOG_DIR/results.math_verify.json"
