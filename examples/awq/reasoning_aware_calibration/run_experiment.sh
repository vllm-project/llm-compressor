#!/bin/bash
# Generate on-policy traces -> AWQ with each calibration set -> lm-eval.
#
# vLLM and llm-compressor often pin incompatible dependency versions, so the
# three stages can run with different interpreters:
#   GEN_PYTHON    python with vllm installed            (trace generation)
#   QUANT_PYTHON  python with llmcompressor installed   (calibration)
#   LM_EVAL       lm_eval executable with the vllm extra (evaluation)
#
# Stages whose outputs already exist are skipped, so the script can be re-run
# after an interruption or with more ARMS.
#
# Each entry of ARMS is "<arm>" or "<arm>:<loss-token-budget>", see
# calibrate_awq.py. The defaults reproduce the table in README.md.
set -e
cd "$(dirname "$0")"

MODEL=${MODEL:-Qwen/Qwen3-1.7B}
DATASET=${DATASET:-mlabonne/open-perfectblend}
NUM_SAMPLES=${NUM_SAMPLES:-256}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}  # length cap of the generated traces
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-2048}  # length cap of calibration samples
SCHEME=${SCHEME:-W4A16}
ARMS=${ARMS:-"default offpolicy onpolicy onpolicy:offpolicy offpolicy:onpolicy"}
TASKS=${TASKS:-mmlu_cot_llama}
MAX_GEN_TOKS=${MAX_GEN_TOKS:-4096}
TP=${TP:-1}  # tensor parallel size of vLLM, for generation and evaluation
LIMIT=${LIMIT:-}  # e.g. LIMIT=20 for 20 questions per subject
EVAL_ARGS=${EVAL_ARGS:-}  # extra lm_eval arguments, e.g. "--num_fewshot 0"
OUT=${OUT:-out/$(basename "$MODEL")-$(basename "$DATASET")}
GEN_PYTHON=${GEN_PYTHON:-python}
QUANT_PYTHON=${QUANT_PYTHON:-python}
LM_EVAL=${LM_EVAL:-lm_eval}

mkdir -p "$OUT"
if [ ! -s "$OUT/traces.jsonl" ]; then
    $GEN_PYTHON generate_on_policy_traces.py --model "$MODEL" \
        --dataset "$DATASET" --num-samples "$NUM_SAMPLES" \
        --max-new-tokens "$MAX_NEW_TOKENS" --tensor-parallel-size "$TP" \
        --output "$OUT/traces.jsonl" --prompts-output "$OUT/prompts.jsonl"
fi

models=("$MODEL")
for spec in $ARMS; do
    arm=${spec%%:*}
    budget=; [ "$spec" != "$arm" ] && budget=${spec#*:}
    save_dir=$OUT/$SCHEME-$arm${budget:+-budget-$budget}
    if [ ! -f "$save_dir/config.json" ]; then
        $QUANT_PYTHON calibrate_awq.py --model "$MODEL" --arm "$arm" \
            ${budget:+--loss-token-budget "$budget"} --scheme "$SCHEME" \
            --dataset "$DATASET" --max-seq-length "$MAX_SEQ_LENGTH" \
            --traces "$OUT/traces.jsonl" --prompts "$OUT/prompts.jsonl" \
            --save-dir "$save_dir"
    fi
    models+=("$save_dir")
done

# Qwen3 thinks before answering: evaluate in thinking mode and leave room for
# the reasoning, otherwise answers are cut off before they are extracted
for path in "${models[@]}"; do
    name=$(basename "$path"); [ "$path" = "$MODEL" ] && name=bf16
    result=$OUT/eval-$TASKS${LIMIT:+-limit$LIMIT}/$name
    if [ ! -d "$result" ]; then
        $LM_EVAL --model vllm \
            --model_args pretrained="$path",enable_thinking=True,think_end_token="</think>",max_model_len=$((MAX_GEN_TOKS + 1024)),gpu_memory_utilization=0.85,tensor_parallel_size=$TP,seed=0 \
            --tasks "$TASKS" --apply_chat_template \
            --gen_kwargs max_gen_toks=$MAX_GEN_TOKS --batch_size auto \
            ${LIMIT:+--limit $LIMIT} $EVAL_ARGS --log_samples --output_path "$result"
    fi
done
