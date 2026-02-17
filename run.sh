source $DEV_ENV_DIR/.bash_profile

# Define arrays of options to test
MODEL_IDS=(
    # "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    "meta-llama/Meta-Llama-3-8B-Instruct"
    # "Qwen/Qwen3-30B-A3B"
)

DEVICE_MAPS=(
    "cpu"
    # "disk"
    # "cuda"
)

GPU_COUNTS=(
    1
    2
    3
    4
    5
    6
    7
)

export OUTPUT_FILE="./metrics.json"

echo "${MODEL_IDS[@]}"
echo "${DEVICE_MAPS[@]}"
echo "${GPU_COUNTS[@]}"

# Loop through all combinations
for MODEL_ID in "${MODEL_IDS[@]}"; do
    for DEVICE_MAP in "${DEVICE_MAPS[@]}"; do
        for NUM_GPUS in "${GPU_COUNTS[@]}"; do
            SAVE_DIR=$MODEL_ID-$NUM_GPUS-$DEVICE_MAP
            EVAL_OUTPUT_DIR="./${SAVE_DIR}_eval"
            TIMESTAMP=$(date +%Y%m%d_%H%M%S)
            QUANT_LOG="${SAVE_DIR}-${TIMESTAMP}-quant.log"
            EVAL_LOG="${SAVE_DIR}-${TIMESTAMP}-eval.log"

            echo "=========================================="
            echo "Running experiment:"
            echo "  MODEL_ID: $MODEL_ID"
            echo "  DEVICE_MAP: $DEVICE_MAP"
            echo "  NUM_GPUS: $NUM_GPUS"
            echo "  SAVE_DIR: $SAVE_DIR"
            echo "  QUANT_LOG: $QUANT_LOG"
            echo "  EVAL_LOG: $EVAL_LOG"
            echo "=========================================="
            run $NUM_GPUS torchrun --nproc_per_node=$NUM_GPUS test_ddp.py \
                --model_id $MODEL_ID \
                --device_map $DEVICE_MAP \
                --save_dir $SAVE_DIR \
                --output_file $OUTPUT_FILE \
                2>&1 | tee "$QUANT_LOG"

            echo "EVAL"
            run 4 lm_eval \
                --model vllm \
                --model_args pretrained=./$SAVE_DIR,dtype=auto,max_model_len=2048,add_bos_token=True,tensor_parallel_size=4 \
                --tasks gsm8k \
                --batch_size auto \
                --output_path $EVAL_OUTPUT_DIR \
                2>&1 | tee "$EVAL_LOG"


            if [ -d "$SAVE_DIR" ]; then
                echo "Deleting $SAVE_DIR"
                rm -rf "$SAVE_DIR"
            fi
        done
    done
done


echo "All experiments completed!"



