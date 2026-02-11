source $DEV_ENV_DIR/.bash_profile

# Define arrays of options to test
MODEL_IDS=(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    # "meta-llama/Meta-Llama-3-8B-Instruct"
)

DEVICE_MAPS=(
    "cpu"
    # "disk"
    # "cuda"
)

GPU_COUNTS=(
    # 1
    2
    # 4
)

export MAX_GPUS=4
export OUTPUT_FILE="./metrics.json"

# Loop through all combinations
for MODEL_ID in "${MODEL_IDS[@]}"; do
    for DEVICE_MAP in "${DEVICE_MAPS[@]}"; do
        for NUM_GPUS in "${GPU_COUNTS[@]}"; do
            export SAVE_DIR=$MODEL_ID-$NUM_GPUS-$DEVICE_MAP

            echo "=========================================="
            echo "Running experiment:"
            echo "  MODEL_ID: $MODEL_ID"
            echo "  DEVICE_MAP: $DEVICE_MAP"
            echo "  NUM_GPUS: $NUM_GPUS"
            echo "  SAVE_DIR: $SAVE_DIR"
            echo "=========================================="

            run $NUM_GPUS torchrun --nproc_per_node=$NUM_GPUS test_ddp.py \
                --model_id $MODEL_ID \
                --device_map $DEVICE_MAP \
                --save_dir $SAVE_DIR \
                --output_file $OUTPUT_FILE

            EVAL_OUTPUT_DIR="./${SAVE_DIR}_eval"

            # run $MAX_GPUS lm_eval \
            #     --model vllm \
            #     --model_args pretrained=./$SAVE_DIR,dtype=auto,max_model_len=2048,add_bos_token=True,tensor_parallel_size=$MAX_GPUS \
            #     --tasks gsm8k \
            #     --batch_size auto \
            #     --output_path $EVAL_OUTPUT_DIR

            # Append evaluation metrics to output file
            if [ -f "${EVAL_OUTPUT_DIR}/results.json" ]; then
                python3 -c "
                    import json
                    import os

                    # Read existing metrics
                    with open('$OUTPUT_FILE', 'r') as f:
                        metrics = json.load(f)

                    # Read eval results
                    with open('${EVAL_OUTPUT_DIR}/results.json', 'r') as f:
                        eval_results = json.load(f)

                    # Add eval results to the last entry (current experiment)
                    if metrics:
                        metrics[-1]['lm_eval_results'] = eval_results

                    # Write back
                    with open('$OUTPUT_FILE', 'w') as f:
                        json.dump(metrics, f, indent=2)
                "
                echo "Evaluation metrics added to $OUTPUT_FILE"
            else
                echo "Warning: Evaluation results not found at ${EVAL_OUTPUT_DIR}/results.json"
            fi

            echo "Experiment completed."
            echo ""
        done
    done
done

echo "All experiments completed!"