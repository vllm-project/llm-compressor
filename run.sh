#!/bin/bash
# set -e  # Exit on error

source $DEV_ENV_DIR/.bash_profile

# Base directory for evaluations
EVAL_BASE_DIR="./eval_results"
mkdir -p $EVAL_BASE_DIR

# # Function to evaluate a base (unquantized) model
# eval_base_model() {
#     local model_id=$1
#     local eval_task=$2
#     local max_model_len=$3

#     local model_short_name=$(echo $model_id | sed 's|.*/||')

#     echo "============================================"
#     echo "Evaluating base model: $model_id"
#     echo "============================================"

#     EVAL_OUTPUT_DIR="$EVAL_BASE_DIR/${model_short_name}_base_eval"
#     mkdir -p $EVAL_OUTPUT_DIR

#     # Run evaluation with tensor_parallel
#     echo "EVAL (with tensor_parallel_size=2)"
#     run 4 lm_eval \
#         --model vllm \
#         --model_args pretrained=$model_id,dtype=auto,max_model_len=$max_model_len,add_bos_token=True,tensor_parallel_size=2 \
#         --tasks $eval_task \
#         --batch_size auto \
#         --output_path $EVAL_OUTPUT_DIR

#     # If tensor_parallel eval failed, retry without it
#     if [ $? -ne 0 ]; then
#         echo "Evaluation with tensor_parallel failed, retrying without tensor_parallel..."
#         run 1 lm_eval \
#             --model vllm \
#             --model_args pretrained=$model_id,dtype=auto,max_model_len=$max_model_len,add_bos_token=True \
#             --tasks $eval_task \
#             --batch_size auto \
#             --output_path $EVAL_OUTPUT_DIR
#     fi

#     # If vllm failed, try hf
#     if [ $? -ne 0 ]; then
#         echo "Evaluation with vllm failed, retrying with hf backend..."
#         run 1 lm_eval \
#             --model hf \
#             --model_args pretrained=$model_id,dtype=auto,max_model_len=$max_model_len,add_bos_token=True \
#             --tasks $eval_task \
#             --batch_size auto \
#             --output_path $EVAL_OUTPUT_DIR
#     fi

#     if [ $? -eq 0 ]; then
#         echo "Base model evaluation successful for $model_id"
#     else
#         echo "Base model evaluation failed for $model_id"
#     fi

#     echo ""
# }

# Function to run example, evaluate, and cleanup
run_and_eval() {
    local script_name=$1
    local save_dir=$2
    local eval_task=$3
    local max_model_len=$4

    echo "============================================"
    echo "Running: $script_name"
    echo "============================================"

    # Run the quantization script
    uva rhdev
    export TMPDIR=/mnt/nvme-data/tmp
    run python $script_name
    local quant_status=$?
    uva vllm

    # Check if quantization was successful
    if [ $quant_status -eq 0 ]; then
        echo "Quantization successful, starting evaluation..."

        # Set up eval output directory
        EVAL_OUTPUT_DIR="$EVAL_BASE_DIR/${save_dir}_eval"
        mkdir -p $EVAL_OUTPUT_DIR

        # Run evaluation with tensor_parallel
        echo "EVAL (with tensor_parallel_size=2)"
        run 4 lm_eval \
            --model vllm \
            --model_args pretrained=./$save_dir,dtype=auto,max_model_len=$max_model_len,add_bos_token=True,tensor_parallel_size=2 \
            --tasks $eval_task \
            --batch_size auto \
            --output_path $EVAL_OUTPUT_DIR

        # If tensor_parallel eval failed, retry without it
        if [ $? -ne 0 ]; then
            echo "Evaluation with tensor_parallel failed, retrying without tensor_parallel..."
            run 1 lm_eval \
                --model vllm \
                --model_args pretrained=./$save_dir,dtype=auto,max_model_len=$max_model_len,add_bos_token=True \
                --tasks $eval_task \
                --batch_size auto \
                --output_path $EVAL_OUTPUT_DIR
        fi

        # If vllm failed, try hf
        if [ $? -ne 0 ]; then
            echo "Evaluation with vllm failed, retrying with hf backend..."
            run 1 lm_eval \
                --model hf \
                --model_args pretrained=./$save_dir,dtype=auto,max_model_len=$max_model_len,add_bos_token=True \
                --tasks $eval_task \
                --batch_size auto \
                --output_path $EVAL_OUTPUT_DIR
        fi

        # Check if evaluation was successful
        if [ $? -eq 0 ]; then
            echo "Evaluation successful, cleaning up model files..."
            rm -rf ./$save_dir
            echo "Cleanup complete for $save_dir"
        else
            echo "Evaluation failed for $save_dir, keeping model files for inspection"
        fi
    else
        echo "Quantization failed for $script_name"
    fi

    echo ""
}
run_and_eval "examples/awq/llama_example.py" "Meta-Llama-3-8B-Instruct-awq-w4a16-asym" "gsm8k" 2048
run_and_eval "examples/awq/llama_example_smooth.py" "Meta-Llama-3-8B-Instruct-awq-w4a16-asym-smooth" "gsm8k" 2048

run_and_eval "examples/awq/fp8_block_llama_example.py" "Meta-Llama-3-8B-Instruct-awq-fp8-block" "gsm8k" 2048
run_and_eval "examples/awq/fp8_block_llama_example_smooth.py" "Meta-Llama-3-8B-Instruct-awq-fp8-block-smooth" "gsm8k" 2048

run_and_eval "examples/awq/fp8_dynamic_llama_example.py" "Meta-Llama-3-8B-Instruct-awq-fp8-dynamic" "gsm8k" 2048
run_and_eval "examples/awq/fp8_dynamic_llama_example_smooth.py" "Meta-Llama-3-8B-Instruct-awq-fp8-dynamic-smooth" "gsm8k" 2048

run_and_eval "examples/awq/w4a8_fp8_llama_example.py" "Meta-Llama-3-8B-Instruct-awq-w4afp8" "gsm8k" 2048
run_and_eval "examples/awq/w4a8_fp8_llama_example_smooth.py" "Meta-Llama-3-8B-Instruct-awq-w4afp8-smooth" "gsm8k" 2048

run_and_eval "examples/awq/qwen3_moe_example.py" "Qwen3-30B-A3B-awq-w4a16" "gsm8k" 2048
run_and_eval "examples/awq/qwen3_moe_example_smooth.py" "Qwen3-30B-A3B-awq-w4a16-smooth" "gsm8k" 2048

run_and_eval "examples/awq/qwen3_coder_moe_example.py" "Qwen3-Coder-30B-A3B-Instruct-awq-w4a16" "gsm8k" 2048
run_and_eval "examples/awq/qwen3_coder_moe_example_smooth.py" "Qwen3-Coder-30B-A3B-Instruct-awq-w4a16-smooth" "gsm8k" 2048

run_and_eval "examples/awq/qwen3_next_example.py" "Qwen3-Next-80B-A3B-Thinking-awq-w4a16" "gsm8k" 2048
run_and_eval "examples/awq/qwen3_next_example_smooth.py" "Qwen3-Next-80B-A3B-Thinking-awq-w4a16-smooth" "gsm8k" 2048

# Base model evaluation
echo "============================================"
echo "Starting base model evaluations"
echo "============================================"
# uva vllm
# eval_base_model "meta-llama/Meta-Llama-3-8B-Instruct" "gsm8k" 2048

echo "============================================"
echo "All runs complete!"
echo "Evaluation results saved in: $EVAL_BASE_DIR"
echo "============================================"
