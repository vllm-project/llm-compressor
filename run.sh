#!/bin/bash
# set -e  # Exit on error

source $DEV_ENV_DIR/.bash_profile

# Base directory for evaluations
EVAL_BASE_DIR="./eval_results"
mkdir -p $EVAL_BASE_DIR

# Function to run example, evaluate, and cleanup
run_and_eval() {
    local script_name=$1
    local num_gpus=$2
    local save_dir=$3
    local eval_task=$4
    local max_model_len=$5

    echo "============================================"
    echo "Running: $script_name"
    echo "============================================"

    # Run the quantization script
    uva rhdev
    if [ $num_gpus -gt 1 ]; then
        run $num_gpus torchrun --nproc_per_node=$num_gpus $script_name
    else
        run 1 python $script_name
    fi
    uva vllm
    # Check if quantization was successful
    if [ $? -eq 0 ]; then
        echo "Quantization successful, starting evaluation..."

        # Set up eval output directory
        EVAL_OUTPUT_DIR="$EVAL_BASE_DIR/${save_dir}_eval"
        mkdir -p $EVAL_OUTPUT_DIR

        # Run evaluation with tensor_parallel
        echo "EVAL (with tensor_parallel_size=2)"
        run 2 lm_eval \
            --model vllm \
            --model_args pretrained=./$save_dir,dtype=auto,max_model_len=$max_model_len,add_bos_token=True,tensor_parallel_size=2 \
            --tasks $eval_task \
            --batch_size auto \
            --output_path $EVAL_OUTPUT_DIR

        # If vllm without TP failed, try with expert parallel (useful for MoE models)
        if [ $? -ne 0 ]; then
            echo "Evaluation without tensor_parallel failed, retrying with expert parallel..."
            run 2 lm_eval \
                --model vllm \
                --model_args pretrained=./$save_dir,dtype=auto,max_model_len=$max_model_len,add_bos_token=True,enable_expert_parallel=True \
                --tasks $eval_task \
                --batch_size auto \
                --output_path $EVAL_OUTPUT_DIR
        fi

        # If tensor/expert_parallel eval failed, retry without it
        if [ $? -ne 0 ]; then
            echo "Evaluation with tensor_parallel failed, retrying without tensor_parallel..."
            run 1 lm_eval \
                --model vllm \
                --model_args pretrained=./$save_dir,dtype=auto,max_model_len=$max_model_len,add_bos_token=True \
                --tasks $eval_task \
                --batch_size auto \
                --output_path $EVAL_OUTPUT_DIR
        fi

        # try enforce eager
        if [ $? -ne 0 ]; then
            echo "Evaluation with tensor_parallel failed, retrying without tensor_parallel..."
            run 1 lm_eval \
                --model vllm \
                --model_args pretrained=./$save_dir,dtype=auto,max_model_len=$max_model_len,add_bos_token=True,enforce_eager=True \
                --tasks $eval_task \
                --batch_size auto \
                --output_path $EVAL_OUTPUT_DIR
        fi

        # If vllm failed, try hf
        if [ $? -ne 0 ]; then
            echo "Evaluation with tensor_parallel failed, retrying without tensor_parallel..."
            run 1 lm_eval \
                --model hf \
                --model_args pretrained=./$save_dir,dtype=auto,add_bos_token=True \
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




# AWQ examples

run_and_eval "examples/awq/llama_example_ddp.py" 4 "Meta-Llama-3-8B-Instruct-awq-asym-DDP4" "gsm8k" 2048
run_and_eval "examples/awq/llama_example.py" 1 "Meta-Llama-3-8B-Instruct-awq-asym" "gsm8k" 2048

run_and_eval "examples/awq/llama_example_with_masking_ddp.py" 4 "Meta-Llama-3-8B-Instruct-awq-asym-masked-DDP4" "gsm8k" 2048
run_and_eval "examples/awq/llama_example_with_masking.py" 1 "Meta-Llama-3-8B-Instruct-awq-asym-masked" "gsm8k" 2048

run_and_eval "examples/awq/qwen3_vl_30b_example_ddp.py" 4 "Qwen3-VL-30B-A3B-Instruct-AWQ-W4A16-g32-DDP4" "gsm8k" 2048
run_and_eval "examples/awq/qwen3-vl-30b-a3b-Instruct-example.py" 1 "Qwen3-VL-30B-A3B-Instruct-AWQ-W4A16-mse-seq" "gsm8k" 2048

run_and_eval "examples/awq/qwen3_coder_moe_example_ddp.py" 4 "Qwen3-Coder-30B-A3B-Instruct-W4A16-awq-DDP4" "gsm8k" 2048
run_and_eval "examples/awq/qwen3_coder_moe_example.py" 1 "Qwen3-Coder-30B-A3B-Instruct-W4A16-awq" "gsm8k" 2048

run_and_eval "examples/awq/qwen3_moe_example_ddp.py" 4 "Qwen3-30B-A3B-awq-sym-DDP4" "gsm8k" 2048
run_and_eval "examples/awq/qwen3_moe_example.py" 1 "Qwen3-30B-A3B-awq-sym" "gsm8k" 2048

run_and_eval "examples/awq/qwen3_next_example_ddp.py" 4 "Qwen3-Next-80B-A3B-Thinking-awq-sym-DDP4" "gsm8k" 2048
run_and_eval "examples/awq/qwen3_next_example.py" 1 "Qwen3-Next-80B-A3B-Thinking-awq-sym" "gsm8k" 2048
