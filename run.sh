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
    local eval_tasks=$4  # comma-separated list of tasks, e.g. "gsm8k,wikitext"
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

        local all_evals_passed=true

        # Loop over each eval task
        IFS=',' read -ra TASK_ARRAY <<< "$eval_tasks"
        for eval_task in "${TASK_ARRAY[@]}"; do
            echo "--------------------------------------------"
            echo "Evaluating task: $eval_task"
            echo "--------------------------------------------"

            # Set up eval output directory per task
            EVAL_OUTPUT_DIR="$EVAL_BASE_DIR/${save_dir}_eval/${eval_task}"
            mkdir -p $EVAL_OUTPUT_DIR

            local task_passed=false

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

            if [ $? -eq 0 ]; then
                echo "Evaluation of $eval_task successful."
                task_passed=true
            else
                echo "Evaluation of $eval_task failed."
                all_evals_passed=false
            fi
        done

        # Clean up model files only if all evaluations passed
        if [ "$all_evals_passed" = true ]; then
            echo "All evaluations successful, cleaning up model files..."
            rm -rf ./$save_dir
            echo "Cleanup complete for $save_dir"
        else
            echo "Some evaluations failed for $save_dir, keeping model files for inspection"
        fi
    else
        echo "Quantization failed for $script_name"
    fi

    echo ""
}

run_and_eval "examples/awq/llama_example_smooth.py" 1 "Meta-Llama-3-8B-Instruct-awq-w4a16-asym-smooth" "gsm8k,wikitext" 2048
run_and_eval "examples/awq/llama_example.py" 1 "Meta-Llama-3-8B-Instruct-awq-w4a16-asym" "gsm8k,wikitext" 2048


run_and_eval "examples/awq/fp8_block_llama_example_smooth.py" 1 "Meta-Llama-3-8B-Instruct-awq-fp8-block-smooth" "gsm8k,wikitext" 2048
run_and_eval "examples/awq/fp8_block_llama_example.py" 1 "Meta-Llama-3-8B-Instruct-awq-fp8-block" "gsm8k,wikitext" 2048


# run_and_eval "examples/awq/fp8_dynamic_llama_example_smooth.py" 1 "Meta-Llama-3-8B-Instruct-awq-fp8-dynamic-smooth" "gsm8k,wikitext" 2048
# run_and_eval "examples/awq/fp8_dynamic_llama_example.py" 1 "Meta-Llama-3-8B-Instruct-awq-fp8-dynamic" "gsm8k,wikitext" 2048


# run_and_eval "examples/awq/w4a8_fp8_llama_example_smooth.py" 1 "Meta-Llama-3-8B-Instruct-awq-w4afp8-smooth" "gsm8k,wikitext" 2048
# run_and_eval "examples/awq/w4a8_fp8_llama_example.py" 1 "Meta-Llama-3-8B-Instruct-awq-w4afp8" "gsm8k,wikitext" 2048


# run_and_eval "examples/awq/qwen3_moe_example_smooth.py" 1 "Qwen3-30B-A3B-awq-w4a16-smooth" "gsm8k,wikitext" 2048
# run_and_eval "examples/awq/qwen3_moe_example.py" 1 "Qwen3-30B-A3B-awq-w4a16" "gsm8k,wikitext" 2048


# run_and_eval "examples/awq/qwen3_coder_moe_example_smooth.py" 1 "Qwen3-Coder-30B-A3B-Instruct-awq-w4a16-smooth" "gsm8k,wikitext" 2048
# run_and_eval "examples/awq/qwen3_coder_moe_example.py" 1 "Qwen3-Coder-30B-A3B-Instruct-awq-w4a16" "gsm8k,wikitext" 2048


# run_and_eval "examples/awq/qwen3_next_example_smooth.py" 1 "Qwen3-Next-80B-A3B-Thinking-awq-w4a16-smooth" "gsm8k,wikitext" 2048
# run_and_eval "examples/awq/qwen3_next_example.py" 1 "Qwen3-Next-80B-A3B-Thinking-awq-w4a16" "gsm8k,wikitext" 2048


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
