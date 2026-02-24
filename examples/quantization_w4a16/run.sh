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
    run $num_gpus torchrun --nproc_per_node=$num_gpus $script_name

    # Check if quantization was successful
    if [ $? -eq 0 ]; then
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
            echo "Evaluation with tensor_parallel failed, retrying without tensor_parallel..."
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

# W4A16
# run_and_eval "llama3_ddp_example.py" 4 "Meta-Llama-3-8B-Instruct-W4A16-G128-DDP4" "gsm8k" 2048  # .7111 .7127
# run_and_eval "llama3_ddp_example.py" 1 "Meta-Llama-3-8B-Instruct-W4A16-G128-DDP1" "gsm8k" 2048 # .702 .702

# run_and_eval "qwen3_vl_8b_gptq_int4_ddp_example.py" 4 "Qwen3-VL-8B-Instruct-GPTQ-W4A16-G128-DDP4" "gsm8k" 2048 # .8514 .8476
# run_and_eval "qwen3_vl_8b_gptq_int4_ddp_example.py" 1 "Qwen3-VL-8B-Instruct-GPTQ-W4A16-G128-DDP1" "gsm8k" 2048 # .8491 .8355

# run_and_eval "qwen3_30b_moe_gptq_int4_ddp_example.py" 4 "Qwen3-30B-A3B-GPTQ-W4A16-G128-DDP4" "gsm8k" 2048 # 0.8666 .8749
# run_and_eval "qwen3_30b_moe_gptq_int4_ddp_example.py" 1 "Qwen3-30B-A3B-GPTQ-W4A16-G128-DDP1" "gsm8k" 2048 # .8681 .8772

# run_and_eval "llama4_gptq_int4_ddp_example.py" 4 "Llama-4-Scout-17B-16E-Instruct-GPTQ-W4A16-G128-DDP4" "gsm8k" 8192 # .906 .887
# run_and_eval "llama4_gptq_int4_ddp_example.py" 1 "Llama-4-Scout-17B-16E-Instruct-GPTQ-W4A16-G128-DDP1" "gsm8k" 8192 # 0.9075 0.8878

# run_and_eval "qwen3_vl_235b_moe_gptq_int4_ddp_example.py" 4 "Qwen3-VL-235B-A22B-Instruct-GPTQ-W4A16-G128-DDP4" "gsm8k" 2048 # fail
# run_and_eval "qwen3_vl_235b_moe_gptq_int4_ddp_example.py" 8 "Qwen3-VL-235B-A22B-Instruct-GPTQ-W4A16-G128-DDP8" "gsm8k" 2048 # fail
# run_and_eval "qwen3_vl_235b_moe_gptq_int4_ddp_example.py" 1 "Qwen3-VL-235B-A22B-Instruct-GPTQ-W4A16-G128-DDP1" "gsm8k" 2048 # some disk offload error it looks like


# NVFP4
run_and_eval "qwen3_vl_8b_gptq_nvfp4_ddp_example.py" 4 "Qwen3-VL-8B-Instruct-GPTQ-NVFP4A16-DDP4" "gsm8k" 2048
# run_and_eval "qwen3_vl_8b_gptq_nvfp4_ddp_example.py" 1 "Qwen3-VL-8B-Instruct-GPTQ-NVFP4A16-DDP1" "gsm8k" 2048

# run_and_eval "qwen3_30b_moe_gptq_nvfp4_ddp_example.py" 4 "Qwen3-30B-A3B-GPTQ-NVFP4A16-DDP4" "gsm8k" 2048
# run_and_eval "qwen3_30b_moe_gptq_nvfp4_ddp_example.py" 1 "Qwen3-30B-A3B-GPTQ-NVFP4A16-DDP1" "gsm8k" 2048

# run_and_eval "llama4_gptq_nvfp4_ddp_example.py" 4 "Llama-4-Scout-17B-16E-Instruct-GPTQ-NVFP4A16-DDP4" "gsm8k" 8192 
# run_and_eval "llama4_gptq_nvfp4_ddp_example.py" 1 "Llama-4-Scout-17B-16E-Instruct-GPTQ-NVFP4A16-DDP1" "gsm8k" 8192

# run_and_eval "qwen3_vl_235b_moe_nvfp4_ddp_example.py" 8 "Qwen3-VL-235B-A22B-Instruct-GPTQ-NVFP4A16-DDP8" "gsm8k" 2048
# run_and_eval "qwen3_vl_235b_moe_nvfp4_ddp_example.py" 1 "Qwen3-VL-235B-A22B-Instruct-GPTQ-NVFP4A16-DDP1" "gsm8k" 2048


echo "============================================"
echo "All runs complete!"
echo "Evaluation results saved in: $EVAL_BASE_DIR"
echo "============================================"
