#!/bin/bash

SUCCESS=0

# Parse list of configs.
MODEL_CONFIGS="$PWD/tests/e2e/vLLM/configs"

# Function to process a file
process_model_config() {
    local MODEL_CONFIG=$1
    LOCAL_SUCCESS=0
    
    echo "=== RUNNING MODEL: $MODEL_CONFIG ==="
    export TEST_DATA_FILE=${MODEL_CONFIG}
    pytest -s $PWD/tests/e2e/vLLM/test_vllm.py || LOCAL_SUCCESS=$?
    if [[ $LOCAL_SUCCESS == 0 ]]; then
        echo "=== PASSED MODEL: ${MODEL_CONFIG} ==="
    else
        echo "=== FAILED MODEL: ${MODEL_CONFIG} ==="
    fi
    SUCCESS=$((SUCCESS + LOCAL_SUCCESS))
}

# Use find to locate all files in MODEL_CONFIGS and process them
find "$MODEL_CONFIGS" -type f | while read -r MODEL_CONFIG
do
    process_model_config "$MODEL_CONFIG"
done

if [ "${SUCCESS}" -eq "0" ]; then
    exit 0
else
    exit 1
fi
