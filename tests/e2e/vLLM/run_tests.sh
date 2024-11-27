#!/bin/bash

SUCCESS=0

# Parse list of configs.
MODEL_CONFIGS="$PWD/tests/e2e/vLLM/configs"

for MODEL_CONFIG in "$MODEL_CONFIGS"/*
do
    LOCAL_SUCCESS=0

    echo "=== RUNNING MODEL: $MODEL_CONFIG ==="

    export TEST_DATA_FILE="$MODEL_CONFIG"
    pytest \
        --capture=tee-sys \
        --junitxml="test-results/e2e-$(date +%s).xml" \
        "$PWD/tests/e2e/vLLM/test_vllm.py" || LOCAL_SUCCESS=$?

    if [[ $LOCAL_SUCCESS == 0 ]]; then
        echo "=== PASSED MODEL: $MODEL_CONFIG ==="
    else
        echo "=== FAILED MODEL: $MODEL_CONFIG ==="
    fi

    SUCCESS=$((SUCCESS + LOCAL_SUCCESS))

done

exit "$SUCCESS"
