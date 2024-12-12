#!/bin/bash

SUCCESS=0

while getopts "c:t:" OPT; do
  case ${OPT} in
    c ) 
        CONFIG="$OPTARG"
        ;;
    t )
        TEST="$OPTARG"
        ;;
    \? )
        exit 1
        ;;
  esac
done

# Function to process a file
process_model_config() {
    local MODEL_CONFIG=$1
    LOCAL_SUCCESS=0
    
    echo "=== RUNNING MODEL: $MODEL_CONFIG ==="
    export TEST_DATA_FILE=${MODEL_CONFIG}
    pytest \
        -r a \
        --capture=tee-sys \
        --junitxml="test-results/e2e-$(date +%s).xml" \
        "$TEST" || LOCAL_SUCCESS=$?

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
