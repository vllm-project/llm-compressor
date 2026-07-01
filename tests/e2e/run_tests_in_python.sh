#!/bin/bash

SUCCESS=0

while getopts "c:t:g:" OPT; do
  case ${OPT} in
    c )
        CONFIG="$OPTARG"
        ;;
    t )
        TEST="$OPTARG"
        ;;
    g )
        GROUP="$OPTARG"
        ;;
    \? )
        exit 1
        ;;
  esac
done

script_path=$(dirname "${BASH_SOURCE[0]}")
if [ -z "$CONFIG" ]; then
  CONFIG="${script_path}/configs"
fi

if [ -z "$TEST" ]; then
  TEST="${script_path}/test_vllm.py"
fi

if [ ! -z "$GROUP" ]; then
    echo "Test group is specified: $GROUP"
fi

# Parse list of configs.
for MODEL_CONFIG in "$CONFIG"/*
do
    LOCAL_SUCCESS=0

    # run test if test group is not specified or the config matching the specified test group
    test_group=$(cat $MODEL_CONFIG | grep 'test_group:' | cut -d'"' -f2)
    if [ -z "$GROUP" ] || [[ "$test_group" == "$GROUP" ]]; then

        echo "=== RUNNING MODEL: $MODEL_CONFIG ==="

        export TEST_DATA_FILE="$MODEL_CONFIG"
        pytest \
            --capture=tee-sys \
            "$TEST" || LOCAL_SUCCESS=$?

        if [[ $LOCAL_SUCCESS == 0 ]]; then
            echo "=== PASSED MODEL: $MODEL_CONFIG ==="
        else
            echo "=== FAILED MODEL: $MODEL_CONFIG ==="
        fi

        SUCCESS=$((SUCCESS + LOCAL_SUCCESS))
    fi

done

exit "$SUCCESS"
