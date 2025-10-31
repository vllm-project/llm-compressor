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

script_path=$(dirname "${BASH_SOURCE[0]}")
if [ -d "$CONFIG" ]; then
    echo "Config is provided as a folder: $CONFIG"
    CONFIGS=`ls "$CONFIG"`
elif [ -f "$CONFIG" ]; then
    echo "Config is provided as a file: $CONFIG"
    CONFIGS=`cat "$CONFIG"`
fi
echo "$CONFIGS"

# Parse list of configs.
for MODEL_CONFIG in $(echo -e "$CONFIGS" | sed "s|^|${script_path}/configs/|")
do
    LOCAL_SUCCESS=0

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

done

exit "$SUCCESS"
