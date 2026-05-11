#!/bin/bash

usage() {
  echo "Usage: $0 -c <config> -t <test> -g <test_group> -s <save_dir>"
  exit 1
}

while getopts "c:t:g:s:" OPT; do
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
    s )
        SAVE_DIR="$OPTARG"
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

if [ -z "$SAVE_DIR" ]; then
  echo "Error: -s save_dir is required for rhaiis testing."
  usage
fi

if [ ! -z "$GROUP" ]; then
  echo "Test group is specified: $GROUP"
fi

CONFIGS=`ls "$CONFIG"`

SUCCESS=0

# Parse list of configs and add save_dir
rm -rf $SAVE_DIR/configs
mkdir -p $SAVE_DIR/configs
for MODEL_CONFIG in $(echo -e "$CONFIGS" | sed "s|^|${script_path}/configs/|")
do
    FILE_NAME=$(basename $MODEL_CONFIG)
    CONFIG_FILE=$SAVE_DIR/configs/$FILE_NAME

    save_dir=$(cat $MODEL_CONFIG | grep 'save_dir:' | cut -d' ' -f2)
    model=$(cat $MODEL_CONFIG | grep 'model:' | cut -d'/' -f2)
    scheme=$(cat $MODEL_CONFIG | grep 'scheme:' | cut -d' ' -f2)
    test_group=$(cat $MODEL_CONFIG | grep 'test_group:' | cut -d'"' -f2)

    # run test if test group is not specified or the config matching the specified test group
    if [ -z "$GROUP" ] || [[ "${test_group}" == "$GROUP" ]]; then

        # add or overwrite save_dir for each model
        if [[ -z "$save_dir" ]]; then
            { cat $MODEL_CONFIG; echo -e "\nsave_dir: $SAVE_DIR/$model-$scheme"; } > $CONFIG_FILE
        else
            { cat $MODEL_CONFIG | grep -v 'save_dir'; echo "save_dir: $SAVE_DIR/$save_dir"; } > $CONFIG_FILE
        fi

        echo "=== RUNNING MODEL: $CONFIG_FILE ==="
        cat $CONFIG_FILE

        LOCAL_SUCCESS=0
        export TEST_DATA_FILE="$CONFIG_FILE"
        pytest \
            --capture=tee-sys \
            "$TEST" || LOCAL_SUCCESS=$?

        if [[ $LOCAL_SUCCESS == 0 ]]; then
            echo "=== PASSED MODEL: $CONFIG_FILE ==="
        else
            echo "=== FAILED MODEL: $CONFIG_FILE ==="
        fi

        SUCCESS=$((SUCCESS + LOCAL_SUCCESS))
    fi

done

exit "$SUCCESS"
