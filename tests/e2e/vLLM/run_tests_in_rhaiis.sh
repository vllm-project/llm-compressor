#!/bin/bash

usage() {
  echo "Usage: $0 -c <config> -t <test> -s <save_dir>"
  exit 1
}

while getopts "c:t:s:" OPT; do
  case ${OPT} in
    c )
        CONFIG="$OPTARG"
        ;;
    t )
        TEST="$OPTARG"
        ;;
    s )
        SAVE_DIR="$OPTARG"
        ;;
    \? )
        exit 1
        ;;
  esac
done

if [[ -z "$CONFIG" || -z "$TEST" || -z "$SAVE_DIR" ]]; then
  echo "Error: -c, -t, and -s are required."
  usage
fi

script_path=$(dirname "${BASH_SOURCE[0]}")
if [ -d "$CONFIG" ]; then
    echo "Config is provided as a folder: $CONFIG"
    CONFIGS=`ls "$CONFIG"`
elif [ -f "$CONFIG" ]; then
    echo "Config is provided as a file: $CONFIG"
    CONFIGS=`cat "$CONFIG"`
fi

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

    # add or overwrite save_dir for each model
    if [[ -z "$save_dir" ]]; then
      { cat $MODEL_CONFIG; echo -e "\nsave_dir: $SAVE_DIR/$model-$scheme"; } > $CONFIG_FILE
    else
      { cat $MODEL_CONFIG | grep -v 'save_dir'; echo "save_dir: $SAVE_DIR/$save_dir"; } > $CONFIG_FILE
    fi

    #{ cat $MODEL_CONFIG | grep -v 'save_dir'; echo "save_dir: $SAVE_DIR"; } > $CONFIG_FILE

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

done

exit "$SUCCESS"
