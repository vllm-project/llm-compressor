#!/bin/bash

source /home/HDCharles/rhdev/bin/activate

MODEL_ID="${1:-meta-llama/Meta-Llama-3-8B-Instruct}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Model: $MODEL_ID"
echo ""

run() {
    echo "=== $1 ==="
    shift
    python "$@" --model_id "$MODEL_ID"
    echo ""
}

run W2A16  "$SCRIPT_DIR/weight_only.py"           --num_bits 2
run W3A16  "$SCRIPT_DIR/weight_only.py"           --num_bits 3
run W5A16  "$SCRIPT_DIR/weight_only.py"           --num_bits 5
run W7A16  "$SCRIPT_DIR/weight_only.py"           --num_bits 7
run W2A4   "$SCRIPT_DIR/weight_and_activation.py" --weight_bits 2 --act_bits 4
run W3A4   "$SCRIPT_DIR/weight_and_activation.py" --weight_bits 3 --act_bits 4
run W5A8   "$SCRIPT_DIR/weight_and_activation.py" --weight_bits 5 --act_bits 8
run W7A8   "$SCRIPT_DIR/weight_and_activation.py" --weight_bits 7 --act_bits 8

echo "Done. Saved directories:"
ls -d "$MODEL_ID"*/  2>/dev/null | sed 's/^/  /' || true
