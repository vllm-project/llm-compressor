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

run W2A16       "$SCRIPT_DIR/weight_only.py"           --scheme W2A16
run W3A16       "$SCRIPT_DIR/weight_only.py"           --scheme W3A16
run W5A16       "$SCRIPT_DIR/weight_only.py"           --scheme W5A16
run W6A16       "$SCRIPT_DIR/weight_only.py"           --scheme W6A16
run W7A16       "$SCRIPT_DIR/weight_only.py"           --scheme W7A16

run W2A4        "$SCRIPT_DIR/weight_and_activation.py" --weight_bits 2 --act_bits 4
run W3A4        "$SCRIPT_DIR/weight_and_activation.py" --weight_bits 3 --act_bits 4
run W2A8        "$SCRIPT_DIR/weight_and_activation.py" --weight_bits 2 --act_bits 8
run W3A8        "$SCRIPT_DIR/weight_and_activation.py" --weight_bits 3 --act_bits 8
run W5A8        "$SCRIPT_DIR/weight_and_activation.py" --weight_bits 5 --act_bits 8
run W6A8        "$SCRIPT_DIR/weight_and_activation.py" --weight_bits 6 --act_bits 8
run W7A8        "$SCRIPT_DIR/weight_and_activation.py" --weight_bits 7 --act_bits 8

run W2A16asym   "$SCRIPT_DIR/weight_only.py"           --scheme W2A16 --asymmetric
run W3A16asym   "$SCRIPT_DIR/weight_only.py"           --scheme W3A16 --asymmetric
run W5A16asym   "$SCRIPT_DIR/weight_only.py"           --scheme W5A16 --asymmetric
run W6A16asym   "$SCRIPT_DIR/weight_only.py"           --scheme W6A16 --asymmetric
run W7A16asym   "$SCRIPT_DIR/weight_only.py"           --scheme W7A16 --asymmetric

run W2A4asym    "$SCRIPT_DIR/weight_and_activation.py" --weight_bits 2 --act_bits 4 --asymmetric
run W3A4asym    "$SCRIPT_DIR/weight_and_activation.py" --weight_bits 3 --act_bits 4 --asymmetric
run W2A8asym    "$SCRIPT_DIR/weight_and_activation.py" --weight_bits 2 --act_bits 8 --asymmetric
run W3A8asym    "$SCRIPT_DIR/weight_and_activation.py" --weight_bits 3 --act_bits 8 --asymmetric
run W5A8asym    "$SCRIPT_DIR/weight_and_activation.py" --weight_bits 5 --act_bits 8 --asymmetric
run W6A8asym    "$SCRIPT_DIR/weight_and_activation.py" --weight_bits 6 --act_bits 8 --asymmetric
run W7A8asym    "$SCRIPT_DIR/weight_and_activation.py" --weight_bits 7 --act_bits 8 --asymmetric

# echo "=== mixed_quant_5plus ==="
# python "$SCRIPT_DIR/mixed_quant_5plus.py"
# echo ""

echo "Done. Saved directories:"
ls -d "$MODEL_ID"*/  2>/dev/null | sed 's/^/  /' || true

source /home/HDCharles/vllm/bin/activate
"$SCRIPT_DIR/run_evals.sh"
