#!/bin/bash
# Setup script for LongCat-2.0 NVFP4A16 quantization pipeline
#
# Checks out the required branches of vLLM, transformers, and llm-compressor,
# installs them, and downloads the baseline LongCat-2.0 model.
#
# Usage:
#   bash setup_longcat.sh [--repos-dir /path/to/repos] [--model-dir /path/to/models]
#
# Defaults:
#   --repos-dir: ~/repos
#   --model-dir: ~/hf_hub

set -euo pipefail

REPOS_DIR="${HOME}/repos"
MODEL_DIR="${HOME}/hf_hub"

while [[ $# -gt 0 ]]; do
    case $1 in
        --repos-dir) REPOS_DIR="$2"; shift 2 ;;
        --model-dir) MODEL_DIR="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PATCH_DIR="$SCRIPT_DIR/patches"

mkdir -p "$REPOS_DIR" "$MODEL_DIR"

echo "=== Setup Configuration ==="
echo "Repos directory: $REPOS_DIR"
echo "Model directory: $MODEL_DIR"
echo ""

# --- 1. vLLM (longcat-2.0 branch) ---
echo "=== [1/4] Setting up vLLM ==="
if [ -d "$REPOS_DIR/vllm" ]; then
    echo "vLLM repo exists, fetching longcat-2.0 branch..."
    cd "$REPOS_DIR/vllm"
    git remote add hdcharles https://github.com/HDCharles/vllm.git 2>/dev/null || true
    git fetch hdcharles longcat-2.0
    git checkout longcat-2.0
    git reset --hard hdcharles/longcat-2.0
else
    echo "Cloning vLLM (longcat-2.0 branch)..."
    git clone --branch longcat-2.0 https://github.com/HDCharles/vllm.git "$REPOS_DIR/vllm"
fi
echo "Installing vLLM..."
pip install -e "$REPOS_DIR/vllm" 2>&1 | tail -3
echo ""

# --- 2. transformers (patch on top of main) ---
echo "=== [2/4] Setting up transformers ==="
if [ -d "$REPOS_DIR/transformers" ]; then
    echo "transformers repo exists, resetting to main..."
    cd "$REPOS_DIR/transformers"
    git fetch origin
    git checkout main
    git reset --hard origin/main
else
    echo "Cloning transformers..."
    git clone https://github.com/huggingface/transformers.git "$REPOS_DIR/transformers"
fi
cd "$REPOS_DIR/transformers"
git checkout -B longcat2
if [ -f "$PATCH_DIR/transformers_longcat2.patch" ]; then
    echo "Applying LongCat-2.0 patch..."
    git am "$PATCH_DIR/transformers_longcat2.patch"
else
    echo "WARNING: Patch file not found at $PATCH_DIR/transformers_longcat2.patch"
    echo "  The transformers changes must be applied manually."
fi
echo "Installing transformers..."
pip install -e "$REPOS_DIR/transformers" 2>&1 | tail -3
echo ""

# --- 3. llm-compressor (longcat2 branch) ---
echo "=== [3/4] Setting up llm-compressor ==="
if [ -d "$REPOS_DIR/llm-compressor" ]; then
    echo "llm-compressor repo exists, fetching longcat2 branch..."
    cd "$REPOS_DIR/llm-compressor"
    git fetch origin
    git checkout longcat2
    git reset --hard origin/longcat2
else
    echo "Cloning llm-compressor (longcat2 branch)..."
    git clone --branch longcat2 https://github.com/vllm-project/llm-compressor.git "$REPOS_DIR/llm-compressor"
fi
echo "Installing llm-compressor..."
pip install -e "$REPOS_DIR/llm-compressor[dev]" 2>&1 | tail -3
echo ""

# --- 4. Download baseline model ---
echo "=== [4/4] Downloading LongCat-2.0 baseline (~1.6TB) ==="
huggingface-cli download meituan-longcat/LongCat-2.0 \
    --local-dir "$MODEL_DIR/LongCat-2.0" \
    --local-dir-use-symlinks False
echo "Model downloaded to: $MODEL_DIR/LongCat-2.0"
echo ""

echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Quantize:"
echo "     MODEL_ID=$MODEL_DIR/LongCat-2.0 HF_HUB_DIR=$MODEL_DIR \\"
echo "       python $REPOS_DIR/llm-compressor/examples/model_free_ptq/longcat_2_nvfp4a16.py"
echo ""
echo "  2. Serve with vLLM (adjust --tensor-parallel-size and --cpu-offload-gb for your hardware):"
echo "     vllm serve $MODEL_DIR/LongCat-2.0-NVFP4A16 --trust-remote-code \\"
echo "       --tensor-parallel-size 4 --max-model-len 512 --cpu-offload-gb 100 \\"
echo "       --gpu-memory-utilization 0.99 --enforce-eager \\"
echo "       --kernel-config '{\"moe_backend\": \"emulation\"}' \\"
echo "       --reasoning-parser longcat --override-generation-config '{\"thinking\": true}'"
echo ""
echo "  3. Eval:"
echo "     python $REPOS_DIR/vllm/tests/evals/gsm8k/gsm8k_eval.py \\"
echo "       --port 8000 --num-questions 1319 --num-shots 1 --max-tokens 256"
