#!/bin/bash
# GPTQ torch.compile Full Benchmark Suite
# Runs all scenarios and generates summary report

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/artifacts"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${OUTPUT_DIR}/run_${TIMESTAMP}"

echo "=============================================="
echo "GPTQ torch.compile Benchmark Suite"
echo "=============================================="
echo "Output directory: ${RESULTS_DIR}"
echo "Started at: $(date)"
echo ""

mkdir -p "${RESULTS_DIR}"

# Add local bin to PATH
export PATH="$HOME/.local/bin:$PATH"

# Log environment
echo "Capturing environment info..."
python3 -c "
import torch
import sys
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
" | tee "${RESULTS_DIR}/environment.txt"

echo ""
echo "=============================================="
echo "Scenario A: TinyLlama (baseline + compiled)"
echo "=============================================="
python3 "${SCRIPT_DIR}/benchmark_gptq_compile.py" \
    --scenario tinyllama \
    --mode both \
    --num_runs 3 \
    --output_dir "${RESULTS_DIR}" \
    2>&1 | tee "${RESULTS_DIR}/tinyllama/log.txt"

echo ""
echo "=============================================="
echo "Scenario A: Numerical Check"
echo "=============================================="
python3 "${SCRIPT_DIR}/benchmark_gptq_compile.py" \
    --scenario tinyllama \
    --mode numerical_check \
    --output_dir "${RESULTS_DIR}" \
    2>&1 | tee -a "${RESULTS_DIR}/tinyllama/log.txt"

echo ""
echo "=============================================="
echo "Scenario B: Qwen2.5-3B (baseline + compiled)"
echo "=============================================="
python3 "${SCRIPT_DIR}/benchmark_gptq_compile.py" \
    --scenario qwen3b \
    --mode both \
    --num_runs 3 \
    --output_dir "${RESULTS_DIR}" \
    2>&1 | tee "${RESULTS_DIR}/qwen3b/log.txt"

echo ""
echo "=============================================="
echo "Scenario C: Blocksize Stress Test (64)"
echo "=============================================="
python3 "${SCRIPT_DIR}/benchmark_gptq_compile.py" \
    --scenario blocksize_64 \
    --mode compiled \
    --num_runs 2 \
    --output_dir "${RESULTS_DIR}" \
    2>&1 | tee "${RESULTS_DIR}/blocksize_64/log.txt"

echo ""
echo "=============================================="
echo "Scenario C: Blocksize Stress Test (256)"
echo "=============================================="
python3 "${SCRIPT_DIR}/benchmark_gptq_compile.py" \
    --scenario blocksize_256 \
    --mode compiled \
    --num_runs 2 \
    --output_dir "${RESULTS_DIR}" \
    2>&1 | tee "${RESULTS_DIR}/blocksize_256/log.txt"

# Generate final summary
echo ""
echo "=============================================="
echo "FINAL SUMMARY"
echo "=============================================="

cat > "${RESULTS_DIR}/SUMMARY.md" << 'EOF'
# GPTQ torch.compile Benchmark Results

## Test Matrix

| Scenario | Model | Purpose |
|----------|-------|---------|
| tinyllama | TinyLlama-1.1B | Fast iteration, sanity check |
| qwen3b | Qwen2.5-3B | Production representative |
| blocksize_64 | TinyLlama + block=64 | Shape variability stress |
| blocksize_256 | TinyLlama + block=256 | Shape variability stress |

## Results

EOF

# Append individual summaries
for scenario in tinyllama qwen3b; do
    if [ -f "${RESULTS_DIR}/${scenario}/summary.md" ]; then
        echo "### ${scenario}" >> "${RESULTS_DIR}/SUMMARY.md"
        tail -n +3 "${RESULTS_DIR}/${scenario}/summary.md" >> "${RESULTS_DIR}/SUMMARY.md"
        echo "" >> "${RESULTS_DIR}/SUMMARY.md"
    fi
done

cat "${RESULTS_DIR}/SUMMARY.md"

echo ""
echo "=============================================="
echo "Benchmark Complete!"
echo "=============================================="
echo "Results saved to: ${RESULTS_DIR}"
echo "Finished at: $(date)"
