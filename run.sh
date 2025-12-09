# export TEST_DATA_FILE="${REPOS}/llm-compressor/tests/e2e/vLLM/configs/tinysmokeqwen3moe_w4a16_grouped_quant.yaml"
# pytest tests/e2e/vLLM/test_vllm.py -vs -rs 2>&1 | tee log-mini-int4.log

# export TEST_DATA_FILE="${REPOS}/llm-compressor/tests/e2e/vLLM/configs/qwen1.5-moe-grouped_quant.yaml"
# pytest tests/e2e/vLLM/test_vllm.py -vs -rs 2>&1 | tee log-mini-int4.log

# export TEST_DATA_FILE="${REPOS}/llm-compressor/tests/e2e/vLLM/configs/qwen3_w4a16_grouped_quant.yaml"
# export CADENCE="nightly"
# export SKIP_HF_UPLOAD="yes"
# pytest tests/e2e/vLLM/test_vllm.py -vs -rs 2>&1 | tee log-int4.log

# export TEST_DATA_FILE="${REPOS}/llm-compressor/tests/e2e/vLLM/configs/w4a16_channel_quant.yaml"
# pytest tests/e2e/vLLM/test_vllm.py -vs 2>&1 | tee log-int4.log


# test group quantization # DONE on branch
# CADENCE=weekly TEST_DATA_FILE="${REPOS}/llm-compressor/tests/lmeval/configs/w4a16_awq_sym.yaml" pytest -s -vs -rs "${REPOS}/llm-compressor/tests/lmeval/test_lmeval.py" 2>&1 | tee log_w4a16_awq_new_2.log

# test channel quantization
CADENCE=weekly TEST_DATA_FILE="${REPOS}/llm-compressor/tests/lmeval/configs/fp8_dynamic_test.yaml" pytest -s -vs -rs "${REPOS}/llm-compressor/tests/lmeval/test_lmeval.py" 2>&1 | tee log_fp8_dynamic.log

# test block
CADENCE=weekly TEST_DATA_FILE="${REPOS}/llm-compressor/tests/lmeval/configs/fp8_block_test.yaml" pytest -s -vs -rs "${REPOS}/llm-compressor/tests/lmeval/test_lmeval.py" 2>&1 | tee log_fp8_block.log

# test tensor quantization
# CADENCE=weekly TEST_DATA_FILE="${REPOS}/llm-compressor/tests/lmeval/configs/int8_tensor.yaml" pytest -s -vs -rs "${REPOS}/llm-compressor/tests/lmeval/test_lmeval.py" 2>&1 | tee log_int8_tensor.log