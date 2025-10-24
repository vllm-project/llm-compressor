export TEST_DATA_FILE="${REPOS}/llm-compressor/tests/e2e/vLLM/configs/fp8_dynamic_per_tensor_moe.yaml" # working
pytest tests/e2e/vLLM/test_vllm.py -vs 2>&1 | tee log-fp8.log

export TEST_DATA_FILE="${REPOS}/llm-compressor/tests/e2e/vLLM/configs/w4a16_channel_quant_moe.yaml" # working
pytest tests/e2e/vLLM/test_vllm.py -vs 2>&1 | tee log-int4.log

export TEST_DATA_FILE="${REPOS}/llm-compressor/tests/e2e/vLLM/configs/fp4_nvfp4_moe.yaml"
pytest tests/e2e/vLLM/test_vllm.py -vs 2>&1 | tee log-fp4.log

export TEST_DATA_FILE="${REPOS}/llm-compressor/tests/e2e/vLLM/configs/fp4_nvfp4.yaml"
pytest tests/e2e/vLLM/test_vllm.py -vs 2>&1 | tee log-fp-base.log

