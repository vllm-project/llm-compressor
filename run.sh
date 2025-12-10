# test group quantization # on branch
# CADENCE=weekly TEST_DATA_FILE="${REPOS}/llm-compressor/tests/lmeval/configs/w4a16_awq_sym.yaml" pytest -s -vs -rs "${REPOS}/llm-compressor/tests/lmeval/test_lmeval.py" 2>&1 | tee log_w4a16_awq_new_2.log
# 2025-12-09T12:27:25.564645-0500 | _validate_recovery | INFO - ✓ exact_match,strict-match                 | Base: 0.7620 | Compressed: 0.7170 | Recovery: 94.09% ↑ | Threshold: ≥92.00%
# 2025-12-09T12:27:25.564995-0500 | _validate_recovery | INFO - ✓ exact_match,flexible-extract             | Base: 0.7600 | Compressed: 0.7130 | Recovery: 93.82% ↑ | Threshold: ≥93.00%
# on MAIN
# CADENCE=weekly TEST_DATA_FILE="${REPOS}/llm-compressor/tests/lmeval/configs/w4a16_awq_sym.yaml" pytest -s -vs -rs "${REPOS}/llm-compressor/tests/lmeval/test_lmeval.py" 2>&1 | tee log_w4a16_awq_main.log
# 2025-12-09T16:48:29.917537-0500 | _validate_recovery | INFO - ✓ exact_match,strict-match                 | Base: 0.7620 | Compressed: 0.7090 | Recovery: 93.04% ↑ | Threshold: ≥92.00%
# 2025-12-09T16:48:29.917879-0500 | _validate_recovery | ERROR - ✗ exact_match,flexible-extract             | Base: 0.7600 | Compressed: 0.7060 | Recovery: 92.89% ↑ | Threshold: ≥93.00%

# test channel quantization # on branch
# CADENCE=weekly TEST_DATA_FILE="${REPOS}/llm-compressor/tests/lmeval/configs/fp8_dynamic_test.yaml" pytest -s -vs -rs "${REPOS}/llm-compressor/tests/lmeval/test_lmeval.py" 2>&1 | tee log_fp8_dynamic.log
# 2025-12-09T23:08:54.484729+0000 | _validate_recovery | INFO - ✓ exact_match,strict-match                 | Base: 0.7650 | Compressed: 0.7610 | Recovery: 99.48% ↑ | Threshold: ≥95.00%
# 2025-12-09T23:08:54.485457+0000 | _validate_recovery | INFO - ✓ exact_match,flexible-extract             | Base: 0.7630 | Compressed: 0.7580 | Recovery: 99.34% ↑ | Threshold: ≥95.00%
# TODO on main

# test block quantization # on branch
# CADENCE=weekly TEST_DATA_FILE="${REPOS}/llm-compressor/tests/lmeval/configs/fp8_block_test.yaml" pytest -s -vs -rs "${REPOS}/llm-compressor/tests/lmeval/test_lmeval.py" 2>&1 | tee log_fp8_block.log
# 2025-12-10T01:30:30.381290+0000 | _validate_recovery | INFO - ✓ exact_match,strict-match                 | Base: 0.7650 | Compressed: 0.7720 | Recovery: 100.92% ↑ | Threshold: ≥95.00%
# 2025-12-10T01:30:30.381963+0000 | _validate_recovery | INFO - ✓ exact_match,flexible-extract             | Base: 0.7630 | Compressed: 0.7690 | Recovery: 100.79% ↑ | Threshold: ≥95.00%
# TODO on main

# test tensor quantization # DONE on branch
# CADENCE=weekly TEST_DATA_FILE="${REPOS}/llm-compressor/tests/lmeval/configs/int8_tensor.yaml" pytest -s -vs -rs "${REPOS}/llm-compressor/tests/lmeval/test_lmeval.py" 2>&1 | tee log_int8_tensor.log
# 2025-12-09T15:28:21.949902-0500 | _validate_recovery | ERROR - ✗ exact_match,strict-match                 | Base: 0.7620 | Compressed: 0.7220 | Recovery: 94.75% ↑ | Threshold: ≥95.00%
# 2025-12-09T15:28:21.950059-0500 | _validate_recovery | INFO - ✓ exact_match,flexible-extract             | Base: 0.7600 | Compressed: 0.7240 | Recovery: 95.26% ↑ | Threshold: ≥95.00%
# on MAIN
# CADENCE=weekly TEST_DATA_FILE="${REPOS}/llm-compressor/tests/lmeval/configs/int8_tensor.yaml" pytest -s -vs -rs "${REPOS}/llm-compressor/tests/lmeval/test_lmeval.py" 2>&1 | tee log_int8_tensor_main.log
# 2025-12-09T17:33:51.182178-0500 | _validate_recovery | INFO - ✓ exact_match,strict-match                 | Base: 0.7620 | Compressed: 0.7280 | Recovery: 95.54% ↑ | Threshold: ≥95.00%
# 2025-12-09T17:33:51.182311-0500 | _validate_recovery | INFO - ✓ exact_match,flexible-extract             | Base: 0.7600 | Compressed: 0.7310 | Recovery: 96.18% ↑ | Threshold: ≥95.00%