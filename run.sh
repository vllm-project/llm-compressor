CADENCE=weekly TEST_DATA_FILE=$REPOS/llm-compressor/tests/lmeval/configs/w4a4_nvfp4.yaml pytest -vs -rs $REPOS/llm-compressor/tests/lmeval/test_lmeval.py 2>&1 | tee log_nvfp4_base.log
CADENCE=weekly TEST_DATA_FILE=$REPOS/llm-compressor/tests/lmeval/configs/awq_nvfp4.yaml pytest -vs -rs $REPOS/llm-compressor/tests/lmeval/test_lmeval.py 2>&1 | tee log_nvfp4.log
CADENCE=weekly TEST_DATA_FILE=$REPOS/llm-compressor/tests/lmeval/configs/awq_nvfp4a16.yaml pytest -vs -rs $REPOS/llm-compressor/tests/lmeval/test_lmeval.py 2>&1 | tee log_nvfp4a16.log


