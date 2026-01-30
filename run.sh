# Quantize
python tests/lmeval/quantize_for_lmeval.py --config tests/lmeval/configs/w4a16_awq_sym.yaml 2>&1 | tee quantize.log

# Evaluate
python tests/lmeval/eval_with_vllm.py --config tests/lmeval/configs/w4a16_awq_sym.yaml --compare-baseline 2>&1 | tee vllm.log

