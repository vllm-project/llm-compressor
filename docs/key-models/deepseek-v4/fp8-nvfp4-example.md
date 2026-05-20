## DeepSeekV4 ##

## Quantizing DeepSeekV4 ##
DeepSeekV4 is currently supported experimentally in LLM Compressor. You can quantize the model using the following steps:

1. Install LLM Compressor from scratch and check out the DeepSeekV4 experimental branch
```bash
git clone https://github.com/vllm-project/llm-compressor
cd llm-compressor
git checkout deepseekv4-experimental
uv pip install -e .
uv pip install -U transformers==5.8.0
```

2. Run the example. You can replace the MODEL_ID with any of `inference-optimization/DSV4-tiny-empty` or `RedHatAI/DeepSeek-V4-Flash-BF16`
```bash
python3 examples/quantizing_moe/deepseek_v4_example.py
```

3. Convert the model. This is to account for issues with the DSV4 transformers model definition which may be fixed at a future date
```bash
python3 fix_dsv4_structure.py DeepSeek-V4-Flash-BF16-NVFP4-FP8-BLOCK DeepSeek-V4-Flash-NVFP4-FP8
```

4. Install vllm with the experimental branch. You may want to rebase on main
```bash
git clone https://github.com/neuralmagic/vllm
cd vllm
git checkout kylesayrs/deepseek-ct
uv pip install -U transformers==5.8.0
# optional:
# git rebase main
VLLM_USE_PRECOMPILED=1 uv pip install -e .
```

5. Serve with vLLM
```bash
vllm serve DeepSeek-V4-Flash-NVFP4-FP8 --tensor-parallel-size 4 --kv_cache_dtype="fp8"
```
