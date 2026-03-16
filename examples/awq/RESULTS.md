# AWQ + FP8 Quantization Results

**Model:** Meta-Llama-3-8B-Instruct
**Hardware:** 8x NVIDIA A100-SXM4-80GB
**Date:** Feb 10, 2026

## Summary

Ran the example scripts with both FP8 schemes (FP8_DYNAMIC and FP8_BLOCK) on Meta-Llama-3-8B-Instruct, then evaluated on GSM8K.

This PR adds `RESULTS.md` with reproducible workflow for evaluating AWQ+FP8 quantization schemes on GSM8K.

## GSM8K Results

| Scheme | Strict Match | Flexible Extract |
|--------|-------------|------------------|
| **FP8_DYNAMIC** | 76.42% | 76.19% |
| **FP8_BLOCK** | 75.21% | 74.98% |

**Evaluation details:**
- 1,319 test samples
- Batch size: 16
- Model: Meta-Llama-3-8B-Instruct

## Discussion

This behavior where FP8_BLOCK underperforms FP8_DYNAMIC contradicts our expectation since for RTN FP8_BLOCK outperforms FP8_DYNAMIC, however there are 2 important things to notice.
1) FP8_BLOCK quantization creates quantization `groups` whose size is equivalent to the number of elements in a block, whereas FP8_DYNAMIC quantization creates quantization `groups`
   whose size is equal to the in_features. Thus as long as in_features is less than the block size (128x128=16384) the number of weight scales will actually be higher for per channel quantization.
   For Meta-Llama-3-8B-Instruct the per-channel weight quantization of the FP8_DYNAMIC scheme has more scales than FP8_BLOCK for every weight.
2) Its also noteworthy that for AWQ, the scale factors being searched for during AWQ align directly with the quantization scales of the per channel weight quantization, this is likely why AWQ yields
   such a large improvement for FP8_DYNAMIC
   
## Model Checkpoints

- FP8_DYNAMIC: https://huggingface.co/nm-testing/Meta-Llama-3-8B-Instruct-awq-asym-fp8-dynamic
- FP8_BLOCK: https://huggingface.co/nm-testing/Meta-Llama-3-8B-Instruct-awq-asym-fp8-block

## Setup

Use the existing example scripts from the repo:
```bash
cd examples/awq
python fp8_dynamic_llama_example.py
python fp8_block_llama_example.py
```

## Evaluation

Run GSM8K evaluation using lm-eval:

```bash
lm_eval \
  --model vllm \
  --model_args pretrained=<model_path>,dtype=auto \
  --tasks gsm8k \
  --batch_size 16 \
  --output_path <output_dir>
```

**Important:** Setting `batch_size=16` is critical. The default `auto` picks 1, which significantly increases evaluation time.
