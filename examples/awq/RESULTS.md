# AWQ + FP8 Quantization Results

**Model:** Meta-Llama-3-8B-Instruct
**Hardware:** 8x NVIDIA A100-SXM4-80GB
**Date:** Feb 10, 2026

## Summary

Ran the example scripts with both FP8 schemes (FP8_DYNAMIC and FP8_BLOCK) on Meta-Llama-3-8B-Instruct, then evaluated on GSM8K as requested in #2305. FP8_DYNAMIC performs better overall.

This PR adds `RESULTS.md` with reproducible workflow for evaluating AWQ+FP8 quantization schemes on GSM8K.

## GSM8K Results

| Scheme | Strict Match | Flexible Extract |
|--------|-------------|------------------|
| **FP8_DYNAMIC** | **76.42%** | **76.19%** |
| **FP8_BLOCK** | 75.21% | 74.98% |

FP8_DYNAMIC wins by ~1.2% on strict matching. Both achieve similar performance on flexible extraction.

**Evaluation details:**
- 1,319 test samples
- Batch size: 16
- Model: Meta-Llama-3-8B-Instruct

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
  --model hf \
  --model_args pretrained=<model_path>,dtype=auto \
  --tasks gsm8k \
  --batch_size 16 \
  --output_path <output_dir>
```

**Important:** Setting `batch_size=16` is critical. The default `auto` picks 1, which significantly increases evaluation time.

## Recommendation

**Use FP8_DYNAMIC** for AWQ quantization - better accuracy preservation (76.42% vs 75.21% on GSM8K strict matching) with similar model characteristics.
