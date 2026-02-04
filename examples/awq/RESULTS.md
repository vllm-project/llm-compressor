# AWQ + FP8 Quantization Results

Closes #2305

**Model:** Qwen2.5-0.5B-Instruct
**Hardware:** Google Colab L4 GPU (22.5GB)
**Date:** Feb 4, 2026

## Summary

Ran the example scripts with both FP8 schemes (FP8_DYNAMIC and FP8_BLOCK) on Qwen2.5-0.5B-Instruct, then evaluated on GSM8K as requested in #2305. FP8_DYNAMIC performs better overall.

This PR adds:
- `gsm8k_eval.py` - evaluation script for running GSM8K benchmarks
- `RESULTS.md` - results and reproducible workflow

## Quantization Results

Both schemes compressed the model from ~1.1GB to 0.92GB (~1.2x compression):

| Scheme | Size | Files |
|--------|------|-------|
| FP8_DYNAMIC | 0.92 GB | 11 |
| FP8_BLOCK | 0.92 GB | 11 |

Runtime: ~4-5 minutes each on L4 GPU

## GSM8K Results

| Model | Strict Match | Flexible Extract |
|-------|-------------|------------------|
| **FP8_DYNAMIC** | **22.67%** | **30.78%** |
| **FP8_BLOCK** | 17.97% | 29.80% |

FP8_DYNAMIC wins by ~5% on strict matching. Both are comparable on flexible extraction.

**Evaluation details:**
- 1,319 test samples
- Batch size: 16
- Runtime: ~70-82 min per model on L4

## Models on HuggingFace

- FP8_DYNAMIC: https://huggingface.co/rtj1/Qwen2.5-0.5B-AWQ-FP8-Dynamic
- FP8_BLOCK: https://huggingface.co/rtj1/Qwen2.5-0.5B-AWQ-FP8-Block

## Setup

Used the existing example scripts from the repo:
```bash
cd examples/awq
sed -i 's/meta-llama\/Meta-Llama-3-8B-Instruct/Qwen\/Qwen2.5-0.5B-Instruct/g' *.py
python fp8_dynamic_llama_example.py
python fp8_block_llama_example.py
```

Switched to Qwen2.5-0.5B since it's not gated and quantizes faster than the 8B models.

## Evaluation

Created `gsm8k_eval.py` for running benchmarks:

```bash
lm_eval \
  --model hf \
  --model_args pretrained=<model_path>,dtype=auto \
  --tasks gsm8k \
  --batch_size 16 \
  --output_path <output_dir>
```

**Important:** Setting `batch_size=16` is critical. The default `auto` picks 1, which makes evaluation take 10+ hours instead of ~70 minutes.

## Reproducing

Full workflow takes ~2.7 hours on L4 GPU:
1. Quantize FP8_DYNAMIC (~5 min)
2. Quantize FP8_BLOCK (~4 min)
3. Eval FP8_DYNAMIC (~71 min)
4. Eval FP8_BLOCK (~82 min)

## Notes

- Qwen2.5-0.5B getting 22-30% on GSM8K is reasonable for a 0.5B model 
- L4 GPU (22.5GB) works fine, but A100 would be faster
- Using non-gated models makes this easier to reproduce

## Recommendation

**Use FP8_DYNAMIC** for AWQ quantization - better accuracy preservation (22.67% vs 17.97% on strict matching) with the same compression ratio and runtime.
