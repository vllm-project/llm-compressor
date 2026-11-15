---
title: "GPTQ W4A16 on Qwen2.5 Series with RTX 3050 Ti 4GB"
tags:
  - GPTQ
  - W4A16
  - Qwen2.5
---

<div class="entry-byline" markdown>
[![adj116](https://github.com/adj116.png?size=64){ .entry-avatar }](https://github.com/adj116)
**[@adj116](https://github.com/adj116)** posted on **2026-11-16**
</div>

# GPTQ W4A16 on Qwen2.5 Series with RTX 3050 Ti 4GB

## Summary

Quantized Qwen2.5-0.5B and Qwen2.5-1.5B to W4A16 using GPTQ on a consumer laptop GPU (RTX 3050 Ti 4GB). Tested different calibration sample counts (64 vs 512) and model sizes to find the practical limits of 4GB VRAM hardware. Results show that calibration sample count does not affect model size or inference speed, and 1.5B models can technically fit in 4GB VRAM but leave insufficient headroom for practical use.

## Setup

- **Models:** `Qwen/Qwen2.5-0.5B-Instruct`, `Qwen/Qwen2.5-1.5B-Instruct`
- **Scheme:** W4A16 (4-bit weights, 16-bit activations)
- **Algorithm:** GPTQ (defaults: `block_size=128`, `dampening_frac=0.01`, `actorder="static"`)
- **Calibration:** `open_platypus` dataset, max_seq_length=512
- **Hardware:** 
  - GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU (4 GB VRAM)
  - Driver: 592.00
  - PyTorch: 2.10.0+cu126
  - CUDA: 12.6
  - Python: 3.11.9
- **Software:**
  - llm-compressor: 0.15.0
  - transformers: 5.21.0
  - compressed-tensors: 0.20.1

### Quantization Script

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"  # or 1.5B
OUTPUT_DIR = "./qwen-w4a16-gptq"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

recipe = GPTQModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=["lm_head"]
)

oneshot(
    model=model,
    dataset="open_platypus",
    recipe=recipe,
    max_seq_length=512,
    num_calibration_samples=64,  # or 512
)

model.save_pretrained(OUTPUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUTPUT_DIR)
```

## Results

### Experiment 1: Calibration Samples Comparison (0.5B Model)

| Metric | 64 samples | 512 samples | Difference |
|--------|-----------|-------------|------------|
| Model size | 446.72 MB | 446.72 MB | 0% |
| Model VRAM | 0.457 GB | 0.467 GB | +2.1% |
| Quantization time | ~2 min | 6 min | +200% |

**Finding:** Both runs produced identical model sizes (446.72 MB) and similar VRAM usage (~0.46 GB), confirming that calibration sample count does not affect the quantized model footprint.

**Speed measurement variance:** Initial tests showed a 15% speed difference (13.69 vs 15.75 tok/s), but follow-up validation with 30 runs per model (10 runs × 3 prompts) revealed:
- 64-sample: 11.85 ± 0.64 tok/s (CV: 5.4%)
- 512-sample: 10.38 ± 0.80 tok/s (CV: 7.7%)
- Difference: -12.4% (512-sample slower in this session)

The fact that the difference reversed direction (+15% initially vs -12.4% in validation) indicates inference speed varies across sessions due to thermal throttling, background processes, or other system factors. Calibration sample count does not have a measurable impact on inference speed.

### Experiment 2: Model Size Limits (1.5B Model)

| Metric | 0.5B (64 samples) | 1.5B (64 samples) |
|--------|-------------------|-------------------|
| Model size | 446.72 MB | 1100.65 MB |
| Model VRAM | 0.457 GB | 3.12 GB* |
| Average speed | 11.85 tokens/sec | 2.88 tokens/sec |
| Quantization time | ~2 min | 26.6 min |

*VRAM measured with `torch.cuda.memory_allocated()` after initial model quantization and generation (not re-measured in the isolated precision script). For 1.5B, this approaches the 4GB limit, leaving minimal headroom for KV cache and other GPU operations.

**Finding:** 1.5B model fits in 4GB VRAM but runs at 2.88 tokens/sec (4.1x slower than 0.5B). The model alone consumes ~78% of available VRAM, making it impractical for real-world use cases that require additional memory for KV cache or batch processing.

### Generation Quality

All experiments produced coherent outputs:

**0.5B (64 samples, avg 11.85 tok/s):**
- "Hello, my name is John and I am a software developer..."
- "The capital of France is Paris. It is the largest city in Europe..."
- "def fibonacci(n): a, b = 0, 1; while a < n: print(a)..."

**1.5B (64 samples, avg 2.88 tok/s):**
- "Hello, my name is John and I am a 16 year old boy..."
- "The capital of France is the city that has been the seat of power..."
- "def fibonacci(n): if n == 0: return 0; elif n == 1: re..."

## Observations

### What worked well

1. **4GB VRAM is viable for small models**: 0.5B model runs comfortably with headroom for KV cache
2. **Compression ratio consistent**: ~2.3x across all experiments regardless of calibration samples
3. **Quality preserved**: All models generate coherent text
4. **Calibration time trade-off understood**: 512 samples takes 3x longer but does not improve inference speed

### What to investigate next

1. **3B model**: Test if it fits in 4GB VRAM (unlikely given 1.5B uses 3.12 GB)
2. **Different schemes**: Compare GPTQ with AWQ, SmoothQuant, or FP8
3. **Quality benchmarks**: Run perplexity or task-specific benchmarks to see if 512 samples improve quantization quality even if speed is unchanged
4. **vLLM serving**: Test with vLLM for production-grade serving
5. **Speed variance factors**: Investigate what causes inference speed to vary across sessions (thermal throttling, background processes, etc.)
6. **Larger calibration sets**: Test 1024 or 2048 samples to see if quality improves

### Notes for beginners

1. **PyTorch CUDA**: Install with CUDA support:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
   ```
2. **Python version**: llm-compressor 0.15.0 requires Python 3.11+
3. **Verify CUDA**: Check `torch.cuda.is_available()` returns `True`
4. **lm_head**: Always ignore `lm_head` layer to preserve output quality
5. **Import path**: Use `from llmcompressor.modifiers.gptq import GPTQModifier`
6. **VRAM monitoring**: Use `nvidia-smi` to watch memory usage during quantization

### Limitations of these experiments

1. **No quality benchmarks**: Only qualitative generation test
2. **Single dataset**: Only tested on `open_platypus`
3. **No vLLM serving**: Tested with transformers, not vLLM
4. **Hardware constraints**: RTX 3050 Ti is entry-level; results may differ on better GPUs
5. **Speed measurement variance**: Inference speed varies across sessions (CV: 5-8%), making it difficult to detect small differences. Calibration sample count shows no clear impact on inference speed.
6. **VRAM measurement**: Used `torch.cuda.memory_allocated()` which reflects PyTorch's perspective; for total system VRAM usage, use `nvidia-smi`
7. **Small test set**: Speed measured on only 3 prompts × 50 tokens per run; larger test sets would give more reliable averages
8. **Session variance dominates**: Speed differences reversed direction across measurement sessions (+15% in initial run, -12.4% in validation run), indicating environmental factors (thermals, OS scheduling) have a far greater impact than calibration sample count.
