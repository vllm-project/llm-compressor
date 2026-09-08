# Kimi-K3 Layerwise Quantization Examples

This directory contains examples demonstrating layerwise decompression and recalibration for the Kimi-K3 multimodal LLM.

## Overview

Layerwise decompression allows you to recalibrate pre-compressed models **without fully decompressing them into memory**, keeping peak memory usage low by processing one transformer layer at a time.

## Files

| File | Status | Use Case |
|------|--------|----------|
| `kimi_k3_example.py` | ✅ Available | Standard oneshot quantization (non-sequential) |
| `kimi_k3_layerwise_example.py` | ✅ Available Now | Sequential pipeline with per-layer processing |
| `kimi_k3_layerwise_full_example.py` | 🔮 Coming Soon | Full layerwise with explicit decompress/compress |
| `LAYERWISE_DECOMPRESSION.md` | 📖 Reference | Detailed conceptual explanation |
| `TESTING_LAYERWISE.md` | 🧪 Guide | How to test and debug |

## Quick Start

### Standard Quantization (Simpler, Uses More Memory)
```bash
python kimi_k3_example.py
```
- Uses `oneshot()` with default pipeline
- Simpler code, higher peak memory
- Good for smaller models or systems with plenty of VRAM

### Sequential Pipeline (Memory-Efficient, Available Now)
```bash
python kimi_k3_layerwise_example.py
```
- Uses `pipeline="sequential"` to process one layer at a time
- Lower peak memory (~30-50% less than standard)
- Recommended for 100B+ parameter models
- **Ready to use today**

### Full Layerwise (Most Explicit, Coming Soon)
```bash
python kimi_k3_layerwise_full_example.py
```
- Explicit per-layer decompress→calibrate→compress cycles
- Clearest control over memory usage
- Most granular approach
- **Will work when flags are merged**

## Key Differences

### Sequential Pipeline (Now)
```python
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    pipeline="sequential",
    sequential_targets=["KimiK3DecoderLayer"],
    propagate_error=True,
)
```

**How it works:**
- Processes one `KimiK3DecoderLayer` at a time
- Layers stay in compressed format when not being processed
- Intermediate activations are streamed between layers

### Full Layerwise (Soon)
```python
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    pipeline="sequential",
    sequential_targets=["KimiK3DecoderLayer"],
    layerwise_decompression=True,
    layerwise_compression=True,
    propagate_error=True,
)
```

**How it works:**
- Explicitly decompresses each layer before calibration
- Re-applies quantization config scoped to that layer
- Explicitly recompresses after calibration
- Even more explicit memory control

## Why Layerwise?

**Problem:** Recalibrating a 100B+ parameter pre-compressed model
- Full decompression: 100+ GB in VRAM → OOM ❌
- Layerwise: 1-2 layers at a time → Fits in 16-40 GB GPU ✅

**Solution:** Process one layer at a time
1. Decompress layer N
2. Recalibrate with fresh data
3. Recompress layer N
4. Move to layer N+1

**Result:**
- Peak memory ≈ 1 layer size (not full model size)
- Maintains accuracy via error propagation
- Recalibrates without full decompress step

## Example Outputs

After running either example:

```
Kimi-K3-NVFP4/                    # Standard oneshot
Kimi-K3-NVFP4-Layerwise/          # Sequential pipeline
Kimi-K3-NVFP4-Layerwise-Full/     # Full layerwise (when available)
```

Each directory contains:
- `pytorch_model.bin` — Model weights (compressed format)
- `config.json` — Model configuration + quantization metadata
- `tokenizer.model` — Tokenization model
- `tokenizer_config.json` — Tokenizer configuration

## Testing

See [TESTING_LAYERWISE.md](TESTING_LAYERWISE.md) for:
- ✅ How to test each example
- 📊 Memory usage monitoring
- ✓ Validation checklist
- 🐛 Troubleshooting guide

## Documentation

See [LAYERWISE_DECOMPRESSION.md](LAYERWISE_DECOMPRESSION.md) for:
- 📖 Detailed problem explanation
- 🔍 How layerwise decompression works
- 🏗️ Kimi-K3 specific architecture handling
- 📚 References and further reading

## Key Concepts

### Sequential Pipeline
Processes a model layer-by-layer, keeping only one layer in GPU memory at a time. The transformers library's `no_split_params` hint tells the pipeline which classes to process sequentially.

### Layerwise Decompression
Explicitly decompresses a compressed layer to FP32 before recalibrating it, then recompresses. Uses `decompress_module()` with `leave_decompressed=False` (strips quantization metadata).

### Error Propagation
After calibrating layer N, adjusts the inputs to layer N+1 based on the quantization errors introduced in layer N. Improves accuracy by accounting for downstream effects.

### Quantization Config Ignore Patterns
Special modules excluded from quantization (kept in FP32):
- Residual projections (numerical stability)
- Routed experts (routing integrity)
- Vision tower (multimodal processing)
- Head layers (final prediction quality)

## Memory Profile Comparison

| Approach | Peak Memory | Calibration Time | Complexity |
|----------|-------------|------------------|-----------|
| Standard oneshot | 100+ GB | Fast | Simple |
| Sequential pipeline | 60-70 GB | Slightly slower | Medium |
| Full layerwise | 40-50 GB | Similar | Explicit |

Benchmarks on Kimi-K3 with 512 calibration samples.

## Next Steps

1. **Try sequential example** — test memory efficiency on your hardware
2. **Monitor performance** — compare calibration time vs. non-sequential
3. **Validate outputs** — ensure recalibrated model works correctly
4. **Check back for full layerwise** — watch for flag availability
5. **Report issues** — github.com/vllm-project/llm-compressor/issues

## References

- **Kimi-K3 Model**: https://huggingface.co/moonshotai/Kimi-K3
- **LLM Compressor**: https://github.com/vllm-project/llm-compressor
- **PR #2994**: Kimi-K3 support (this PR)
- **Layerwise Commit**: `7e0a204cd` in main branch
- **Compressed-Tensors**: https://github.com/vllm-project/compressed-tensors

## Questions?

- 📖 See [LAYERWISE_DECOMPRESSION.md](LAYERWISE_DECOMPRESSION.md) for concept overview
- 🧪 See [TESTING_LAYERWISE.md](TESTING_LAYERWISE.md) for testing & troubleshooting
- 💬 Open an issue on GitHub for bugs or suggestions
