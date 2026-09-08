# Layerwise Decompression & Recalibration for Kimi-K3

This document explains the layerwise decompression approach and how to use it with Kimi-K3.

## Problem

When working with pre-compressed large language models (like the pre-quantized Kimi-K3 checkpoint), you often need to **recalibrate** the model — re-running quantization with different settings or algorithms.

The naive approach:
1. Load the compressed model
2. **Fully decompress** the entire model into FP32
3. Recalibrate on fresh calibration data
4. Compress again

**Issue**: For 100B+ parameter models, decompressing everything at once causes **out-of-memory (OOM)** errors, even with sophisticated offloading.

## Solution: Layerwise Decompression

Instead of decompressing the entire model at once, process it **layer-by-layer**:

```
For each layer in the model (sequentially):
  1. Decompress only that layer → FP32
  2. Re-apply quantization config & observers (scoped to just this layer)
  3. Run calibration batches through this layer
  4. Run error propagation (adjust next layer's inputs for better accuracy)
  5. Recompress that layer → original format
  6. Move to next layer
  
  → Peak memory = just 1-2 layers at FP32, not the whole model
```

**Benefits:**
- ✅ Fits in GPU memory (one layer at a time)
- ✅ Maintains model accuracy via error propagation
- ✅ Stays consistent with transformers' cached formats
- ✅ No full-model decompression needed

## Current Implementation

### Sequential Pipeline (Available Now)

The **sequential pipeline** is the current mechanism for per-layer processing:

```python
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    pipeline="sequential",
    sequential_targets=["KimiK3DecoderLayer"],
    batch_size=1,
    propagate_error=True,  # Adjust next layer for better accuracy
)
```

This processes one `KimiK3DecoderLayer` at a time, keeping only that layer decompressed in GPU memory.

### Layerwise Flags (Coming Soon)

When `layerwise_decompression` and `layerwise_compression` flags are available on your branch:

```python
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    pipeline="sequential",
    sequential_targets=["KimiK3DecoderLayer"],
    layerwise_decompression=True,   # Decompress each layer before calibration
    layerwise_compression=True,     # Recompress after calibration
    propagate_error=True,
)
```

These flags enable per-layer decompress→calibrate→compress cycles for even more explicit control.

## Kimi-K3 Specifics

Kimi-K3 is a **multimodal LLM** with special structures:

### Architecture
- **Vision tower**: Handles image inputs (ignored from quantization)
- **Sparse MoE**: Mixture of Experts routing with gated experts
- **Residual projections**: Attention and MLP residual connections

### Quantization Handling
```python
qconfig.ignore += [
    "re:.*mlp_res_proj.*",              # MLP residuals
    "re:.*self_attention_res_proj.*",   # Attention residuals
    "re:.*routed_expert.*",             # Expert routing
    "re:.*output_attn_res_proj.*",      # Output attention residuals
]
```

These are left in FP32 to:
- Maintain numerical stability in residual paths
- Avoid compounding quantization errors in expert routing
- Preserve accuracy in critical attention projections

### Examples

**Sequential Pipeline (current):**
```bash
python examples/quantizing_moe/kimi_k3_layerwise_example.py
```

This uses the sequential pipeline to process layers one at a time.

**Future: Full Layerwise (when flags merge):**
```bash
python examples/quantizing_moe/kimi_k3_layerwise_full_example.py
```

Once `layerwise_decompression` and `layerwise_compression` flags are available.

## How Layerwise Differs from Standard Sequential

| Aspect | Sequential Pipeline | Full Layerwise |
|--------|-------------------|-----------------|
| **Decompress before cal.** | Layer stays in compressed format | Explicitly decompress each layer |
| **Calibration scope** | Applies to whole layer at once | Re-applies config per layer |
| **Memory overhead** | Minimal (depends on transformer format) | Very explicit tracking |
| **Error propagation** | Yes (adjusted per layer) | Yes (with clear per-layer scope) |
| **Recompression** | Implicit during save | Explicit after each layer |

## Testing Layerwise Decompression

### 1. Start with a pre-compressed model
The Kimi-K3 checkpoint ships pre-quantized, making it ideal for testing layerwise recalibration:
```python
model = KimiK3ForConditionalGeneration.from_pretrained(
    "moonshotai/Kimi-K3",
    quantization_config=qconfig,
    device_map="auto",
)
```

### 2. Run calibration with sequential pipeline
```python
oneshot(
    model=model,
    dataset=calibration_ds,
    recipe=recipe,
    pipeline="sequential",
    sequential_targets=["KimiK3DecoderLayer"],
    propagate_error=True,
)
```

### 3. Verify memory usage
Monitor GPU memory throughout — peak should be just a few GB for one layer, not 100GB+ for the full model.

### 4. Check recalibration quality
- Compare output logits before/after recalibration
- Validate perplexity on a held-out dataset
- Check that numerical stability is maintained in residual paths

## Debugging Tips

### If you get OOM during sequential pipeline:
1. **Reduce batch size** (already at 1 in examples)
2. **Enable offloading** with `sequential_offload_device="cpu"` or `"cuda:1"`
3. **Reduce sequence length** temporarily for testing
4. **Check sequential_targets** — make sure you're targeting the actual layer class name

### If quantization fails on specific layers:
1. Check the ignore patterns — residual projections should be skipped
2. Verify MoE experts are being handled via `moe_calibrate_all_experts=True`
3. Run with smaller calibration set first (e.g., 16 samples) to debug quickly

### Memory profiling:
```python
import torch
from torch.cuda import memory_allocated

for i, layer in enumerate(model.model.layers):
    print(f"Layer {i}: {memory_allocated() / 1e9:.1f} GB")
    # Process layer...
```

## References

- **PR**: https://github.com/vllm-project/llm-compressor/pull/2994
- **Layerwise Commit**: `7e0a204cd`
- **Kimi-K3 Model**: https://huggingface.co/moonshotai/Kimi-K3
- **Compressed-Tensors PR**: https://github.com/vllm-project/compressed-tensors/pull/811

## Next Steps

Once you have the layerwise flags available:
1. Migrate examples to use `layerwise_decompression=True` and `layerwise_compression=True`
2. Run performance benchmarks comparing sequential vs. full layerwise approaches
3. Document final memory footprint and calibration time
4. Add tests for pre-compressed model recalibration workflows
