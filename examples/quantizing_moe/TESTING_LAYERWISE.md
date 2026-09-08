# Testing Layerwise Examples for Kimi-K3

This guide walks you through testing the layerwise decompression examples created for Kimi-K3.

## Files Created

1. **`kimi_k3_layerwise_example.py`** — Current implementation using sequential pipeline
   - Uses the available sequential pipeline to process one layer at a time
   - Works on your current branch
   - Demonstrates memory-efficient recalibration

2. **`kimi_k3_layerwise_full_example.py`** — Future implementation with explicit layerwise flags
   - Template for when `layerwise_decompression` and `layerwise_compression` are merged
   - Shows the most explicit per-layer decompress→calibrate→compress pattern
   - Will work once the flags are available

3. **`LAYERWISE_DECOMPRESSION.md`** — Conceptual documentation
   - Explains the problem layerwise decompression solves
   - Documents how it works
   - Shows the difference between sequential pipeline and full layerwise approaches

## Quick Start: Test Sequential Pipeline Version

The sequential pipeline example is ready to use now:

```bash
cd ~/repos/llm-compressor
python examples/quantizing_moe/kimi_k3_layerwise_example.py
```

### What This Does

1. **Loads** pre-compressed Kimi-K3 from HuggingFace
2. **Calibrates** using 512 samples from UltraChat dataset
3. **Processes layers sequentially** — one `KimiK3DecoderLayer` at a time
4. **Saves** recalibrated model to `Kimi-K3-NVFP4-Layerwise/`

### Expected Output

```
Downloading weights from moonshotai/Kimi-K3...
Loading dataset HuggingFaceH4/ultrachat_200k...
[==================================================] 100%

Calibrating with sequential pipeline...
[==================================================] 100%

✓ Model saved to Kimi-K3-NVFP4-Layerwise
  Sequential pipeline recalibrated the model layer-by-layer,
  keeping memory usage low throughout the process.
```

### Key Parameters

- `pipeline="sequential"` — Activates per-layer processing
- `sequential_targets=["KimiK3DecoderLayer"]` — Processes one decoder layer at a time
- `propagate_error=True` — Adjusts next layer's inputs for better accuracy
- `batch_size=1` — Minimal memory footprint per layer

## Testing: Memory Usage

Monitor GPU memory during calibration:

```bash
# Terminal 1: Run the example
python examples/quantizing_moe/kimi_k3_layerwise_example.py

# Terminal 2: Monitor GPU memory
watch -n 1 nvidia-smi
```

**Expected memory profile:**
- Starts: ~40-50 GB (full model loaded)
- During calibration: ~60-70 GB peak (current layer decompressed + buffers)
- NOT ~100+ GB full decompression

## Testing: Model Quality

After calibration, verify the recalibrated model still works correctly:

```python
from transformers import AutoTokenizer
from llmcompressor.modeling.kimi_k3 import KimiK3ForConditionalGeneration

model = KimiK3ForConditionalGeneration.from_pretrained(
    "./Kimi-K3-NVFP4-Layerwise",
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    "./Kimi-K3-NVFP4-Layerwise",
    trust_remote_code=True,
)

# Generate some text
messages = [
    {"role": "user", "content": "What is machine learning?"}
]
text = tokenizer.apply_chat_template(messages, tokenize=False)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Testing: Compare Sequential vs Non-Sequential

Compare calibration time and memory between sequential and non-sequential approaches:

```bash
# Sequential (memory-efficient, current)
time python examples/quantizing_moe/kimi_k3_layerwise_example.py

# Non-sequential (simpler, uses more memory)
time python examples/quantizing_moe/kimi_k3_example.py
```

**Expected results:**
- Sequential may be slightly slower (more overhead)
- Sequential uses ~30-50% less peak GPU memory
- Both should produce similar accuracy

## Future: Test Full Layerwise (Coming Soon)

Once `layerwise_decompression` and `layerwise_compression` flags are merged:

```bash
python examples/quantizing_moe/kimi_k3_layerwise_full_example.py
```

This will add even more explicit per-layer control:
- Each layer is **explicitly decompressed** before calibration
- Quantization config is **re-applied** scoped to that layer
- Layer is **explicitly compressed** after calibration
- Even more granular memory control

## Troubleshooting

### OOM During Calibration

If you hit out-of-memory errors:

1. **Reduce calibration samples** (test with 64 instead of 512):
   ```python
   NUM_CALIBRATION_SAMPLES = 64
   ```

2. **Reduce sequence length** (test with 512 instead of 2048):
   ```python
   MAX_SEQUENCE_LENGTH = 512
   ```

3. **Enable CPU offloading** (add to oneshot call):
   ```python
   sequential_offload_device="cpu",
   ```

4. **Use a smaller model** for testing:
   ```python
   MODEL_ID = "moonshotai/Kimi-K3-0.4B"  # Smaller variant
   ```

### Calibration Hangs

If calibration appears to hang:

1. Check that the dataset loads correctly:
   ```bash
   python -c "from datasets import load_dataset; ds = load_dataset('HuggingFaceH4/ultrachat_200k', split='train_sft[:10]'); print(ds)"
   ```

2. Verify model loads without errors:
   ```bash
   python -c "from llmcompressor.modeling.kimi_k3 import KimiK3ForConditionalGeneration; m = KimiK3ForConditionalGeneration.from_pretrained('moonshotai/Kimi-K3', device_map='cpu', trust_remote_code=True); print('✓ Model loaded')"
   ```

3. Try a minimal test with 1 sample:
   ```python
   NUM_CALIBRATION_SAMPLES = 1
   MAX_SEQUENCE_LENGTH = 512
   ```

### Model Accuracy Issues

After recalibration, if the model produces nonsensical outputs:

1. **Check quantization ignored modules**:
   ```python
   # Ensure residual projections are ignored
   qconfig.ignore += [
       "re:.*mlp_res_proj.*",
       "re:.*self_attention_res_proj.*",
       "re:.*routed_expert.*",
       "re:.*output_attn_res_proj.*",
   ]
   ```

2. **Verify MoE calibration**:
   ```python
   moe_calibrate_all_experts=True,  # Ensure all experts are calibrated
   ```

3. **Check error propagation**:
   ```python
   propagate_error=True,  # Should be enabled for sequential
   ```

## Validation Checklist

Before considering the example working:

- [ ] Example runs without errors
- [ ] Model is saved to `Kimi-K3-NVFP4-Layerwise/`
- [ ] Peak GPU memory is < 80 GB (not 120+ GB)
- [ ] Recalibrated model can generate text coherently
- [ ] Model produces reasonable responses (no garbage output)
- [ ] Calibration completes in < 30 minutes (on single GPU)

## Next Steps

1. **Run the sequential example** — test memory efficiency
2. **Compare to non-sequential** — verify time/memory tradeoff
3. **Test model quality** — ensure accuracy is maintained
4. **Monitor for layerwise PR** — ready to test full version when merged
5. **Add to CI/CD** — include in regression tests once stable

## References

- Sequential Pipeline: `examples/quantizing_moe/kimi_k3_layerwise_example.py`
- Full Layerwise (future): `examples/quantizing_moe/kimi_k3_layerwise_full_example.py`
- Documentation: `examples/quantizing_moe/LAYERWISE_DECOMPRESSION.md`
- Original PR #2994: https://github.com/vllm-project/llm-compressor/pull/2994
- Layerwise commit: `7e0a204cd` (in main branch)
