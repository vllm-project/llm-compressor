# Saving a Compressed Model

The `llmcompressor` library extends Hugging Face's `save_pretrained` method with additional arguments to support model compression functionality. Serialization is handled by [compressed-tensors](https://github.com/neuralmagic/compressed-tensors), which manages the on-disk format for quantized and sparse models. This document explains these extra arguments and how to use them effectively.

## How It Works

When you import `llmcompressor`, it automatically wraps the model's original `save_pretrained` method with an enhanced version that supports compression. This happens in two ways:

1. **Direct modification**: When you call `modify_save_pretrained(model)` directly
2. **Automatic wrapping**: When you call `oneshot(...)`, which wraps `save_pretrained` under the hood

This means that after applying compression with `oneshot`, your model's `save_pretrained` method is already enhanced with compression capabilities, and you can use the additional arguments described below.

## Additional Arguments

When saving your compressed models, you can use the following extra arguments with the `save_pretrained` method:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `quantization_format` | `Optional[str]` | `None` | The on-disk serialization format for quantized weights, defined by `compressed_tensors.QuantizationFormat`. If not provided, it is inferred from the model's quantization scheme. See the compressed-tensors documentation for available formats. |
| `save_compressed` | `bool` | `True` | Controls whether to save the model in a compressed format. Set to `False` to save in the original frozen state. |

## Examples

### Applying Compression with oneshot

The simplest approach is to use `oneshot`, which handles both compression and wrapping `save_pretrained`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier

# Load model
model = AutoModelForCausalLM.from_pretrained("your-model")
tokenizer = AutoTokenizer.from_pretrained("your-model")

# Apply compression - this also wraps save_pretrained
oneshot(
    model=model,
    recipe=[GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"])],
    # Other oneshot parameters...
)

# Now you can use the enhanced save_pretrained
SAVE_DIR = "your-model-W8A8-compressed"
model.save_pretrained(
    SAVE_DIR,
    save_compressed=True
)
tokenizer.save_pretrained(SAVE_DIR)
```

### Setting quantization_format Explicitly

You can override the inferred format by passing `quantization_format` directly using `compressed_tensors.QuantizationFormat`. This is useful when you want to control exactly how weights are serialized on disk:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from compressed_tensors import QuantizationFormat
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

model = AutoModelForCausalLM.from_pretrained("your-model")
tokenizer = AutoTokenizer.from_pretrained("your-model")

oneshot(
    model=model,
    recipe=[QuantizationModifier(targets="Linear", scheme="W4AFP8", ignore=["lm_head"])],
)

SAVE_DIR = "your-model-W4AFP8"
model.save_pretrained(
    SAVE_DIR,
    save_compressed=True,
    quantization_format=QuantizationFormat.pack_quantized,
)
tokenizer.save_pretrained(SAVE_DIR)
```

### Preserving and Quantizing MTP Layers

Some models ship Multi-Token Prediction (MTP) layers used as the draft model for speculative decoding in vLLM. Transformers omits these layers from the loaded backbone, so LLM Compressor reads them separately from the source checkpoint when saving. Both `oneshot(output_dir=...)` and `oneshot(...); model.save_pretrained(...)` write MTP alongside the backbone. Source layouts and shards are validated before calibration.

Use `mtp_quant_scheme=None` and `mtp_dequantize=False` (the defaults) to keep the source MTP format. Dense tensors are copied unchanged; quantized weights are reproduced in the same format through the compressed-tensors converter. Use `mtp_dequantize=True` with no quantization scheme to dequantize or cast MTP to BF16 and exclude its runtime modules from quantization. Compressed-tensors sources must include compatible `mtp_group` metadata.

For a native FP8 checkpoint, first load the backbone with Transformers'
`FineGrainedFP8Config(dequantize=True)`, then pass that model to `oneshot`.
The backbone must be fully dequantized. MTP processing reads the original
source separately: the default reproduces the source FP8 format;
`mtp_dequantize=True` requests BF16 MTP. Backbone and MTP precision are independent.
A still-quantized FP8 backbone remains unsupported.

Pass a quantization preset or `QuantizationScheme` as `mtp_quant_scheme` to apply data-free quantization to the MTP layers. Fully dynamic activation quantization such as `FP8_DYNAMIC` is supported. Schemes that require calibration are not applied. A conversion that cannot be applied is fatal unless `mtp_dequantize=True`, which falls back to BF16. Unexpected errors, resource failures, and source dequantization errors remain fatal.

MTP processing supports the Qwen3.5/Qwen3.8-27B dense architecture,
GLM-5.3 Flash (`Glm5Next`) and DSA (`GlmMoeDsa`), and NVIDIA's Nemotron3.5
Lightning (`NemotronH`) layout, including the instantiated causal-LM aliases.
Qwen3.5 MoE supports dense MTP preservation and BF16 conversion; MTP quantization is not supported.
Each architecture has an explicit projection layout so unsupported tensors fail
instead of being quantized by a broad name heuristic. Other architectures warn and
skip unloaded MTP processing; this does not preserve their MTP tensors.

```python
SAVE_DIR = "your-model-NVFP4-MTP"
oneshot(
    model=model,
    processor=tokenizer,
    recipe=recipe,
    dataset=dataset,
    output_dir=SAVE_DIR,
    mtp_quant_scheme="FP8_DYNAMIC",  # or "MXFP4", "NVFP4A16", or None
    mtp_dequantize=False,
)
```

Both MTP arguments passed to `oneshot` are also used by later calls to that model's
wrapped `save_pretrained`. Missing shards, incompatible layouts, failed source
dequantization, and checkpoint write failures remain fatal. All MTP tensors are
routed through the compressed-tensors converter: quantized sources are dequantized
and requantized to the target format rather than reusing source bytes.

Native block-FP8 sources are dequantized and requantized to the same block-FP8
format through the converter, using the source's own block size. Block-FP8 MoE
expert weights must have dimensions divisible by their block size. For example,
Nemotron3.5 Lightning's 1856-wide experts cannot use the `FP8_BLOCK` preset; use
`FP8_DYNAMIC` instead.

A requested quantization scheme determines the final MTP format regardless of
`mtp_dequantize`; required intermediate dequantization happens automatically.

Choosing a scheme:

| `mtp_quant_scheme` | Notes |
|--------------|-------|
| `None` (default) | Reproduces the source MTP format through the converter; `mtp_dequantize=True` saves BF16 instead. |
| `"FP8_DYNAMIC"` | Keeps calibration-free runtime activation quantization and uses weight-derived per-channel scales. |
| `"FP8_BLOCK"` | Applies data-free block-FP8 weight quantization with dynamic activations. |
| `"MXFP4"` | Applies data-free MXFP4 weight and dynamic activation quantization. |
| `"NVFP4A16"` | Applies data-free NVFP4 weight-only quantization. |

Runtime compatibility depends on the architecture, format, and backend.
Nemotron MXFP4 has a vLLM MoE scale-loading issue; use `FP8_DYNAMIC` or
`NVFP4A16` for that model.

Calibration-based MTP quantization, including GPTQ and AWQ, is not supported by this pathway because the MTP layers are not constructed for calibration.

## Notes

!!! warning
    Sparse compression (including 2of4 sparsity) is no longer supported by LLM Compressor due lack of hardware support and user interest. Please see https://github.com/vllm-project/vllm/pull/36799 for more information.

- When loading compressed models with `from_pretrained`, the compression format is automatically detected by `compressed-tensors`.
- To use compressed models with vLLM, simply load them as you would any model:
  ```python
  from vllm import LLM
  model = LLM("./your-model-compressed")
  ```
- Compression configurations are saved in the model's `config.json` and are automatically applied when loading.
