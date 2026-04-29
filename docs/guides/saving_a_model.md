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