# Enhanced `save_pretrained` Arguments

The `llmcompressor` library extends Hugging Face's `save_pretrained` method with additional arguments to support model compression functionality. This document explains these extra arguments and how to use them effectively.

## How It Works

When you import `llmcompressor`, it automatically wraps the model's original `save_pretrained` method with an enhanced version that supports compression. This happens in two ways:

1. **Direct modification**: When you call `modify_save_pretrained(model)` directly
2. **Automatic wrapping**: When you call `oneshot(...)`, which wraps `save_pretrained` under the hood

This means that after applying compression with `oneshot`, your model's `save_pretrained` method is already enhanced with compression capabilities, and you can use the additional arguments described below.

## Additional Arguments

When saving your compressed models, you can use the following extra arguments with the `save_pretrained` method:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sparsity_config` | `Optional[SparsityCompressionConfig]` | `None` | Optional configuration for sparsity compression. This should be provided if there's existing sparsity in the model. If None and `skip_sparsity_compression_stats` is False, configuration will be automatically inferred from the model. |
| `quantization_format` | `Optional[str]` | `None` | Optional format string for quantization. If not provided, it will be inferred from the model. |
| `save_compressed` | `bool` | `True` | Controls whether to save the model in a compressed format. Set to `False` to save in the original dense format. |
| `skip_sparsity_compression_stats` | `bool` | `True` | Controls whether to skip calculating sparsity statistics (e.g., global sparsity and structure) when saving the model. Set to `False` to include these statistics. If you are not providing a `sparsity_config`, you should set this to `False` to automatically generate the config for you. |
| `disable_sparse_compression` | `bool` | `False` | When set to `True`, skips any sparse compression during save, even if the model has been previously compressed. |

## Workflow for Models with Existing Sparsity

When working with models that already have sparsity:

1. If you know the sparsity configuration, provide it directly via `sparsity_config`
2. If you don't know the sparsity configuration, set `skip_sparsity_compression_stats` to `False` to automatically infer it from the model

This workflow ensures that the correct sparsity configuration is either provided or generated when saving models with existing sparsity.

## Examples

### Applying Compression with oneshot

The simplest approach is to use `oneshot`, which handles both compression and wrapping `save_pretrained`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

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
    save_compressed=True  # Use the enhanced functionality
)
tokenizer.save_pretrained(SAVE_DIR)
```

### Manual Approach (Without oneshot)

If you need more control, you can wrap `save_pretrained` manually:

```python
from transformers import AutoModelForCausalLM
from llmcompressor.transformers.sparsification import modify_save_pretrained

# Load model
model = AutoModelForCausalLM.from_pretrained("your-model")

# Manually wrap save_pretrained
modify_save_pretrained(model)

# Now you can use the enhanced save_pretrained
model.save_pretrained(
    "your-model-path",
    save_compressed=True,
    skip_sparsity_compression_stats=False  # To automatically infer sparsity config
)
```

### Saving with Custom Sparsity Configuration

```python
from compressed_tensors.sparsification import SparsityCompressionConfig

# Create custom sparsity config
custom_config = SparsityCompressionConfig(
    format="2:4",
    block_size=16
)

# Save with custom config
model.save_pretrained(
    "your-model-custom-sparse",
    sparsity_config=custom_config,
)
```

## Notes

- When loading compressed models with `from_pretrained`, the compression format is automatically detected.
- To use compressed models with vLLM, simply load them as you would any model:
  ```python
  from vllm import LLM
  model = LLM("./your-model-compressed")
  ```
- Compression configurations are saved in the model's config file and are automatically applied when loading.

For more information about compression algorithms and formats, please refer to the documentation and examples in the llmcompressor repository.