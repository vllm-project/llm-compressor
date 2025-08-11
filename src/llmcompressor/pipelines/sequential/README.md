# Sequential Pipeline #
The sequential pipeline is a data pipeline, primarily used for compressing models with the
[GPTQModifier](/src/llmcompressor/modifiers/quantization/gptq/base.py) or the
[SparseGPTModifier](/src/llmcompressor/modifiers/obcq/base.py).

## Configuration Options

### Disable Quantization During Calibration

You can optionally disable quantization during calibration by setting the `disable_quantization_during_calibration` parameter to `True`. This can be useful for debugging or when quantization interferes with the calibration process.

**Usage:**
```python
from llmcompressor import oneshot
from llmcompressor.args import DatasetArguments

# Create dataset arguments with quantization disabled during calibration
dataset_args = DatasetArguments(
    dataset="ultrachat-200k",
    disable_quantization_during_calibration=True
)

# Use in oneshot
oneshot(
    model=model,
    dataset=dataset_args,
    recipe=recipe,
    # ... other arguments
)
```

**Default:** `False` (quantization remains enabled during calibration)
