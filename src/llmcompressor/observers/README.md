# Observers Overview

An `Observer` in `llm-compressor` is a utility class responsible for analyzing tensors (e.g., weights, activations) and producing quantization parameters such as `scale` and `zero_point`. These observers are used by quantization modifiers to compute the statistics necessary for transforming tensors into lower precision formats.

Observers are designed to be flexible and support a variety of quantization strategies, including per-tensor, per-group, per-channel, and per-token quantization.

## Base Class

### [Observer](./observers/base.py)
Base class for all observers. 
The base class handles:
- Group-wise scale/zero_point computation
- Token-wise and channel-wise quantization logic
- Optional support for `g_idx` (group index mappings)
- Recording observed tokens for logging and analysis
- Resetting internal state during lifecycle transitions

This class is not used directly but provides the scaffolding for all custom observers.

## Implemented Observers

### [MinMax](./observers/min_max.py)
Computes `scale` and `zero_point` by tracking the minimum and maximum of the observed tensor. This is the simplest and most common observer. Works well for symmetric and asymmetric quantization.

### [MSE](./observers/mse.py)
Computes quantization parameters by minimizing the Mean Squared Error (MSE) between the original and quantized tensor. Optionally maintains a moving average of min/max values for smoother convergence.

### [SmoothQuant Observer](./observers/smoothquant_observer.py)
Used in conjunction with the `SmoothQuant` modifier. Tracks outlier activations and computes scales for pre-conditioning inputs before quantization. Applies smoothing to activations while preserving model semantics.

## Quantization Strategies

Observers support multiple quantization strategies via the `QuantizationArgs.strategy` field:

- `TENSOR`: Global scale and zero_point across entire tensor.
- `GROUP`, `TENSOR_GROUP`: Slice tensor into equal-sized groups along columns.
- `CHANNEL`: Per-channel quantization (e.g., across output dimensions).
- `TOKEN`: Quantize activations along token or sequence dimensions.
- `BLOCK`: *(Not yet implemented)* Placeholder for block-wise quantization.

## Observer Configuration Parameters

Observers can be configured with optional keyword arguments that control their behavior. These are passed through the `QuantizationArgs.observer_kwargs` dictionary and parsed internally when the observer is initialized.

Below are the supported configuration parameters and their default values:

| Argument             | Default Value |
|----------------------|---------------|
| `maxshrink`          | `0.20`        |
| `grid`               | `100.0`       |
| `averaging_constant` | `0.01`        |
| `norm`               | `2.0`         |
| `patience`           | `5`           |


### Example yaml recipe

```yaml
quantization_stage:
  quantization_modifiers:
    GPTQModifier:
      weights:
        observer: mse
        observer_kwargs:
          maxshrink: 0.1
          patience: 10
          averaging_constant: 0.05
          grid: 128.0
          norm: 2.0
        num_bits: 4
        type: int
        symmetric: true
        strategy: channel
      targets:
        - Linear
```

## Example Usage

```python
from llmcompressor.observers import Observer
from compressed_tensors.quantization.quant_args import QuantizationArgs

args = QuantizationArgs(num_bits=4, strategy="group", group_size=128)
observer = Observer.load_from_registry(
        "minmax",
        quantization_args=args,
        ignore_averaging_constant=True,
    )

x = torch.randn(64, 512)
scale, zero_point = observer(x)
```

`ignore_avegrating_constant` is used when user doesn't want to save moving average of min/max values.
