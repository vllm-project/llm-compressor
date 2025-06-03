# Observers Overview

An `Observer` in `llm-compressor` is a utility class responsible for analyzing tensors (e.g., weights, activations) and producing quantization parameters such as `scale` and `zero_point`. These observers are used by quantization modifiers to compute the statistics necessary for transforming tensors into lower precision formats.

Observers are designed to be flexible and support a variety of quantization strategies, including per-tensor, per-group, per-channel, and per-token quantization.

## Base Class

### [Observer](../src/llmcompressor/observers/base.py)
Base class for all observers. Subclasses must implement the `calculate_qparams` method to define how quantization parameters are computed.

The base class handles:
- Group-wise scale/zero_point computation
- Token-wise and channel-wise quantization logic
- Optional support for `g_idx` (group index mappings)
- Recording observed tokens for logging and analysis
- Resetting internal state during lifecycle transitions

This class is not used directly but provides the scaffolding for all custom observers.

## Implemented Observers

### [MinMax](../src/llmcompressor/observers/min_max.py)
Computes `scale` and `zero_point` by tracking the minimum and maximum of the observed tensor. This is the simplest and most common observer. Works well for symmetric and asymmetric quantization.

Best used for:
- Int8 or Int4 symmetric quantization
- Channel-wise or group-wise strategies

### [MSE](../src/llmcompressor/observers/mse.py)
Computes quantization parameters by minimizing the Mean Squared Error (MSE) between the original and quantized tensor. Optionally maintains a moving average of min/max values for smoother convergence.

Best used when:
- Calibration accuracy is critical
- Quantization error needs to be tightly controlled

## Quantization Strategies

Observers support multiple quantization strategies via the `QuantizationArgs.strategy` field:

- `TENSOR`: Global scale and zero_point across entire tensor.
- `GROUP`, `TENSOR_GROUP`: Slice tensor into equal-sized groups along columns.
- `CHANNEL`: Per-channel quantization (e.g., across output dimensions).
- `TOKEN`: Quantize activations along token or sequence dimensions.
- `BLOCK`: *(Not yet implemented)* Placeholder for block-wise quantization.

## Observer Configuration Parameters

Observers can be configured with optional keyword arguments that control their behavior. These are passed through the `QuantizationArgs.observer_kwargs` dictionary and parsed internally when the observer is initialized.

Below are the supported configuration parameters and their meanings:

| Argument            | Default Value |
|---------------------|---------------|
| `maxshrink`         | `0.20`        |
| `patience`          | `5`           |
| `averaging_constant`| `0.01`        |
| `grid`              | `100.0`       |
| `norm`              | `2.0`         |

## Example Usage

```python
from llmcompressor.observers import Observer
from compressed_tensors.quantization.quant_args import QuantizationArgs

args = QuantizationArgs(num_bits=4, strategy="group", group_size=128)
observer = Observer.load_from_registry("minmax", quantization_args=args)

x = torch.randn(64, 512)
scale, zero_point = observer(x)
```

## Example yaml Usage
``` yaml
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