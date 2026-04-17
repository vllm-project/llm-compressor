# Observers Overview

An `Observer` in `llm-compressor` is a utility class responsible for analyzing weight and activation tensors during calibration and computing **min and max values** that characterize the tensor's range. These min/max values are then passed to `compressed-tensors` — specifically `calculate_qparams` for standard quantization and `generate_gparam` for FP4 global scales — to produce the final `scale` and `zero_point` used for quantization.

Observers do not compute scales or zero points directly. Their responsibility is to accurately characterize the range of a tensor so that `compressed-tensors` can derive the appropriate quantization parameters.

Observers are designed to be flexible and support a variety of quantization strategies, including per-tensor, per-group, per-channel, and per-token quantization.

## Base Class

### [Observer](../../src/llmcompressor/observers/base.py)
Base class for all observers. Subclasses must implement `get_min_max` and `get_global_min_max` to define how tensor range statistics are computed.

The base class handles:
- Reshaping and slicing tensors according to the quantization strategy (group, channel, token, etc.)
- Passing the resulting min/max values to `calculate_qparams` in `compressed-tensors` to produce `scale` and `zero_point`
- Computing global min/max and passing to `generate_gparam` in `compressed-tensors` for FP4 global scale generation (e.g., NVFP4, MXFP4)
- Optional support for `g_idx` (group index mappings)

This class is not used directly but provides the scaffolding for all custom observers.

## Implemented Observers

### MinMax Observers

All MinMax observers compute min/max values by tracking the minimum and maximum of the observed tensor. They differ in how they handle state across multiple calibration batches.

#### [memoryless_minmax](../../src/llmcompressor/observers/min_max.py)
Computes min/max from each observed tensor independently, with no memory of past observations. Each calibration batch is treated in isolation.

Best used when:
- Only a single calibration batch is used
- You want the most recent observation to fully determine the range

#### [static_minmax](../../src/llmcompressor/observers/min_max.py)
Tracks the running global min/max across all calibration batches. The final range is the union of all observed ranges — the smallest min and largest max seen across all batches.

Best used for:
- Scenarios where the range must encompass all possible observed values
- Int8 or Int4 symmetric quantization with multiple calibration batches

#### [minmax](../../src/llmcompressor/observers/min_max.py) *(default)*
Computes min/max using an exponential moving average across calibration batches, controlled by `averaging_constant`. This smooths out batch-to-batch variance while still adapting to new observations.

Best used for:
- Most standard quantization scenarios (Int8, Int4)
- Channel-wise or group-wise strategies with multiple calibration batches

### MSE Observers

MSE observers find the min/max range that minimizes the mean quantization error, using a grid search over candidate shrink factors. They are more expensive than MinMax observers but can yield better accuracy, particularly for low-bit quantization.

#### [memoryless_mse](../../src/llmcompressor/observers/mse.py)
Performs an MSE grid search on each observed tensor independently, with no memory of past observations.

Best used when:
- A single calibration batch is used
- Minimizing quantization error is more important than calibration speed

#### [mse](../../src/llmcompressor/observers/mse.py)
Performs an MSE grid search and maintains a moving average of the resulting min/max values across calibration batches, controlled by `averaging_constant`.

Best used when:
- Calibration accuracy is critical across multiple batches
- Quantization error needs to be tightly controlled (e.g., 4-bit weight quantization)

### IMatrix Observer

The IMatrix observer weights quantization error by per-input-channel activation importance (E[x²]), so channels that carry more signal get more careful range optimization.

#### [imatrix_mse](../../src/llmcompressor/observers/imatrix.py)
Extends the MSE grid search with importance weighting: `err = sum(importance * |Q(w) - w|^p)`. Importance scores (E[x²] per input channel) are collected by a preceding `IMatrixGatherer` modifier during a dedicated calibration pass.

Supports CHANNEL, GROUP, and TENSOR_GROUP strategies for weight-only `Linear` modules. Falls back silently to uniform MSE (i.e., standard `memoryless_mse` behavior) whenever importance data is unavailable — for example, when no `IMatrixGatherer` preceded the quantization step — unless `strict=True` is set.

Best used when:
- 4-bit weight quantization accuracy is critical
- You want to combine with GPTQ for further improvement

**Requires `IMatrixGatherer`** as the first modifier in your recipe to trigger the calibration pass that collects E[x²]. See [IMatrixGatherer](#imatrixgatherer) below.

**Results** (W4A16, Llama-3.1-8B, group_size=128, WikiText-2 PPL):

| Config | PPL |
|---|---|
| FP16 baseline | 6.24 |
| RTN `memoryless_minmax` | 6.96 |
| GPTQ | 6.92 |
| AWQ | 6.89 |
| RTN `imatrix_mse` | 6.85 |
| GPTQ + `imatrix_mse` | 6.83 |

### IMatrixGatherer

[`IMatrixGatherer`](../../src/llmcompressor/modifiers/transform/imatrix/base.py) is a modifier (not an observer) that must precede `QuantizationModifier` or `GPTQModifier` in your recipe when using `imatrix_mse`. It orchestrates a dedicated calibration pass during which the observer collects E[x²] per input channel via forward pre-hooks. It does **not** quantize weights.

At the end of the calibration epoch, it calls `observer.detach()` on each instrumented module, which computes `_imatrix_importance` and leaves it on the module for the subsequent quantization pass.

| Parameter | Default | Description |
|---|---|---|
| `targets` | `["Linear"]` | Module types to instrument for importance collection. |
| `ignore` | `["lm_head"]` | Layer name patterns to skip. |
| `weight_observer` | `"imatrix_mse"` | Observer to attach during the calibration pass. |

## Quantization Strategies

Observers support multiple quantization strategies via the `QuantizationArgs.strategy` field:

- `TENSOR`: Global min/max across the entire tensor.
- `GROUP`, `TENSOR_GROUP`: Slice tensor into equal-sized groups along columns.
- `CHANNEL`: Per-channel min/max (e.g., across output dimensions).
- `TOKEN`: Per-token min/max along token or sequence dimensions.
- `BLOCK`: *(Not yet implemented)* Placeholder for block-wise quantization.

## Observer Configuration Parameters

Observers can be configured with optional keyword arguments via `QuantizationArgs.observer_kwargs`.

### MinMax observers (`minmax`, `static_minmax`, `memoryless_minmax`)

| Argument             | Default | Description |
|----------------------|---------|-------------|
| `averaging_constant` | `0.01`  | EMA weight for moving average observers. Only applies to `minmax`. Higher values weight recent observations more heavily. |

### MSE observers (`mse`, `memoryless_mse`)

| Argument             | Default | Description |
|----------------------|---------|-------------|
| `maxshrink`          | `0.20`  | Maximum shrink amount in grid steps. Number of search steps is `int(maxshrink * grid)`. |
| `patience`           | `5`     | Number of consecutive steps without improvement before early stopping. |
| `grid`               | `100.0` | Resolution of the shrink search. Higher values give finer granularity. |
| `norm`               | `2.4`   | Exponent used when computing the error. `norm=2` approximates MSE. |
| `averaging_constant` | `0.01`  | EMA weight for moving average. Only applies to `mse`. |

### IMatrix observer (`imatrix_mse`)

| Argument      | Default | Description |
|---------------|---------|-------------|
| `maxshrink`   | `0.95`  | Maximum shrink factor for the grid search. The search evaluates `int(maxshrink * grid)` shrink steps. |
| `patience`    | `5`     | Number of consecutive steps without improvement before early stopping. |
| `grid`        | `20`    | Number of grid steps. Higher values give finer granularity at the cost of speed. |
| `norm`        | `3.0`   | Exponent used when computing the importance-weighted error. |
| `strict`      | `False` | If `True`, raise an error instead of falling back to uniform MSE when importance data is unavailable. |

## Example Usage

```python
import torch
from llmcompressor.observers import Observer
from compressed_tensors.quantization import QuantizationArgs

args = QuantizationArgs(num_bits=4, strategy="group", group_size=128)
observer = Observer.load_from_registry(
    "minmax",
    base_name="weight",
    args=args,
)

x = torch.randn(64, 512)
scale, zero_point = observer(x)
```

## Example YAML Usage

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
