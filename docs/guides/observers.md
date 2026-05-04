# Observers Overview

An `Observer` in `llm-compressor` is a utility class responsible for analyzing weight and activation tensors during calibration. Observers work in two phases:

1. **Observe**: accumulate statistics from tensors via `forward()` / `update_statistics()`. Most built-in observers accumulate `min_vals`/`max_vals`, but observers can track any statistics they need (e.g., `imatrix_mse` tracks `_imatrix_sum`/`_imatrix_count`).
2. **Compute**: derive quantization parameters (`scale`, `zero_point`, and optionally `global_scale`) from accumulated statistics via `get_qparams()`, which calls `compute_qparams_from_statistics()`.

The default `compute_qparams_from_statistics` passes `min_vals`/`max_vals` to `compressed-tensors` — specifically `calculate_qparams` for standard quantization and `generate_gparam` for FP4 global scales — to produce the final quantization parameters. Observers with custom statistics can override this method.

This two-phase design enables correct handling of fused layers (e.g., Q/K/V projections) that must share a `global_scale`: all observers accumulate statistics first, then `global_scale` is computed from the combined statistics of all fused observers.

Observers are designed to be flexible and support a variety of quantization strategies, including per-tensor, per-group, per-channel, and per-token quantization.

## Base Class

### [Observer](../../src/llmcompressor/observers/base.py)
Base class for all observers. Subclasses must implement `update_statistics` to define how statistics are accumulated. Subclasses that use statistics other than `min_vals`/`max_vals` must also override `compute_qparams_from_statistics` and `has_statistics`.

The base class handles:
- Reshaping and slicing tensors according to the quantization strategy (group, channel, token, etc.) via `flatten_for_calibration`
- Computing `scale` and `zero_point` from accumulated statistics via `compute_qparams_from_statistics` (default implementation uses `min_vals`/`max_vals` with `calculate_qparams` in `compressed-tensors`)
- Computing `global_scale` for FP4 schemes (e.g., NVFP4, MXFP4) from the combined statistics of all fused observers via `generate_gparam` in `compressed-tensors`
- Fusing observers across layer groups (Q/K/V, gate/up) via `Observer.fuse()` for shared `global_scale`
- DDP synchronization of accumulated statistics via `sync_activation_stats()` using the declarative `_act_sync_dict`

This class is not used directly but provides the scaffolding for all custom observers.

### Key Methods

| Method | Description |
|--------|-------------|
| `forward(observed)` | Accumulates statistics from the observed tensor. Returns `self` (for chaining). Does **not** compute qparams. |
| `update_statistics(observed)` | Abstract. Subclasses implement this to accumulate statistics from a pre-shaped tensor. |
| `compute_qparams_from_statistics()` | Converts accumulated statistics into a `QParamsDict`. Default implementation uses `min_vals`/`max_vals`. Override for custom statistics. |
| `get_qparams()` | Calls `compute_qparams_from_statistics()` and returns a `QParamsDict` with keys `scale`, `zero_point`, and `global_scale`. |
| `Observer.fuse(observers)` | Static method. Links observers so they share `global_scale` computed from combined statistics. |
| `sync_activation_stats()` | All-reduces accumulated statistics across DDP ranks using `_act_sync_dict`. |
| `has_statistics` | Property. Returns `True` if the observer has been called at least once. Default checks for `min_vals`; override for custom statistics. |

### QParamsDict

Observers return a `QParamsDict` (a `TypedDict`) from `get_qparams()`:

```python
class QParamsDict(TypedDict, total=False):
    scale: torch.Tensor
    zero_point: torch.Tensor
    global_scale: Optional[torch.Tensor]  # only set for TENSOR_GROUP
```

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

At the end of the calibration epoch, it calls `observer.detach()` on each instrumented module, which leaves raw `_imatrix_sum` and `_imatrix_count` accumulators on the module for the subsequent quantization pass to pick up.

| Parameter | Default | Description |
|---|---|---|
| `targets` | `["Linear"]` | Module types to instrument for importance collection. |
| `ignore` | `["lm_head"]` | Layer name patterns to skip. |
| `weight_observer` | `"imatrix_mse"` | Observer to attach during the calibration pass. |

## Observer Fusion (global_scale)

For TENSOR_GROUP quantization schemes (e.g., NVFP4), layers that are fused at inference time (Q/K/V projections, gate/up MLP projections) must share the same `global_scale`. This is handled automatically by observer fusion:

1. `fuse_weight_observers(model)` scans the model for known fused layer groups and calls `Observer.fuse()` to link their weight observers
2. When any fused observer computes `get_qparams()`, it takes the absmax across its own statistics **and** all fused observers' statistics to produce a shared `global_scale`

The fused layer groups are defined in `FUSED_LAYER_NAMES`:
- `gate_proj` / `up_proj` (MLP)
- `q_proj` / `k_proj` / `v_proj` (attention)
- `q_a_proj` / `kv_a_proj_with_mqa` (DeepSeek multi-latent attention)
- `w1` / `w3` (MoE expert layers)

## DDP Synchronization

Each observer subclass declares an `_act_sync_dict` mapping attribute names to DDP reduce operations. The base class `sync_activation_stats()` iterates this dict to all-reduce accumulated statistics across ranks:

| Observer | Synced Attributes | Reduce Op |
|----------|-------------------|-----------|
| `static_minmax` | `min_vals`, `max_vals` | MIN, MAX |
| `minmax` / `mse` | `min_vals`, `max_vals` | AVG |
| `memoryless_minmax` / `memoryless_mse` | *(none)* | — |
| `imatrix_mse` | `_imatrix_sum`, `_imatrix_count` | SUM |

## Quantization Strategies

Observers support multiple quantization strategies via the `QuantizationArgs.strategy` field:

- `TENSOR`: Statistics computed across the entire tensor.
- `GROUP`, `TENSOR_GROUP`: Tensor sliced into equal-sized groups along columns. `TENSOR_GROUP` additionally computes a `global_scale`.
- `CHANNEL`: Per-channel statistics (e.g., across output dimensions).
- `TOKEN`: Per-token statistics along token or sequence dimensions.
- `BLOCK`: Block-wise quantization with configurable block structure.

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

# Phase 1: accumulate statistics
x = torch.randn(64, 512)
observer(x)

# Phase 2: compute quantization parameters
qparams = observer.get_qparams()
scale = qparams["scale"]
zero_point = qparams["zero_point"]
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
