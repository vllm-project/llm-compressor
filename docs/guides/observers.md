# Observers Overview

An `Observer` in `llm-compressor` is a utility class responsible for analyzing weight and activation tensors during calibration. Observers work in two phases:

1. **Observe**: accumulate statistics from tensors via `forward()` / `update_statistics_from_observed()`. Most built-in observers accumulate `min_vals`/`max_vals`, but observers can track any statistics they need (e.g., `imatrix_mse` tracks `_imatrix_sum`/`_imatrix_count`).
2. **Compute**: derive quantization parameters (`scale`, `zero_point`, and optionally `global_scale`) from accumulated statistics via `get_qparams()`.

The default `get_qparams` passes `min_vals`/`max_vals` to `compressed-tensors` — specifically `calculate_qparams` for standard quantization and `generate_gparam` for FP4 global scales — to produce the final quantization parameters. Observers with custom statistics can override this method.

This two-phase design enables correct handling of fused layers (e.g., Q/K/V projections) that must share a `global_scale`: all observers accumulate statistics first, then `global_scale` is computed from the combined statistics of all fused observers.

Observers are designed to be flexible and support a variety of quantization strategies, including per-tensor, per-group, per-channel, and per-token quantization.

## Base Class

### [Observer](../../src/llmcompressor/observers/base.py)
Base class for all observers. Subclasses must implement `update_statistics_from_observed` to define how statistics are accumulated. Subclasses that use statistics other than `min_vals`/`max_vals` must also override `get_qparams` and `has_statistics`.

The base class handles:
- Reshaping and slicing tensors according to the quantization strategy (group, channel, token, etc.) via `flatten_for_calibration`
- Computing `scale` and `zero_point` from accumulated statistics via `get_qparams` (default implementation uses `min_vals`/`max_vals` with `calculate_qparams` in `compressed-tensors`)
- Computing `global_scale` for FP4 schemes (e.g., NVFP4, MXFP4) from the combined statistics of all fused observers via `generate_gparam` in `compressed-tensors`
- Fusing observers across layer groups (Q/K/V, gate/up) via `Observer.fuse()` for shared `global_scale`
- DDP synchronization of accumulated statistics via `sync_activation_stats()` using the declarative `_act_sync_dict`

This class is not used directly but provides the scaffolding for all custom observers.

### Key Methods

| Method | Description |
|--------|-------------|
| `forward(observed)` | Accumulates statistics from the observed tensor. Returns `self` (for chaining). Does **not** compute qparams. |
| `update_statistics_from_observed(observed)` | Abstract. Subclasses implement this to accumulate statistics from a pre-shaped tensor. |
| `get_qparams()` | Converts accumulated statistics into a `QParamsDict` with keys `scale`, `zero_point`, and `global_scale`. For TENSOR_GROUP, auto-observes any fused observers without statistics before computing shared `global_scale`. Default implementation uses `min_vals`/`max_vals`. Override for custom statistics. |
| `Observer.fuse(observers_and_modules)` | Static method. Takes an iterable of `(observer, module)` tuples. Links observers with weak module references so they share `global_scale` computed from combined statistics. |
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

All MinMax observers compute min/max values by tracking the minimum and maximum of the observed tensor. They differ in how they handle state across multiple observations.

#### [memoryless_minmax](../../src/llmcompressor/observers/min_max.py)
Computes min/max from each observed tensor independently, with no memory of past observations. Each observation is handled in isolation.

Best used when:
- Only a single observation e.g. for weight quantization
- You want the most recent observation to fully determine the range

#### [static_minmax](../../src/llmcompressor/observers/min_max.py) *(default)*
Tracks the running global min/max across all observations. The final range is the union of all observed ranges — the smallest min and largest max seen across all batches.

Best used for:
- Scenarios where the range must encompass all possible observed values
- Most standard activation quantization scenarios with static/local quantization (FP8, NVFP4)

#### [minmax](../../src/llmcompressor/observers/min_max.py) 
Computes min/max using an exponential moving average across observations, controlled by `averaging_constant`. This smooths out batch-to-batch variance while still adapting to new observations.

Best used for:
- Scenarios with static/local activation quantization where there are infrequent outliers we'd like to average out but MSE observers are too slow.
- Scenarios where the activations change over time and we want to gradually update the statistics over time.

### MSE Observers

MSE observers find the min/max range that minimizes the mean quantization error, using a grid search over candidate scaled min/max factors. They are more expensive than MinMax observers but can yield better accuracy, particularly for integer and low-bit floating point quantization.

#### [memoryless_mse](../../src/llmcompressor/observers/mse.py)
Performs an MSE grid search on each observed tensor independently, with no memory of past observations.

Best used when:
- A single calibration batch is used
- Minimizing quantization error is more important than calibration speed

#### [mse](../../src/llmcompressor/observers/mse.py)
Performs an MSE grid search and maintains a moving average of the resulting min/max values across observations, controlled by `averaging_constant`.

Best used when:
- Calibration accuracy is critical across multiple batches
- Quantization error needs to be tightly controlled (e.g., 4-bit weight quantization)
- You are doing NVFP4 (see expanded observer below) or integer quantization

#### [NVFP4 expanded MSE](../../src/llmcompressor/observers/mse.py) (`nvfp4_expanded_mse`)

`nvfp4_expanded_mse` is a memoryless MSE observer with defaults tuned for NVFP4
weight quantization. Its defaults search from 1.8x down to approximately 0.8x of the observed
range. This lets the selected range be wider than the strict observed range which can be beneficial as observed in the Four Over Six paper.

##### FourOverSix comparison

FourOverSix, introduced in the [Four Over Six paper](https://arxiv.org/pdf/2512.02010),
is an NVFP4 quantization approach that attempts to mitigate for the fact that FP4 has a relatively large gap between representable values near 4 and 6. In some cases quantizing the max to 4 (a 1.5x decrease in range) rather than 6 (1x decrease) can be beneficial. However, rather than checking the 1x and 1.5x values, we found that searching a range of values (which includes 1x and 1.5x) yielded significantly better results. For this reason among others, we chose to implement the expand observer rather than adding explicit four over six support.

The following results are from [PR #2950](https://github.com/vllm-project/llm-compressor/pull/2950).
The table preserves the PR's reported `+delta` values i.e. increase in PPL above the baseline bf16 eval;

| Configuration | Llama-8B | Qwen3-8B | Qwen3-14B | Qwen3-32B | Llama-70B | MoE-30B | Avg Delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| minimax (RTN) | +0.333 | +0.284 | +0.138 | **+0.099** | +0.201 | +0.208 | +0.211 |
| FourOverSix | **+0.230** | +0.098 | +0.128 | +0.183 | +0.261 | +0.300 | +0.200 |
| `nvfp4_expanded_mse` | +0.242 | **+0.028** | **+0.093** | +0.171 | **+0.149** | **+0.149** | **+0.139** |

### IMatrix Observer

The IMatrix observer weights quantization error by per-input-channel activation importance (E[x²]), so channels that carry more signal get more careful range optimization.

Supports CHANNEL, GROUP, and TENSOR_GROUP strategies for weight-only `Linear` modules. Falls back silently to uniform MSE (i.e., standard `memoryless_mse` behavior) whenever importance data is unavailable.

Best used when:
- 4-bit weight quantization accuracy is critical
- You want to combine with GPTQ for further improvement

#### [NVFP4 expanded IMatrix](../../src/llmcompressor/observers/imatrix.py) (`nvfp4_expanded_imatrix`)

`nvfp4_expanded_imatrix` applies the same NVFP4 range expansion and search as
`nvfp4_expanded_mse`, but weighs each quantization error by per-input-channel
importance like the normal ImatrixObserver

**Results** (W4A16, Llama-3.1-8B, group_size=128, WikiText-2 PPL):

| Config | PPL |
|---|---|
| FP16 baseline | 6.24 |
| RTN `memoryless_minmax` | 6.96 |
| GPTQ | 6.92 |
| AWQ | 6.89 |
| RTN `imatrix_mse` | 6.85 |
| GPTQ + `imatrix_mse` | 6.83 |

### NVFP4 Expanded Observer Configuration

The two NVFP4 expanded observers are selected as weight observers:

```python
from llmcompressor.modifiers.quantization import QuantizationModifier

recipe = QuantizationModifier(
    scheme="NVFP4",
    # Use "nvfp4_expanded_imatrix" when activation importance weighting is desired.
    weight_observer="nvfp4_expanded_mse",
)
```

To configure the search explicitly, pass the arguments through
`QuantizationArgs.observer_kwargs`:

```python
from compressed_tensors.quantization import QuantizationArgs

weights = QuantizationArgs(
    num_bits=4,
    type="float",
    symmetric=True,
    strategy="tensor_group",
    group_size=16,
    observer="nvfp4_expanded_imatrix",
    observer_kwargs={"expand": 1.8, "strict": True},
)
```

Both observers perform their local-range search with `global_scale=None`.
After the final min/max values are selected, the normal observer flow computes
NVFP4 `global_scale`; for fused `TENSOR_GROUP` layers it is still shared across
the fused observers. Expansion therefore changes the local range search and
not the fused-global-scale procedure. Using either observer as an NVFP4 weight
observer does not remove NVFP4's normal calibration requirement for its
activation scales; the IMatrix variant additionally uses those calibration
inputs to collect importance statistics.

## Observer Fusion (global_scale)

For TENSOR_GROUP quantization schemes (e.g., NVFP4), layers that are fused at inference time (Q/K/V projections, gate/up MLP projections) must share the same `global_scale`. This is handled automatically by observer fusion:

1. `fuse_weight_observers(model)` scans the model for known fused layer groups and calls `Observer.fuse(observers_and_modules)` with `(observer, module)` tuples
2. Each observer stores weak references to its fused partners' modules in `_fusions: dict[Observer, ref[Module]]`
3. When `get_qparams()` is called on any fused observer:
   - It automatically observes any fused observers that don't have statistics yet by calling them with their stored module references
   - It computes `global_scale` from the absmax across its own statistics **and** all fused observers' statistics

This design eliminates the need to explicitly observe all fused modules before calling `get_qparams()` - unobserved fused modules are handled on-demand. The weak references prevent circular reference cycles between observers and modules. It also handles situations where the subgraphs being quantized at the same time don't include all fused modules.

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
| `memoryless_minmax` / `memoryless_mse` / `nvfp4_expanded_mse` | *(none)* | — |
| `imatrix_mse` / `nvfp4_expanded_imatrix` | `_imatrix_sum`, `_imatrix_count` | SUM |

## Quantization Strategies

Observers support multiple quantization strategies via the `QuantizationArgs.strategy` field:

- `TENSOR`: Statistics computed across the entire tensor.
- `GROUP`, `TENSOR_GROUP`: Tensor sliced into equal-sized groups along columns. `TENSOR_GROUP` additionally computes a `global_scale`.
- `CHANNEL`: Per-channel statistics (e.g., across output dimensions).
- `TOKEN`: Per-token statistics along token or sequence dimensions.
- `BLOCK`: Block-wise quantization with configurable block structure.

### GPTQ Activation Ordering (`actorder`)

Column reordering for GPTQ `actorder` is handled at compression time, after
observer statistics have been collected; it does not change observer ranges or
the statistics accumulated during calibration. When enabled, GPTQ sorts the
working weight columns by descending Hessian diagonal (activation importance),
reorders both axes of the working Hessian, performs the solve, and restores the
original column order before saving. The resulting checkpoint keeps the normal
column order and runtime quantization format; any `g_idx` used to select group
scales is an internal GPTQ work tensor, not a runtime activation-ordering
requirement.

The currently supported GPTQ values are:

- `static`: the default for `GPTQModifier`; an alias for `weight`.
- `weight`: activation ordering during the GPTQ solve only. It keeps the normal
  runtime quantization format and typically gives a small accuracy improvement
  over no activation ordering without adding runtime latency.
- `None`: disables activation ordering.

The older `group` and `dynamic` activation-ordering modes have been removed,
and `actorder=true` is rejected. Set `actorder` on `GPTQModifier` or on the
weight `QuantizationArgs`; conflicting values at both levels raise an error.

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
| `expand`             | `1.0`   | Multiplier for the initial min/max range. Must be at least `1.0`; the NVFP4 expanded MSE observer defaults to `1.8`. |
| `triton_error_buffer` | `0.30` for integer/FP8, `1.00` for FP4 | Relative error buffer used by the CUDA Triton implementation when applying per-group patience. The NVFP4 expanded MSE observer defaults to `1.00`. |
| `averaging_constant` | `0.01`  | EMA weight for moving average. Only applies to `mse`. |

### MSE Triton Kernels and Error Buffer

MSE observers dispatch their grid search through a CUDA Triton backend when the
observed tensor is on CUDA with `float32`, `float16`, or `bfloat16` values and
the quantization format is supported: integer quantization up to 8 bits, or
floating-point quantization at 4 or 8 bits. Unsupported inputs automatically
use the eager implementation instead.

The Triton implementation uses two kernels depending on the amount of data in
each quantization parameter:

- For up to 512 values (`num_observations * group_size <= 512`), a packed kernel
  evaluates several quantization parameters together in one program.
- For larger inputs, a chunked kernel computes errors in 512-value chunks and a
  reduction kernel combines the partial errors before selecting the best range.

Both paths use the shared quantize/dequantize implementation and track
`patience` independently for each quantization group. This allows groups whose
search has converged to stop while other groups continue evaluating candidates.
The Triton paths preserve the eager search's quantization arithmetic while
reducing the cost of large MSE calibration searches, including packed BF16
NVFP4 groups. Because patience is tracked per group and uses the error buffer,
the Triton path can stop at a different grid point than the eager path when
early stopping is reached.

`triton_error_buffer` controls when a candidate resets the Triton patience
counter. A candidate whose error is within the buffer of the best error so far
is considered close enough to reset the counter:

```text
within_buffer = candidate_error <= best_error * (1 + triton_error_buffer)
```

For example, `triton_error_buffer=0.30` treats errors up to 30% above the
current best as close enough to continue the search. A larger buffer makes
early stopping more permissive and usually allows more grid points to be
evaluated; a smaller buffer makes stopping more aggressive. The buffer affects
only the Triton early-stopping logic, not the error objective itself, and is
ignored by the eager implementation.

MSE and iMatrix grid searches run with `global_scale=None` — the shrink search optimizes using FP32 scales, since `global_scale` cancels out when comparing quantization error across shrink candidates. The actual `global_scale` is computed later in `get_qparams()` from the final min/max values.

### IMatrix observer (`imatrix_mse`)

| Argument      | Default | Description |
|---------------|---------|-------------|
| `maxshrink`   | `0.95`  | Maximum shrink factor for the grid search. The search evaluates `int(maxshrink * grid)` shrink steps. |
| `patience`    | `5`     | Number of consecutive steps without improvement before early stopping. |
| `grid`        | `20`    | Number of grid steps. Higher values give finer granularity at the cost of speed. |
| `norm`        | `3.0`   | Exponent used when computing the importance-weighted error. |
| `expand`      | `1.0`   | Multiplier for the initial min/max range. Must be at least `1.0`; `nvfp4_expanded_imatrix` defaults to `1.8`. |
| `strict`      | `False` | If `True`, raise an error instead of falling back to uniform MSE when importance data is unavailable. |

## DDP (Distributed) Support

Observers support distributed data-parallel calibration. When activation quantization runs across multiple ranks (each rank processing a disjoint partition of the calibration dataset), observer statistics must be synchronized before quantization parameters are computed so all ranks produce identical results.

### How it works

Each observer class declares a `_act_sync_dict` class attribute mapping statistic attribute names to their `torch.distributed.ReduceOp`. The base class `sync_activation_stats()` method iterates this dict and issues async `dist.all_reduce` calls:

```python
# Called by QuantizationMixin.sync_obs_act_stats() at each layer boundary
pending_comms = observer.sync_activation_stats()
# ... wait_for_comms(pending_comms) once all observers in the subgraph are queued
```

Only **activation** statistics are synchronized. Weight statistics are never synced because weights are identical across all ranks (broadcast during model load).

### Per-observer sync behavior

| Observer | Synced statistics | Reduce op | Notes |
|----------|-------------------|-----------|-------|
| `static_minmax` | `min_vals`, `max_vals` | MIN, MAX | Global min/max across all ranks |
| `minmax` (EMA) | `min_vals`, `max_vals` | AVG | Global min/max across all ranks |
| `memoryless_minmax` | *(none)* | — | Stateless; each rank's data is independent |
| `mse` (EMA) | `min_vals`, `max_vals` | AVG | Averages the per-rank MSE-optimal ranges |
| `memoryless_mse` / `nvfp4_expanded_mse` | *(none)* | — | Stateless; each rank's data is independent |
| `imatrix_mse` / `nvfp4_expanded_imatrix` | `_imatrix_sum`, `_imatrix_count` | SUM | Accumulates importance scores across ranks before normalization |

For more information on the distributed oneshot workflow, see [Distributed Oneshot](./big_models_and_distributed/distributed_oneshot.md).

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
