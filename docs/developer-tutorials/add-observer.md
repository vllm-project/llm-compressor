# Adding a New Observer

Observers analyze weight and activation tensors during calibration to compute the statistics needed for quantization. This guide explains how observers fit into the quantization pipeline and how to implement a custom one.

## What is an Observer?

When a quantized layer runs a calibration forward pass, it passes the weight or activation tensor to an observer. The observer accumulates **statistics** that characterize the tensor's distribution. These statistics are later used to compute quantization parameters (`scale`, `zero_point`, and optionally `global_scale`) via `compressed-tensors`.

Observers work in two phases:

1. **Observe**: `forward()` reshapes the tensor and calls `update_statistics_from_observed()` to accumulate statistics
2. **Compute**: `get_qparams()` converts accumulated statistics into a `QParamsDict`

In situations that require a global scale (e.g., NVFP4) and for weights that require fusion of this global_scale (QKV, MoE), the observers are fused together so that they can jointly calculate a fused global_scale. This requires all fused observers have accumulated statistics.

The base `Observer` class handles all slicing and reshaping for group-wise, channel-wise, and token-wise strategies before calling your subclass. Your subclass needs to define how statistics are accumulated. The base class `get_qparams()` handles converting `min_vals`/`max_vals` into quantization parameters.

## The Observer Contract

All observers subclass `llmcompressor.observers.Observer`. At minimum, you must implement `update_statistics_from_observed`. If your observer uses `min_vals`/`max_vals` as its statistics, the base class `get_qparams` handles the rest. If your observer uses different statistics, you must also override `get_qparams`.

### Simple case: min/max statistics

Most observers accumulate `min_vals` and `max_vals`. In this case, you only need `update_statistics_from_observed` — the base class `get_qparams` will pass your `min_vals`/`max_vals` to `calculate_qparams` and handle `global_scale` for TENSOR_GROUP automatically:

```python
import torch
from llmcompressor.observers import Observer

@Observer.register("my_observer")
class MyObserver(Observer):

    _act_sync_dict = {}

    def update_statistics_from_observed(self, observed: torch.Tensor) -> None:
        """
        Update internal statistics from the observed tensor.

        The base class has already reshaped the tensor into
        shape (num_observations, *qparam_shape, group_size).

        :param observed: pre-processed tensor ready for statistics computation
        """
        self.min_vals = torch.amin(observed, dim=(0, -1))
        self.max_vals = torch.amax(observed, dim=(0, -1))
```

### Custom statistics

If your observer uses statistics other than `min_vals`/`max_vals` (e.g., mean/std, histograms), you must also override `get_qparams` to convert your statistics into a `QParamsDict`. You may also need to override `has_statistics` since the base class checks for `min_vals` by default.

The following example tracks mean and standard deviation, then derives min/max as mean ± k·std to clip outliers before computing quantization parameters:

```python
import torch
from compressed_tensors.quantization import QuantizationStrategy
from compressed_tensors.quantization.utils import calculate_qparams, generate_gparam
from llmcompressor.observers.base import Observer, QParamsDict

@Observer.register("normal_observer")
class NormalObserver(Observer):
    """
    Derives quantization range from mean ± k·std of the observed tensor.
    Configure k via observer_kwargs (default: 2.0).
    """

    _act_sync_dict = {}

    @property
    def has_statistics(self) -> bool:
        return hasattr(self, "_mean") and hasattr(self, "_std")

    def update_statistics_from_observed(self, observed: torch.Tensor) -> None:
        self._mean = observed.mean(dim=(0, -1))
        self._std = observed.std(dim=(0, -1))

    def get_qparams(self) -> QParamsDict:
        k = self.args.observer_kwargs.get("k", 2.0)
        min_vals = self._mean - k * self._std
        max_vals = self._mean + k * self._std

        global_scale = None
        if self.args.strategy == QuantizationStrategy.TENSOR_GROUP:
            global_absmax = torch.max(-min_vals.min(), max_vals.max())
            for obs in self._fused_observers:
                assert obs.has_statistics
                global_absmax = torch.max(global_absmax, obs._mean.abs().max())
            global_scale = generate_gparam(
                -global_absmax.reshape(1), global_absmax.reshape(1)
            )

        scale, zero_point = calculate_qparams(
            min_vals=min_vals,
            max_vals=max_vals,
            quantization_args=self.args,
            global_scale=global_scale,
        )
        return {"scale": scale, "zero_point": zero_point, "global_scale": global_scale}
```

### DDP synchronization

Each observer must also declare `_act_sync_dict` — a class-level dict mapping statistic attribute names to DDP reduce operations. The base class `sync_activation_stats()` uses this to all-reduce statistics across ranks between calibration batches.

- If your observer accumulates state across batches, declare which attributes need syncing and with what operation
- Memoryless observers that overwrite statistics each batch should set `_act_sync_dict = {}`
- Weight observers typically don't need sync (weights are identical across ranks), but activation observers do

```python
from torch import distributed as dist

@Observer.register("my_observer")
class MyObserver(Observer):
    _act_sync_dict = {
        "min_vals": dist.ReduceOp.MIN,
        "max_vals": dist.ReduceOp.MAX,
    }
    ...
```

The `@Observer.register("my_observer")` decorator registers your observer under the given name so it can be referenced in recipes by string.

## How the Base Class Uses Your Statistics

The default `get_qparams` reads `min_vals`/`max_vals` and passes them to `calculate_qparams` from `compressed-tensors`:

```python
# Inside Observer.get_qparams (simplified):

# For TENSOR_GROUP: compute global_scale from this observer
# and all fused observers' min_vals/max_vals
global_scale = None
if self.args.strategy == QuantizationStrategy.TENSOR_GROUP:
    global_absmax = torch.max(-self.min_vals.min(), self.max_vals.max())
    for obs in self._fused_observers:
        global_absmax = torch.max(global_absmax, -obs.min_vals.min())
        global_absmax = torch.max(global_absmax, obs.max_vals.max())
    global_scale = generate_gparam(-global_absmax, global_absmax)

scale, zero_point = calculate_qparams(
    min_vals=self.min_vals,
    max_vals=self.max_vals,
    quantization_args=self.args,
    global_scale=global_scale,
)
return {"scale": scale, "zero_point": zero_point, "global_scale": global_scale}
```

`calculate_qparams` handles the actual scale and zero point computation — symmetric vs asymmetric quantization, dtype clamping, MX scale generation, and so on. Your observer controls what statistics are accumulated and how they map to quantization parameters.

If you override `get_qparams` and your observer supports TENSOR_GROUP, you are responsible for computing `global_scale` from the fused observers yourself (see `self._fused_observers`).

## Stateful Observers

Some observers accumulate statistics across multiple calibration batches. To do this, check for existing state in `update_statistics_from_observed`:

```python
@Observer.register("my_observer")
class MyObserver(Observer):

    _act_sync_dict = {
        "min_vals": dist.ReduceOp.MIN,
        "max_vals": dist.ReduceOp.MAX,
    }

    def update_statistics_from_observed(self, observed: torch.Tensor) -> None:
        min_vals = torch.amin(observed, dim=(0, -1))
        max_vals = torch.amax(observed, dim=(0, -1))

        if self.has_statistics:
            min_vals = torch.min(min_vals, self.min_vals)
            max_vals = torch.max(max_vals, self.max_vals)

        self.min_vals = min_vals
        self.max_vals = max_vals
```

## Example: A Percentile-Clipping Observer

The following observer clips outliers by returning min/max values from a configurable percentile range rather than the absolute extremes. This can improve accuracy when tensors have extreme outlier values that would otherwise inflate the quantization range.

```python
import torch
from llmcompressor.observers import Observer

@Observer.register("percentile")
class PercentileObserver(Observer):
    """
    Clips outliers by setting min_vals/max_vals to a configurable percentile range.

    Configure via observer_kwargs:
        percentile (float): the upper percentile to retain, e.g. 99.9
    """

    _act_sync_dict = {}

    def update_statistics_from_observed(self, observed: torch.Tensor) -> None:
        percentile = self.args.observer_kwargs.get("percentile", 99.9)
        lower = 100.0 - percentile
        upper = percentile

        self.min_vals = torch.tensor(
            [
                torch.quantile(observed[:, i, :].flatten(), lower / 100.0).item()
                for i in range(observed.shape[-2])
            ]
        )
        self.max_vals = torch.tensor(
            [
                torch.quantile(observed[:, i, :].flatten(), upper / 100.0).item()
                for i in range(observed.shape[-2])
            ]
        )
```

### Using the Observer in a Recipe

Reference the registered name (`"percentile"`) via the `observer` field in `QuantizationArgs`:

```python
from llmcompressor.modifiers.quantization import QuantizationModifier
from compressed_tensors.quantization import QuantizationArgs

recipe = QuantizationModifier(
    targets="Linear",
    scheme={
        "weights": QuantizationArgs(
            num_bits=8,
            type="int",
            symmetric=True,
            strategy="channel",
            observer="percentile",
            observer_kwargs={"percentile": 99.5},
        )
    },
    ignore=["lm_head"],
)
```

Or from a YAML recipe:

```yaml
quantization_stage:
  quantization_modifiers:
    QuantizationModifier:
      targets:
        - Linear
      ignore:
        - lm_head
      scheme:
        weights:
          num_bits: 8
          type: int
          symmetric: true
          strategy: channel
          observer: percentile
          observer_kwargs:
            percentile: 99.5
```

## Tips

- **Implement `update_statistics_from_observed` and optionally `get_qparams`.** If your statistics are `min_vals`/`max_vals`, the base class `get_qparams` handles the conversion to `scale`, `zero_point`, and `global_scale`. If you use different statistics, override `get_qparams` (and `has_statistics`) as well.
- **`update_statistics_from_observed` receives a pre-shaped tensor.** The base class has already sliced the input according to `QuantizationArgs.strategy` (group, channel, token, etc.). You do not need to handle reshaping yourself.
- **`global_scale` is handled automatically for min/max observers.** For TENSOR_GROUP strategies, the default `get_qparams()` computes `global_scale` from the combined statistics of all fused observers. If you override `get_qparams`, you must handle fused `global_scale` yourself.
- **Set `_act_sync_dict` correctly.** Every observer must declare this. If your observer accumulates state across batches, map each statistic attribute to its reduce operation. Memoryless observers should set `_act_sync_dict = {}`.
- **`observer_kwargs` is the right place for hyperparameters.** Access them via `self.args.observer_kwargs.get(...)`.
- **Match the shape contract for min/max observers.** If using the default `get_qparams`, set `self.min_vals` and `self.max_vals` to tensors of shape `(*qparam_shape,)` — one scalar per quantization group/channel/token.
- **Existing observers are good references.** See `min_max.py` for a simple stateless min/max example, `mse.py` for a stateful one, and `imatrix.py` for an observer that uses custom statistics beyond min/max.
