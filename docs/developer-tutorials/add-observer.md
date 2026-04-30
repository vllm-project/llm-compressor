# Adding a New Observer

Observers analyze weight and activation tensors during calibration to compute the statistics needed for quantization. This guide explains how observers fit into the quantization pipeline and how to implement a custom one.

## What is an Observer?

When a quantized layer runs a calibration forward pass, it passes the weight or activation tensor to an observer. The observer accumulates **min and max values** (`min_vals`, `max_vals`) that characterize the tensor's range. These statistics are later used to compute quantization parameters (`scale`, `zero_point`, and optionally `global_scale`) via `compressed-tensors`.

Observers work in two phases:

1. **Observe**: `forward()` reshapes the tensor and calls `update_statistics()` to accumulate `min_vals`/`max_vals`
2. **Compute**: `get_qparams()` passes the accumulated statistics to `calculate_qparams` (and `generate_gparam` for FP4 schemes) to produce a `QParamsDict`

This separation allows fused layers (Q/K/V, gate/up) to accumulate statistics independently, then compute a shared `global_scale` from the combined statistics of all fused observers.

The base `Observer` class handles all slicing and reshaping for group-wise, channel-wise, and token-wise strategies before calling your subclass. Your subclass only needs to answer: **given this pre-shaped tensor, how should `min_vals` and `max_vals` be updated?**

## The Observer Contract

All observers subclass `llmcompressor.observers.Observer` and must implement one method:

```python
import torch
from llmcompressor.observers import Observer

@Observer.register("my_observer")
class MyObserver(Observer):

    def update_statistics(self, observed: torch.Tensor) -> None:
        """
        Update internal min_vals and max_vals from the observed tensor.

        The base class has already reshaped the tensor into
        shape (num_observations, *qparam_shape, group_size).
        Set self.min_vals and self.max_vals with shape (*qparam_shape,).

        :param observed: pre-processed tensor ready for statistics computation
        """
        self.min_vals = torch.amin(observed, dim=(0, -1))
        self.max_vals = torch.amax(observed, dim=(0, -1))
```

The `@Observer.register("my_observer")` decorator registers your observer under the given name so it can be referenced in recipes by string.

## How the Base Class Uses Your Statistics

The base class `compute_qparams_from_statistics` method reads your `min_vals`/`max_vals` and passes them to `calculate_qparams` from `compressed-tensors`:

```python
# Inside Observer.compute_qparams_from_statistics (simplified):
scale, zero_point = calculate_qparams(
    min_vals=self.min_vals,
    max_vals=self.max_vals,
    quantization_args=self.args,
    global_scale=global_scale,
)
return {"scale": scale, "zero_point": zero_point, "global_scale": global_scale}
```

`calculate_qparams` handles the actual scale and zero point computation — symmetric vs asymmetric quantization, dtype clamping, MX scale generation, and so on. Your observer only controls the min/max values fed into it.

For TENSOR_GROUP strategies (FP4 schemes), `global_scale` is computed automatically from the combined statistics of all fused observers before being passed to `calculate_qparams`. You do not need to handle this in your subclass.

## DDP Synchronization

If your observer accumulates statistics across batches and will be used for activation quantization in DDP, declare which attributes need synchronization via `_act_sync_dict`:

```python
from torch import distributed as dist

@Observer.register("my_observer")
class MyObserver(Observer):
    _act_sync_dict = {
        "min_vals": dist.ReduceOp.MIN,
        "max_vals": dist.ReduceOp.MAX,
    }

    def update_statistics(self, observed: torch.Tensor) -> None:
        ...
```

The base class `sync_activation_stats()` iterates this dict and issues async all-reduce operations. Memoryless observers that overwrite statistics each batch should set `_act_sync_dict = {}`.

## Stateful Observers

Some observers accumulate statistics across multiple calibration batches. To do this, check for existing state in `update_statistics`:

```python
@Observer.register("my_observer")
class MyObserver(Observer):

    _act_sync_dict = {
        "min_vals": dist.ReduceOp.MIN,
        "max_vals": dist.ReduceOp.MAX,
    }

    def update_statistics(self, observed: torch.Tensor) -> None:
        min_vals = torch.amin(observed, dim=(0, -1))
        max_vals = torch.amax(observed, dim=(0, -1))

        if hasattr(self, "min_vals"):
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

    def update_statistics(self, observed: torch.Tensor) -> None:
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

- **Implement `update_statistics`, not scale/zero_point computation.** Your subclass sets `self.min_vals` and `self.max_vals`. The base class handles converting those to `scale`, `zero_point`, and `global_scale` via `compressed-tensors`.
- **`update_statistics` receives a pre-shaped tensor.** The base class has already sliced the input according to `QuantizationArgs.strategy` (group, channel, token, etc.). You do not need to handle reshaping yourself.
- **`global_scale` is handled automatically.** For TENSOR_GROUP strategies, `compute_qparams_from_statistics()` computes `global_scale` from the combined `min_vals`/`max_vals` of all fused observers. You do not need to implement any global scale logic.
- **Declare `_act_sync_dict` for DDP.** If your observer accumulates state across batches, declare which attributes need all-reduce and with what operation. Memoryless observers should set `_act_sync_dict = {}`.
- **`observer_kwargs` is the right place for hyperparameters.** Access them via `self.args.observer_kwargs.get(...)`.
- **Match the shape contract.** `update_statistics` must set `self.min_vals` and `self.max_vals` to tensors of shape `(*qparam_shape,)` — one scalar per quantization group/channel/token.
- **Existing observers are good references.** See `min_max.py` for a simple stateless example and `mse.py` for a more complex stateful one.
