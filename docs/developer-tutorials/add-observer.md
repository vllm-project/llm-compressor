# Adding a New Observer

Observers analyze weight and activation tensors during calibration to compute the statistics needed for quantization. This guide explains how observers fit into the quantization pipeline and how to implement a custom one.

## What is an Observer?

When a quantized layer runs a calibration forward pass, it passes the weight or activation tensor to an observer. The observer's job is to compute **min and max values** from the observed tensor. These min/max values are then passed to `compressed_tensors.quantization.utils.calculate_qparams`, which converts them into `scale` and `zero_point` tensors used for quantization.

Observers do **not** compute scales or zero points directly — that responsibility belongs to `compressed-tensors`. The observer's only job is to characterize the tensor's range via min and max values.

For schemes that require a global scale (e.g., NVFP4, MXFP4), the observer's `get_global_min_max` output is similarly passed to `compressed_tensors.quantization.utils.generate_gparam`, which generates the global scale used to keep per-group local scales within a target dtype range (e.g., FP8 for NVFP4 group scales).

The base `Observer` class handles all slicing and reshaping for group-wise, channel-wise, and token-wise strategies before calling your subclass. Your subclass only needs to answer: **given this pre-shaped tensor, what are the min and max values?**

## The Observer Contract

All observers subclass `llmcompressor.observers.Observer` and must implement two methods:

```python
import torch
from llmcompressor.observers import Observer
from llmcompressor.observers.base import MinMaxTuple

@Observer.register("my_observer")
class MyObserver(Observer):

    def get_min_max(self, observed: torch.Tensor) -> MinMaxTuple:
        """
        Compute min and max from the observed tensor.

        The base class has already reshaped the tensor into
        shape (num_observations, *qparam_shape, group_size).
        These min/max values are passed to calculate_qparams
        in compressed-tensors to produce scale and zero_point.

        :param observed: pre-processed tensor ready for statistics computation
        :return: (min_vals, max_vals) with shape (*qparam_shape,)
        """
        ...

    def get_global_min_max(self, observed: torch.Tensor) -> MinMaxTuple:
        """
        Compute global min and max for global scale calculation (e.g., NVFP4, MXFP4).

        The base class reshapes the tensor to (num_observations, 1, group_size) before
        calling this method. The returned values are passed to generate_gparam in
        compressed-tensors to produce a global scale that keeps per-group local
        scales within the target dtype range.

        :param observed: per-tensor reshaped tensor
        :return: (min_val, max_val) scalar tensors of shape (1,)
        """
        ...
```

The `@Observer.register("my_observer")` decorator registers your observer under the given name so it can be referenced in recipes by string.

## How the Base Class Uses Your Output

The base class `_forward_with_minmax` method calls your `get_min_max` and passes the result directly to `calculate_qparams` from `compressed-tensors`:

```python
# Inside Observer._forward_with_minmax (simplified):
min_vals, max_vals = self.get_min_max(observed)
scales, zero_points = calculate_qparams(
    min_vals=min_vals,
    max_vals=max_vals,
    quantization_args=self.args,
    global_scale=global_scale,
)
```

`calculate_qparams` handles the actual scale and zero point computation — symmetric vs asymmetric quantization, dtype clamping, MX scale generation, and so on. Your observer only controls the min/max values fed into it.

For global scales (FP4 schemes), the base class calls your `get_global_min_max` and passes the result to `generate_gparam`:

```python
# Inside Observer._get_global_scale_with_minmax (simplified):
global_min_vals, global_max_vals = self.get_global_min_max(observed)
global_scale = generate_gparam(global_min_vals, global_max_vals)
```

## Stateful Observers

Some observers accumulate statistics across multiple calibration batches. To do this, initialize state in `__init__` and update it in `get_min_max`:

```python
@Observer.register("my_observer")
class MyObserver(Observer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.running_min = None
        self.running_max = None

    def get_min_max(self, observed: torch.Tensor) -> MinMaxTuple:
        min_vals = torch.amin(observed, dim=(0, -1))
        max_vals = torch.amax(observed, dim=(0, -1))

        if self.running_min is not None:
            min_vals = torch.min(min_vals, self.running_min)
            max_vals = torch.max(max_vals, self.running_max)

        self.running_min = min_vals
        self.running_max = max_vals

        return min_vals, max_vals
```

## Example: A Percentile-Clipping Observer

The following observer clips outliers by returning min/max values from a configurable percentile range rather than the absolute extremes. This can improve accuracy when tensors have extreme outlier values that would otherwise inflate the quantization range.

```python
import torch
from llmcompressor.observers import Observer
from llmcompressor.observers.base import MinMaxTuple

@Observer.register("percentile")
class PercentileObserver(Observer):
    """
    Returns per-channel min/max values clipped to a configurable percentile
    range, discarding outliers beyond the given percentile. The resulting
    min/max values are passed to calculate_qparams in compressed-tensors
    to produce scale and zero_point.

    Configure via observer_kwargs:
        percentile (float): the upper percentile to retain, e.g. 99.9
    """

    def get_min_max(self, observed: torch.Tensor) -> MinMaxTuple:
        percentile = self.args.observer_kwargs.get("percentile", 99.9)
        lower = 100.0 - percentile
        upper = percentile

        min_vals = torch.tensor(
            [
                torch.quantile(observed[:, i, :].flatten(), lower / 100.0).item()
                for i in range(observed.shape[-2])
            ]
        )
        max_vals = torch.tensor(
            [
                torch.quantile(observed[:, i, :].flatten(), upper / 100.0).item()
                for i in range(observed.shape[-2])
            ]
        )

        return min_vals, max_vals

    def get_global_min_max(self, observed: torch.Tensor) -> MinMaxTuple:
        percentile = self.args.observer_kwargs.get("percentile", 99.9)
        lower = 100.0 - percentile
        flat = observed.flatten()
        min_val = torch.quantile(flat, lower / 100.0).unsqueeze(0)
        max_val = torch.quantile(flat, percentile / 100.0).unsqueeze(0)
        return min_val, max_val
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

- **Observers return min/max, not scale/zero_point.** The conversion from min/max → scale/zero_point is handled by `calculate_qparams` in `compressed-tensors`. Focus your implementation on accurately characterizing the tensor range.
- **`get_min_max` receives a pre-shaped tensor.** The base class has already sliced the input according to `QuantizationArgs.strategy` (group, channel, token, etc.). You do not need to handle reshaping yourself.
- **`get_global_min_max` is only used for FP4 schemes** (NVFP4, MXFP4) that require a global scale. For standard int8/fp8 quantization, the base class will not call it.
- **`observer_kwargs` is the right place for hyperparameters.** Access them via `self.args.observer_kwargs.get(...)`.
- **Match the shape contract.** `get_min_max` must return tensors of shape `(*qparam_shape,)` — one scalar per quantization group/channel/token. `get_global_min_max` must return shape `(1,)`.
- **Existing observers are good references.** See `min_max.py` for a simple stateless example and `mse.py` for a more complex stateful one.
