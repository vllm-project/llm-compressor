# Adding a New Observer

Observers compute the quantization parameters — `scale` and `zero_point` — from observed tensors during calibration. This tutorial explains how observers fit into the quantization pipeline and how to implement a custom one.

## What is an Observer?

When a quantized layer runs a calibration forward pass, it passes the weight or activation tensor to an observer. The observer analyzes the tensor and returns `(scale, zero_point)` values used to quantize that tensor.

The base class (`Observer`) handles all the mechanics of group-wise, channel-wise, and token-wise quantization — slicing the tensor appropriately and calling `calculate_qparams`. Your subclass only needs to implement one thing: given a prepared tensor, compute the min and max values.

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

        The tensor has already been reshaped by the base class into
        shape (num_observations, *qparam_shape, group_size).

        :param observed: pre-processed tensor ready for statistics computation
        :return: (min_vals, max_vals) with shape (*qparam_shape,)
        """
        ...

    def get_global_min_max(self, observed: torch.Tensor) -> MinMaxTuple:
        """
        Compute global min and max for global scale calculation (e.g., MXFP4).

        The tensor has already been reshaped to (1, 1, total_elements).

        :param observed: per-tensor reshaped tensor
        :return: (min_val, max_val) scalar tensors
        """
        ...
```

The `@Observer.register("my_observer")` decorator registers your observer under the given name so it can be referenced in recipes by string.

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

The following observer clips outliers by computing quantization parameters from a configurable percentile range instead of the full min/max. This can improve accuracy when the tensor has extreme outlier values.

```python
import torch
from llmcompressor.observers import Observer
from llmcompressor.observers.base import MinMaxTuple

@Observer.register("percentile")
class PercentileObserver(Observer):
    """
    Computes quantization parameters by clipping to a symmetric percentile
    range, discarding outlier values beyond the given percentile.

    Configure via observer_kwargs:
        percentile (float): the upper percentile to retain, e.g. 99.9
    """

    def get_min_max(self, observed: torch.Tensor) -> MinMaxTuple:
        percentile = self.args.observer_kwargs.get("percentile", 99.9)
        lower = 100.0 - percentile
        upper = percentile

        min_vals = torch.tensor(
            [
                torch.quantile(observed[..., i], lower / 100.0).item()
                for i in range(observed.shape[-2])
            ]
        )
        max_vals = torch.tensor(
            [
                torch.quantile(observed[..., i], upper / 100.0).item()
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

- **`get_min_max` receives a pre-shaped tensor.** The base class has already sliced the input into groups, channels, or tokens according to `QuantizationArgs.strategy`. You do not need to handle reshaping yourself.
- **`observer_kwargs` is the right place for hyperparameters.** Access them via `self.args.observer_kwargs.get(...)`.
- **Match the shape contract.** `get_min_max` must return tensors of shape `(*qparam_shape,)` — one scalar per quantization group/channel/token. `get_global_min_max` must return shape `(1,)`.
- **Existing observers are good references.** See `min_max.py` for a simple stateless example and `mse.py` for a more complex stateful one.
