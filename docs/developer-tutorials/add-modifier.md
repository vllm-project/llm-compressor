# Adding a New Modifier

Modifiers are the core extension point in LLM Compressor. Each compression algorithm — GPTQ, AWQ, SmoothQuant, and others — is implemented as a modifier. This tutorial walks through the modifier contract, lifecycle, and how to implement a custom modifier.

## What is a Modifier?

A modifier is a subclass of the `Modifier` base class that hooks into the compression pipeline at well-defined lifecycle points. When you call `oneshot`, LLM Compressor:

1. Instantiate modifiers from the recipe
2. Call `initialize` on each modifier, which calls `on_initialize`
3. For each calibration pipeline:
    * Dispatch the model
    * For each calibration epoch:
        - Fire `CALIBRATION_START` event, calling `on_calibration_start`
        - Run calibration forward passes (quantization disabled)
        - Fire `SEQUENTIAL_EPOCH_END` event after each layer group (sequential pipeline) or once for the entire model (basic/data-free pipelines), calling `on_sequential_epoch_end`
        - Fire `CALIBRATION_END` event, calling `on_calibration_end`
4. Call `finalize` on each modifier, which calls `on_finalize`

Modifiers express what they want to do at each stage by implementing lifecycle hooks.

## The Modifier Contract

All modifiers subclass `llmcompressor.modifiers.Modifier` and must implement `on_initialize`. All other lifecycle hooks are optional.

```python
import torch

from llmcompressor.modifiers import Modifier
from llmcompressor.core import State, Event

class MyModifier(Modifier):
    # Pydantic fields — declare your parameters here
    my_param: float = 1.0

    # Set to True if this modifier needs calibration data
    requires_calibration_data: bool = False

    def on_initialize(self, state: State, **kwargs) -> bool:
        # Called once before calibration begins.
        # Set up hooks, attach attributes to modules, etc.
        # Return True if initialization succeeded.
        ...
        return True

    def on_finalize(self, state: State, **kwargs) -> bool:
        # Called after calibration completes.
        # Clean up hooks, apply final transformations, etc.
        # Return True if finalization succeeded.
        ...
        return True

    def on_event(self, state: State, event: Event, **kwargs):
        # Called on every event, unconditionally, before lifecycle events are
        # dispatched. Override to respond to custom event types or to implement
        # cross-cutting behavior.
        ...

    ## Training lifecycle events ##

    def on_start(self, state: State, event: Event, **kwargs):
        # Called when the modifier starts based on the `start` parameter.
        # The base class automatically dispatches this when `start <= event.current_index`.
        # For training scenarios with explicit start/end steps.
        ...

    def on_update(self, state: State, event: Event, **kwargs):
        # Called on every event while the modifier is active (between on_start and
        # on_end). Rarely needed — only useful for per-batch callbacks such as
        # dynamic pruning schedules. Compression modifiers (GPTQ, AWQ, etc.) do
        # not use this hook.
        ...

    def on_end(self, state: State, event: Event, **kwargs):
        # Called when the modifier ends based on the `end` parameter.
        # The base class automatically dispatches this when `end >= event.current_index`.
        # For training scenarios with explicit start/end steps.
        ...

    ## Calibration lifecycle events ##

    def on_calibration_start(self, state: State, event: Event, **kwargs):
        # Called at the start of each calibration epoch.
        # This is where most compression modifiers initialize their state for
        # the calibration pass (e.g., set up observers, reset statistics).
        ...

    def on_sequential_epoch_end(
        self, state: State, event: Event, modules: list[torch.nn.Module], **kwargs
    ):
        # Called at the end of a sequential layer group (sequential pipeline) or
        # once for the entire model (basic/data-free pipelines).
        # This is where quantization modifiers compute weight and activation
        # quantization parameters.
        ...

    def on_calibration_end(self, state: State, event: Event, **kwargs):
        # Called at the end of each calibration epoch.
        # This is where most compression modifiers finalize their calibration
        # (e.g., apply compression, enable quantization).
        ...
```

### Lifecycle Summary

| Hook | When it runs | Required |
|------|-------------|----------|
| `on_initialize` | Once, during `initialize()`, before calibration | Yes |
| `on_finalize` | Once, during `finalize()`, after all pipelines complete | No |
| `on_event` | Every event, unconditionally (before lifecycle dispatch) | No |
| `requires_calibration_data` | Bool field queried during pipeline selection to decide sequential vs data-free | No (defaults to `False`) |
| `on_calibration_start` | At the start of each calibration epoch (fired by `CALIBRATION_START` event) | No |
| `on_sequential_epoch_end` | After each layer group (sequential) or once for entire model (basic/data-free) (fired by `SEQUENTIAL_EPOCH_END` event) | No |
| `on_calibration_end` | At the end of each calibration epoch (fired by `CALIBRATION_END` event) | No |
| `on_start` | When `start <= event.current_index` (training lifecycle, auto-dispatched) | No |
| `on_update` | Every event while active (between `on_start` and `on_end`); rarely used | No |
| `on_end` | When `end >= event.current_index` (training lifecycle, auto-dispatched) | No |

> **Note on calibration vs training lifecycle:** The base `Modifier` class now provides first-class support for both calibration and training lifecycles:
> - **Calibration lifecycle** (`on_calibration_start`, `on_sequential_epoch_end`, `on_calibration_end`): These hooks are automatically dispatched when the corresponding event types are fired by the calibration pipeline. All compression modifiers (GPTQ, AWQ, SmoothQuant, SparseGPT, etc.) use these hooks.
> - **Training lifecycle** (`on_start`, `on_update`, `on_end`): These hooks are auto-dispatched based on the `start` and `end` parameters. They are primarily used for training scenarios with explicit step ranges (e.g., dynamic pruning schedules).
> - `on_event`: Called on every event before any lifecycle dispatch. Use this for cross-cutting behavior or custom event types.
>
> **Note on `SEQUENTIAL_EPOCH_END`:** This event is fired by **all** pipelines (sequential, basic, and data-free), not just the sequential pipeline. For the sequential pipeline, it fires after each layer group with a subgraph scoped to that group. For basic and data-free pipelines, it fires once with a subgraph covering the entire model. Built-in quantization modifiers (QuantizationModifier, GPTQModifier) use this event to compute weight and activation quantization parameters — DDP synchronization of activation statistics, weight observation, and qparam computation (for both activations and weights) all happen here.
>
> **Quantization is disabled during calibration.** All pipelines disable quantization during calibration forward passes via `DisableQuantization`. This means calibration hooks see unquantized activations. Quantization parameters are computed at `SEQUENTIAL_EPOCH_END` and quantization is enabled at `CALIBRATION_END`.

### The `State` Object

`state.model` gives you the `torch.nn.Module` being compressed. This is the primary object you will interact with in most hooks.

### Pydantic Parameters

The `Modifier` base class is a subclass of the Pydantic `BaseModel` class, meaning that all algorithm parameters are declared as class-level fields. This structure allows modifiers to be instantiated directly as python objects or loaded a YAML recipe.

```python
from pydantic import Field

class MyModifier(Modifier):
    targets: list[str] = Field(default_factory=lambda: ["Linear"])
    scale_factor: float = 0.5
    ignore: list[str] = Field(default_factory=list)
```

## Attaching Hooks with `HooksMixin`

`Modifier` inherits from `HooksMixin`, which provides a managed way to register PyTorch forward hooks. Hooks registered through `HooksMixin` are automatically removed when `finalize` is called.

```python
from llmcompressor.modifiers import Modifier
from llmcompressor.core import State

class MyModifier(Modifier):
    def on_initialize(self, state: State, **kwargs) -> bool:
        for name, module in state.model.named_modules():
            if "Linear" in type(module).__name__:
                self.register_hook(
                    module,
                    self._forward_hook,
                    "forward",
                )
        return True

    def _forward_hook(self, module, inputs, output):
        """
        Runs after every forward pass through this module
        Using the `HooksMixin` interface ensures that your modifiers hooks are enabled during calibration passes through the model and disabled during propagation passes through the model. See [Sequential Pipeline](src/llmcompressor/pipelines/sequential/pipeline.py) for more information.
        """
        ...
```

## Example: A Weight-Clamping Modifier

The following modifier clamps the stored weight tensors of all `Linear` layers to a fixed absolute magnitude after calibration completes (`CALIBRATION_END`).

```python
import torch
from pydantic import Field, PrivateAttr
from compressed_tensors.utils import match_named_modules
from llmcompressor.modifiers import Modifier
from llmcompressor.core import State, Event, EventType

class WeightClampModifier(Modifier):
    """
    Clamps the magnitude of Linear layer weight tensors to a maximum absolute
    value. Applied layer-by-layer on SEQUENTIAL_EPOCH_END (sequential pipeline)
    or all at once on CALIBRATION_END (basic pipeline).

    :param max_weight_magnitude: maximum allowed absolute value for any weight
    :param targets: module types to target
    :param ignore: module names to skip
    """

    max_weight_magnitude: float = 1.0
    targets: list[str] = Field(default_factory=lambda: ["Linear"])
    ignore: list[str] = Field(default_factory=list)
    _clamped: set[str] = PrivateAttr(default_factory=set)

    def on_initialize(self, state: State, **kwargs) -> bool:
        if self.max_weight_magnitude <= 0:
            raise ValueError("max_weight_magnitude must be positive")

        # Verify that at least one target module exists in the model
        matched = list(match_named_modules(state.model, self.targets, self.ignore))
        if not matched:
            raise ValueError(
                f"No modules matched targets={self.targets} ignore={self.ignore}"
            )
        return True

    def on_calibration_end(self, state: State, event: Event, **kwargs):
        # Clamp all target modules at the end of calibration
        for name, module in match_named_modules(
            state.model, self.targets, self.ignore
        ):
            if name in self._clamped:
                continue
            with torch.no_grad():
                module.weight.clamp_(
                    -self.max_weight_magnitude,
                    self.max_weight_magnitude,
                )
            self._clamped.add(name)

    def on_finalize(self, state: State, **kwargs) -> bool:
        self._clamped.clear()
        return True
```

### Using the Modifier with `oneshot`

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

oneshot(
    model=model,
    recipe=[WeightClampModifier(max_weight_magnitude=0.5, targets=["Linear"], ignore=["lm_head"])],
)

model.save_pretrained("Qwen3-0.6B-clamped")
tokenizer.save_pretrained("Qwen3-0.6B-clamped")
```

### Using the Modifier from a YAML Recipe

```yaml
weight_clamp_stage:
  weight_clamp_modifiers:
    WeightClampModifier:
      max_weight_magnitude: 0.5
      targets:
        - Linear
      ignore:
        - lm_head

```

## Tips

- **Only override what you need.** The default implementations of all lifecycle hooks except `on_initialize` are no-ops or return `True` — you do not need to call `super()` for these.
- **Use `match_named_modules`** (from `compressed_tensors.utils`) to filter modules by type name or path pattern, consistent with how other modifiers handle `targets` and `ignore`.
- **Keep `on_initialize` lightweight.** Expensive operations (e.g., full-model passes) should be deferred to calibration lifecycle hooks or `on_finalize`.
- **`on_update` is rarely needed.** Only override it if you need a per-batch callback while the modifier is active — e.g., `MagnitudeModifier` uses it to update sparsity each batch. Compression modifiers (GPTQ, AWQ, SmoothQuant, etc.) do not use it.
- **Modifiers never fire events — the pipeline does.** All lifecycle events (`CALIBRATION_START`, `SEQUENTIAL_EPOCH_END`, `CALIBRATION_END`) are fired by the calibration pipeline. Your modifier only reacts to them by implementing the corresponding hooks. The base `Modifier` class automatically routes these events to the appropriate lifecycle methods.
- **All calibration pipelines fire `SEQUENTIAL_EPOCH_END`.** The sequential pipeline fires it between layer groups (scoped to each group), while the basic and data-free pipelines fire it once for the entire model. Modifiers like GPTQ, SparseGPT, and QuantizationModifier use this event to trigger compression and qparam computation.
