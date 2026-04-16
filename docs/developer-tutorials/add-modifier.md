# Adding a New Modifier

Modifiers are the core extension point in LLM Compressor. Each compression algorithm — GPTQ, AWQ, SmoothQuant, and others — is implemented as a modifier. This tutorial walks through the modifier contract, lifecycle, and how to implement a custom modifier.

## What is a Modifier?

A modifier is a subclass of the `Modifier` base class that hooks into the compression pipeline at well-defined lifecycle points. When you call `oneshot`, LLM Compressor:

1. Instantiate modifiers from the recipe
2. Call `on_initialize` on each modifier
3. For each pipeline:
    * Dispatch the model, then call `on_start` for each modifier in the pipeline
    * Calibrate the model with calibration data, triggering the `calibration_epoch_start`, `sequential_epoch_end`, and `calibration_epoch_end` events for each modifier in the pipeline
4. Call `on_end` on each modifier in the pipeline
5. Call `on_finalize` on each modifier once all pipelines have finished

Modifiers express what they want to do at each stage by implementing lifecycle hooks.

## The Modifier Contract

All modifiers subclass `llmcompressor.modifiers.Modifier` and must implement `on_initialize`. All other lifecycle hooks are optional.

```python
from llmcompressor.modifiers import Modifier
from llmcompressor.core import State, Event

class MyModifier(Modifier):
    # Pydantic fields — declare your parameters here
    my_param: float = 1.0

    def on_initialize(self, state: State, **kwargs) -> bool:
        # Called once before calibration begins.
        # Set up hooks, attach attributes to modules, etc.
        # Return True if initialization succeeded.
        ...
        return True

    def on_start(self, state: State, event: Event, **kwargs):
        # Called when calibration starts for each calibration pipeline this modifier is a part of.
        # The base class dispatches this on the first BATCH_START event, but in
        # practice most modifiers trigger it themselves from on_event by checking
        # for CALIBRATION_EPOCH_START (see note below).
        ...

    def on_update(self, state: State, event: Event, **kwargs):
        # Called on every event while the modifier is active (between on_start and
        # on_end). Rarely needed — only useful for per-batch callbacks such as
        # dynamic pruning schedules. Compression modifiers (GPTQ, AWQ, etc.) do
        # not use this hook.
        ...

    def on_end(self, state: State, event: Event, **kwargs):
        # Called when calibration ends.
        # The base class dispatches this on BATCH_END, but in practice all
        # modifiers call it manually from on_event on CALIBRATION_EPOCH_END.
        ...

    def on_event(self, state: State, event: Event, **kwargs):
        # Called on every event, unconditionally, before on_start/on_update/on_end
        # dispatch. Override to respond to specific EventTypes such as
        # CALIBRATION_EPOCH_START or SEQUENTIAL_EPOCH_END that fall outside
        # the BATCH_START / BATCH_END pattern.
        ...

    def on_finalize(self, state: State, **kwargs) -> bool:
        # Called after calibration completes.
        # Clean up hooks, apply final transformations, etc.
        # Return True if finalization succeeded.
        ...
        return True
```

### Lifecycle Summary

| Hook | When it runs | Required |
|------|-------------|----------|
| `on_initialize` | Once, before calibration | Yes |
| `on_event` | Every event, unconditionally (before start/update/end dispatch) | No |
| `on_start` | Most modifiers call it manually from `on_event` on `CALIBRATION_EPOCH_START` | No |
| `on_update` | Every event while active (between `on_start` and `on_end`); rarely used outside of pruning modifiers | No |
| `on_end` | In practice all modifiers call it manually from `on_event` on `CALIBRATION_EPOCH_END` | No |
| `on_finalize` | Once, after calibration | No |

> **Note on `on_start` / `on_end` vs `on_event`:** The base class dispatches `on_start` on the first `BATCH_START` event and `on_end` on `BATCH_END`. However, all built-in modifiers (GPTQ, AWQ, SmoothQuant, SparseGPT, etc.) bypass this by overriding `on_event` and calling `self.on_start()` / `self.on_end()` themselves on `CALIBRATION_EPOCH_START` / `CALIBRATION_EPOCH_END`. If you are writing a new modifier, follow this pattern.

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

The following modifier clamps the stored weight tensors of all `Linear` layers to a fixed absolute magnitude after calibration completes (`CALIBRATION_EPOCH_END`).

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
    or all at once on CALIBRATION_EPOCH_END (basic pipeline).

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

    def on_event(self, state: State, event: Event, **kwargs):
        if event.type_ == EventType.CALIBRATION_EPOCH_START:
            if not self.started_:
                self.on_start(state, event)

        elif event.type_ == EventType.CALIBRATION_EPOCH_END:
            # Clamp all target modules at the end of calibration
            self._clamp_modules(state)
            if not self.ended_:
                self.on_end(state, event)

    def on_start(self, state: State, event: Event, **kwargs):
        self.started_ = True

    def on_end(self, state: State, event: Event, **kwargs):
        self.ended_ = True

    def _clamp_modules(self, state: State):
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

- **Only override what you need.** The default implementations of `on_start`, `on_update`, `on_end`, and `on_finalize` are no-ops or return `True` — you do not need to call `super()` for these.
- **Use `match_named_modules`** (from `compressed_tensors.utils`) to filter modules by type name or path pattern, consistent with how other modifiers handle `targets` and `ignore`.
- **Keep `on_initialize` lightweight.** Expensive operations (e.g., full-model passes) should be deferred to `on_start` or `on_finalize`.
- **`on_update` is rarely needed.** Only override it if you need a per-batch callback while the modifier is active — e.g., `MagnitudeModifier` uses it to update sparsity each batch. Compression modifiers (GPTQ, AWQ, SmoothQuant, etc.) do not use it.
- **Modifiers never fire events — the pipeline does.** All lifecycle events (`CALIBRATION_EPOCH_START`, `SEQUENTIAL_EPOCH_END`, etc.) are fired by the calibration pipeline. Your modifier only reacts to them. The sequential pipeline additionally fires `SEQUENTIAL_EPOCH_END` between layer groups, which modifiers like GPTQ and SparseGPT use to trigger compression layer-by-layer.
