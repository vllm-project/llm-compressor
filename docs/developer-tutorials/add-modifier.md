# Adding a New Modifier

Modifiers are the core extension point in LLM Compressor. Each compression algorithm — GPTQ, AWQ, SmoothQuant, and others — is implemented as a modifier. This tutorial walks through the modifier contract, lifecycle, and how to implement a custom one.

## What is a Modifier?

A modifier is a Pydantic model that hooks into the compression pipeline at well-defined lifecycle points. When you call `oneshot`, LLM Compressor:

1. Instantiates modifiers from the recipe
2. Calls `initialize` on each modifier
3. Runs calibration batches, firing `Event`s that modifiers respond to
4. Calls `finalize` on each modifier

Modifiers express what they want to do at each stage by overriding lifecycle hooks.

## How Events Work

Not all lifecycle hooks are driven by events. `on_initialize` and `on_finalize` are called directly by `CompressionLifecycle` — before and after the pipeline runs respectively. Everything in between is event-driven.

| Hook | Called by |
|------|-----------|
| `on_initialize` | `CompressionLifecycle.initialize()` |
| `on_event` / `on_start` / `on_update` / `on_end` | `CompressionLifecycle.event()` → `Modifier.update_event()` |
| `on_finalize` | `CompressionLifecycle.finalize()` |

The pipeline fires events by calling methods on `LifecycleCallbacks` (aliased as `callbacks`), which routes them through the active session into `CompressionLifecycle.event()`. Modifiers never fire events themselves — they only react to them.

## The Modifier Contract

All modifiers subclass `llmcompressor.modifiers.Modifier` and must implement `on_initialize`. All other hooks are optional.

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
        # Called when calibration starts.
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
| `on_start` | First `BATCH_START` event (base class); most modifiers call it manually from `on_event` on `CALIBRATION_EPOCH_START` | No |
| `on_update` | Every event while active (between `on_start` and `on_end`); rarely used outside of pruning modifiers | No |
| `on_end` | `BATCH_END` when modifier ends (base class); in practice all modifiers call it manually from `on_event` on `CALIBRATION_EPOCH_END` | No |
| `on_finalize` | Once, after calibration | No |

> **Note on `on_start` / `on_end` vs `on_event`:** The base class dispatches `on_start` on the first `BATCH_START` event and `on_end` on `BATCH_END`. However, all built-in modifiers (GPTQ, AWQ, SmoothQuant, SparseGPT, etc.) bypass this by overriding `on_event` and calling `self.on_start()` / `self.on_end()` themselves on `CALIBRATION_EPOCH_START` / `CALIBRATION_EPOCH_END`. If you are writing a new modifier, follow this pattern.

### The `State` Object

`state.model` gives you the `torch.nn.Module` being compressed. This is the primary object you will interact with in most hooks.

### Pydantic Parameters

Because `Modifier` is a Pydantic model with `extra="forbid"`, all parameters must be declared as class-level fields. This also means your modifier can be instantiated directly in Python or from a YAML recipe.

```python
class MyModifier(Modifier):
    targets: list[str] = ["Linear"]
    scale_factor: float = 0.5
    ignore: list[str] = []
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
        # Runs after every forward pass through this module
        ...
```

## Example: A Weight-Clamping Modifier

The following modifier clamps the stored weight tensors of all `Linear` layers to a fixed absolute magnitude. It follows the real modifier pattern by handling both pipelines in `on_event`: when using the sequential pipeline it clamps weights layer-by-layer as each subgraph completes (`SEQUENTIAL_EPOCH_END`), and when using the basic pipeline it clamps all weights at once at the end of calibration (`CALIBRATION_EPOCH_END`).

```python
import torch
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
    targets: list[str] = ["Linear"]
    ignore: list[str] = []

    def on_initialize(self, state: State, **kwargs) -> bool:
        if self.max_weight_magnitude <= 0:
            raise ValueError("max_weight_magnitude must be positive")

        # Verify that at least one target module exists in the model
        matched = list(match_named_modules(state.model, self.targets, self.ignore))
        if not matched:
            raise ValueError(
                f"No modules matched targets={self.targets} ignore={self.ignore}"
            )

        self._clamped: set[str] = set()
        return True

    def on_event(self, state: State, event: Event, **kwargs):
        if event.type_ == EventType.CALIBRATION_EPOCH_START:
            if not self.started_:
                self.on_start(state, event)

        elif event.type_ == EventType.SEQUENTIAL_EPOCH_END:
            # Sequential pipeline: clamp weights for the just-finished subgraph
            subgraph = kwargs.get("subgraph")
            if subgraph is not None:
                self._clamp_modules(state, modules=subgraph.modules())

        elif event.type_ == EventType.CALIBRATION_EPOCH_END:
            # Basic pipeline: clamp any modules not yet handled
            self._clamp_modules(state)
            if not self.ended_:
                self.on_end(state, event)

    def on_start(self, state: State, event: Event, **kwargs):
        self.started_ = True

    def on_end(self, state: State, event: Event, **kwargs):
        self.ended_ = True

    def _clamp_modules(self, state: State, modules=None):
        for name, module in match_named_modules(
            state.model, self.targets, self.ignore
        ):
            if name in self._clamped:
                continue
            if modules is not None and module not in modules:
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

model = AutoModelForCausalLM.from_pretrained("your-model")
tokenizer = AutoTokenizer.from_pretrained("your-model")

oneshot(
    model=model,
    recipe=[WeightClampModifier(max_weight_magnitude=0.5, targets=["Linear"], ignore=["lm_head"])],
)

model.save_pretrained("your-model-clamped", save_compressed=True)
tokenizer.save_pretrained("your-model-clamped")
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
- **Prefer `on_event` over `on_start` for epoch-level control.** Most modifiers override `on_event` and call `self.on_start()` manually on `CALIBRATION_EPOCH_START` rather than relying on the base class `BATCH_START` dispatch.
- **`on_update` is rarely needed.** Only override it if you need a per-batch callback while the modifier is active — e.g., `MagnitudeModifier` uses it to update sparsity each batch. Compression modifiers (GPTQ, AWQ, SmoothQuant, etc.) do not use it.
- **Modifiers never fire events — the pipeline does.** All lifecycle events (`CALIBRATION_EPOCH_START`, `BATCH_START`, `SEQUENTIAL_EPOCH_END`, etc.) are fired by the calibration pipeline. Your modifier only reacts to them. The sequential pipeline additionally fires `SEQUENTIAL_EPOCH_END` between layer groups, which modifiers like GPTQ and SparseGPT use to trigger compression layer-by-layer.
