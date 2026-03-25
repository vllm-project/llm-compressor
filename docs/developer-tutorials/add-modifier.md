# Adding a New Modifier

Modifiers are the core extension point in LLM Compressor. Each compression algorithm — GPTQ, AWQ, SmoothQuant, and others — is implemented as a modifier. This tutorial walks through the modifier contract, lifecycle, and how to implement a custom one.

## What is a Modifier?

A modifier is a Pydantic model that hooks into the compression pipeline at well-defined lifecycle points. When you call `oneshot`, LLM Compressor:

1. Instantiates modifiers from the recipe
2. Calls `initialize` on each modifier
3. Runs calibration batches, firing `Event`s that modifiers respond to
4. Calls `finalize` on each modifier

Modifiers express what they want to do at each stage by overriding lifecycle hooks.

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
        # Called when calibration starts (first BATCH_START event).
        ...

    def on_update(self, state: State, event: Event, **kwargs):
        # Called on every event while the modifier is active.
        ...

    def on_end(self, state: State, event: Event, **kwargs):
        # Called when calibration ends.
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
| `on_start` | First `BATCH_START` event | No |
| `on_update` | Every event while active | No |
| `on_end` | `BATCH_END` when modifier ends | No |
| `on_finalize` | Once, after calibration | No |

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

The following modifier clamps the absolute magnitude of all `Linear` layer weights during `on_finalize`, after calibration is complete.

```python
import torch
from compressed_tensors.utils import match_named_modules
from llmcompressor.modifiers import Modifier
from llmcompressor.core import State

class WeightClampModifier(Modifier):
    """
    Clamps the magnitude of Linear layer weights to a maximum absolute value.

    :param max_weight_magnitude: maximum allowed absolute weight value
    :param targets: module types to target
    :param ignore: module names to skip
    """

    max_weight_magnitude: float = 1.0
    targets: list[str] = ["Linear"]
    ignore: list[str] = []

    def on_initialize(self, state: State, **kwargs) -> bool:
        return True

    def on_finalize(self, state: State, **kwargs) -> bool:
        for name, module in match_named_modules(
            state.model, self.targets, self.ignore
        ):
            with torch.no_grad():
                module.weight.clamp_(
                    -self.max_weight_magnitude,
                    self.max_weight_magnitude,
                )
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
    recipe=[WeightClampModifier(max_weight_magnitude=0.5, ignore=["lm_head"])],
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
