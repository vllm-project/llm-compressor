# Adding MoE Calibration Support for a New Model

Mixture of Experts (MoE) models route each token to only a subset of expert layers. This creates a calibration problem: experts that are not activated for a given token never see calibration data, which can result in poorly calibrated quantization parameters, numerical instability, or NaNs.

LLM Compressor solves this by replacing MoE modules with calibration-friendly versions that route all tokens through all experts during calibration, while keeping only the routed expert outputs for the final result.

For background, see [Quantizing MoEs with a custom definition](../../examples/quantizing_moe/README.md#quantizing-moes-with-a-custom-definition).

## When Do You Need This?

You need a calibration module definition when:

- Quantizing with a **data-dependent algorithm** (GPTQ, AWQ, AutoRound) on an MoE model
- Quantizing with **static activation quantization** (FP8 per-tensor, INT8 per-tensor, NVFP4) on an MoE model

Simple weight-only data-free quantization (e.g., RTN W4A16) does not require calibration data and is not affected.

## The MoECalibrationModule Contract

All MoE calibration modules subclass `MoECalibrationModule` and must:

1. Be decorated with `@MoECalibrationModule.register("OriginalClassName")` where `OriginalClassName` is the exact class name of the MoE block being replaced
2. Implement `__init__(self, original, config, calibrate_all_experts=True)` accepting the original module instance
3. Implement `forward()` with the same input/output signature as the original, routing all tokens through all experts when `calibrate_all_experts=True`
4. Set `is_permanent` to control whether the module is restored after calibration

If `is_permanent=True`, the calibration module stays in place after calibration and is used for inference. This is necessary when the model's expert weights are stored in a packed format (e.g., a single 3D tensor) that must be unpacked for per-expert quantization and vLLM compatibility. If `is_permanent=False`, implement `restore(original)` to return the original module after calibration.

```python
import torch
from llmcompressor.modeling.moe_context import MoECalibrationModule


@MoECalibrationModule.register("MyModelMoE")  # exact class name from transformers
class CalibrationMyModelMoE(MoECalibrationModule):

    is_permanent = True  # stays in place for vLLM compatibility

    def __init__(self, original, config, calibrate_all_experts: bool = True):
        super().__init__()
        self.experts = ...       # unpack or copy experts from original
        self.router = original.router
        self.calibrate_all_experts = calibrate_all_experts

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...
```

## The `forward` Pattern

The key behavior difference between normal MoE routing and calibration routing:

- **Normal routing**: only tokens selected by the router run through each expert
- **Calibration routing**: all tokens run through every expert (but only the routed tokens contribute to the output)

The Llama4 pattern — where the router returns separate scores and logits and a shared expert always runs on all tokens:

```python
def forward(self, hidden_states):
    hidden_states = hidden_states.reshape(-1, self.hidden_dim)
    router_scores, router_logits = self.router(hidden_states)
    out = self.shared_expert(hidden_states)  # always runs on all tokens

    _, router_indices = torch.topk(router_logits, self.top_k, dim=1)
    expert_mask = torch.nn.functional.one_hot(
        router_indices, num_classes=self.num_experts
    ).permute(2, 1, 0)  # (num_experts, top_k, batch_size * seq_len)

    for i in range(self.num_experts):
        token_idx = torch.where(expert_mask[i].squeeze(0))

        if self.calibrate_all_experts:
            # Run ALL tokens through the expert to collect calibration statistics.
            # Only the routed tokens contribute to the output.
            expert_out = self.experts[i](hidden_states)[token_idx]
        else:
            expert_out = self.experts[i](hidden_states[token_idx])

        if len(token_idx) > 0:
            weighted_output = expert_out * router_scores[:, i][token_idx].reshape(-1, 1)
            out[token_idx] += weighted_output

    return out, router_logits
```

!!! note
    The routing scores are applied to the expert **output** rather than the input. Applying scores to the input before passing to the expert can produce NaNs during calibration.

## Example: Llama4

The existing `SequentialLlama4TextMoe` (in `src/llmcompressor/modeling/llama4.py`) is the canonical reference implementation. It registers as a replacement for `Llama4TextMoe` and handles a key Llama4-specific detail: expert weights are stored as a single packed 3D tensor (`gate_up_proj` of shape `(num_experts, hidden, 2*intermediate)`) which must be unpacked into individual `Llama4TextMLP` modules for per-expert calibration.

This is handled by a helper class `SequentialLlama4TextExperts` that converts the packed tensor into a `ModuleList` of unpacked experts:

```python
class SequentialLlama4TextExperts(torch.nn.ModuleList):
    def __init__(self, config: Llama4TextConfig, original: Llama4TextExperts):
        self.num_experts = original.gate_up_proj.shape[0]
        with skip_weights_initialize():
            super().__init__([Llama4TextMLP(config) for _ in range(self.num_experts)])

        for i in range(self.num_experts):
            gate_up = original.gate_up_proj[i]
            down = original.down_proj[i]
            gate_proj, up_proj = gate_up.chunk(2, dim=-1)

            self[i].gate_proj.weight.data = gate_proj.t().contiguous()
            self[i].up_proj.weight.data = up_proj.t().contiguous()
            self[i].down_proj.weight.data = down.t().contiguous()
```

Key details from the Llama4 implementation:

- `is_permanent = True` — the unpacked expert form is required for vLLM inference, so the module is not restored after calibration
- Expert weights are unpacked from a 3D packed tensor into a `ModuleList` of individual MLPs
- The config passed to `__init__` is a multimodal `Llama4Config`; text-specific settings are extracted via `config.get_text_config()`
- A `shared_expert` runs on all tokens unconditionally and its output is used as the accumulation base

## Step-by-Step: Adding Support for a New Model

### Step 1: Identify the MoE block class name

Find the class in the transformers library that implements the MoE routing for your model:

```python
from transformers.models.your_model.modeling_your_model import YourModelMoE
print(YourModelMoE.__name__)  # e.g. "YourModelMoE"
```

### Step 2: Determine whether experts are packed

Inspect the original MoE module to see how experts are stored:

```python
import inspect
print(inspect.getsource(YourModelMoE.__init__))
```

- If experts are stored as a `ModuleList` of individual layers, you can copy them directly.
- If experts are stored as a packed 3D tensor (like Llama4), you need a helper class to unpack them into a `ModuleList` before calibration.

### Step 3: Create the calibration module

Create a new file `src/llmcompressor/modeling/your_model.py`:

```python
from typing import Tuple

import torch
from transformers.models.your_model.configuration_your_model import YourModelConfig
from transformers.models.your_model.modeling_your_model import YourModelMoE as OriginalYourModelMoE

from llmcompressor.modeling.moe_context import MoECalibrationModule


@MoECalibrationModule.register("YourModelMoE")
class SequentialYourModelMoE(MoECalibrationModule):
    """
    Calibration version of YourModelMoE that sends all tokens to all experts
    during calibration to ensure proper quantization statistics are collected.
    """

    is_permanent = True  # set False if unpacking is not needed and you want restoration

    def __init__(
        self,
        original: OriginalYourModelMoE,
        config: YourModelConfig,
        calibrate_all_experts: bool = True,
    ):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_local_experts

        # Unpack packed experts if needed, or copy directly:
        # self.experts = SequentialYourModelExperts(config, original.experts)
        self.experts = original.experts
        self.router = original.router
        self.shared_expert = original.shared_expert
        self.calibrate_all_experts = calibrate_all_experts

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_scores, router_logits = self.router(hidden_states)
        out = self.shared_expert(hidden_states)

        _, router_indices = torch.topk(router_logits, self.top_k, dim=1)
        expert_mask = torch.nn.functional.one_hot(
            router_indices, num_classes=self.num_experts
        ).permute(2, 1, 0)

        for i in range(self.num_experts):
            token_idx = torch.where(expert_mask[i].squeeze(0))

            if self.calibrate_all_experts:
                expert_out = self.experts[i](hidden_states)[token_idx]
            else:
                expert_out = self.experts[i](hidden_states[token_idx])

            if len(token_idx) > 0:
                weighted_output = expert_out * router_scores[:, i][token_idx].reshape(-1, 1)
                out[token_idx] += weighted_output

        return out, router_logits
```

### Step 4: Import the calibration module at the call site

The `@MoECalibrationModule.register(...)` decorator only takes effect when the module is imported. Import it before calling `oneshot`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modeling.your_model import SequentialYourModelMoE  # noqa: F401
from llmcompressor.modifiers.quantization import QuantizationModifier

model_id = "your-org/your-moe-model"
model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

oneshot(
    model=model,
    dataset=ds,
    recipe=[QuantizationModifier(targets="Linear", scheme="NVFP4", ignore=["lm_head"])],
    num_calibration_samples=512,
    max_seq_length=2048,
)

model.save_pretrained("your-moe-model-FP8", save_compressed=True)
tokenizer.save_pretrained("your-moe-model-FP8")
```

## Tips

- **The register name must exactly match the original class name** (case-sensitive). Inspect `module.__class__.__name__` if unsure.
- **Check whether experts are packed.** If the model stores experts as a single high-dimensional tensor rather than a `ModuleList`, you need to unpack them before calibration — see the `SequentialLlama4TextExperts` pattern.
- **Match the original `forward` signature exactly**, including return values. Llama4, for example, returns `(out, router_logits)`.
- **Apply routing scores to expert outputs, not inputs.** Applying scores before the expert forward pass can produce NaNs during calibration.
- **Use `is_permanent=True` when the unpacked form is required for inference** (e.g., vLLM needs individual expert modules). Use `is_permanent=False` when you only need calibration coverage and want the original structure restored afterwards.
- **Test with a small model or a few calibration samples first** to confirm all experts are reached and no NaNs appear.
