# Adding MoE Calibration Support for a New Model

Mixture of Experts (MoE) models route each token to only a subset of expert layers. This creates a calibration problem: experts that are not activated for a given token never see calibration data, which can result in poorly calibrated quantization parameters, numerical instability, or NaNs.

LLM Compressor solves this by temporarily replacing MoE modules with calibration-friendly versions that route all tokens through all experts during calibration, while keeping only the routed expert outputs for the final result.

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
5. Implement `restore(original)` if `is_permanent=False`

```python
import torch
from llmcompressor.modeling.moe_context import MoECalibrationModule


@MoECalibrationModule.register("MyModelMoE")  # exact class name from transformers
class CalibrationMyModelMoE(MoECalibrationModule):

    is_permanent = False

    def __init__(self, original, config, calibrate_all_experts: bool = True):
        super().__init__()
        # Copy the attributes you need from the original module
        self.experts = original.experts
        self.gate = original.gate
        self.calibrate_all_experts = calibrate_all_experts

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...

    def restore(self, original: torch.nn.Module) -> torch.nn.Module:
        return original
```

## The `forward` Pattern

The key behavior difference between normal MoE routing and calibration routing is:

- **Normal routing**: only tokens selected by the gate run through each expert
- **Calibration routing**: all tokens run through every expert (but only the gated outputs contribute to the final result)

The standard pattern:

```python
def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    topk_indices, topk_weights = self.gate(hidden_states)
    hidden_states_flat = hidden_states.view(-1, hidden_states.shape[-1])

    final_hidden_states = torch.zeros_like(hidden_states_flat, dtype=topk_weights.dtype)
    expert_mask = torch.nn.functional.one_hot(topk_indices, num_classes=len(self.experts))
    expert_mask = expert_mask.permute(2, 0, 1)

    for expert_idx, expert in enumerate(self.experts):
        mask = expert_mask[expert_idx]
        token_indices, weight_indices = torch.where(mask)
        has_tokens = token_indices.numel() > 0

        if self.calibrate_all_experts:
            # Run ALL tokens through the expert to collect calibration statistics.
            # Only the routed tokens contribute to the output.
            expert_output_full = expert(hidden_states_flat)
            if not has_tokens:
                continue  # stats collected, no output contribution
            expert_output = expert_output_full[token_indices]
        else:
            if not has_tokens:
                continue
            expert_output = expert(hidden_states_flat[token_indices])

        expert_weights = topk_weights[token_indices, weight_indices]
        final_hidden_states.index_add_(
            0, token_indices, expert_output * expert_weights.unsqueeze(-1)
        )

    return final_hidden_states.view_as(hidden_states)
```

## Example: GLM-4.7

The existing `CalibrationGlm4MoeMoE` (in `src/llmcompressor/modeling/glm4_moe.py`) is the canonical reference implementation. It registers itself as a replacement for `Glm4MoeMoE` from the transformers library, copies the gate, experts, and shared experts from the original module, and implements the calibration forward pass described above.

Key details from that implementation:

- `is_permanent = False` — the original MoE module is restored after calibration completes
- `restore()` returns the original module unchanged
- The forward pass handles shared experts (added after the MoE block) in addition to routed experts

## Step-by-Step: Adding Support for a New Model

### Step 1: Identify the MoE block class name

Find the class in the transformers library that implements the MoE routing for your model. For example:

```python
from transformers.models.your_model.modeling_your_model import YourModelMoE
print(YourModelMoE.__name__)  # e.g. "YourModelMoE"
```

### Step 2: Create the calibration module

Create a new file `src/llmcompressor/modeling/your_model.py`:

```python
import torch
from transformers.models.your_model.configuration_your_model import YourModelConfig
from transformers.models.your_model.modeling_your_model import YourModelMoE as OriginalYourModelMoE

from llmcompressor.modeling.moe_context import MoECalibrationModule


@MoECalibrationModule.register("YourModelMoE")
class CalibrationYourModelMoE(MoECalibrationModule):
    """
    Calibration version of YourModelMoE that sends all tokens to all experts
    during calibration to ensure proper quantization statistics are collected.
    """

    is_permanent = False

    def __init__(
        self,
        original: OriginalYourModelMoE,
        config: YourModelConfig,
        calibrate_all_experts: bool = True,
    ):
        super().__init__()
        self.config = config
        self.experts = original.experts
        self.gate = original.gate
        self.calibrate_all_experts = calibrate_all_experts

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        topk_indices, topk_weights = self.gate(hidden_states)
        hidden_states_flat = hidden_states.view(-1, hidden_states.shape[-1])

        final_hidden_states = torch.zeros_like(
            hidden_states_flat, dtype=topk_weights.dtype
        )
        expert_mask = torch.nn.functional.one_hot(
            topk_indices, num_classes=len(self.experts)
        ).permute(2, 0, 1)

        for expert_idx, expert in enumerate(self.experts):
            mask = expert_mask[expert_idx]
            token_indices, weight_indices = torch.where(mask)
            has_tokens = token_indices.numel() > 0

            if self.calibrate_all_experts:
                expert_output_full = expert(hidden_states_flat)
                if not has_tokens:
                    continue
                expert_output = expert_output_full[token_indices]
            else:
                if not has_tokens:
                    continue
                expert_output = expert(hidden_states_flat[token_indices])

            expert_weights = topk_weights[token_indices, weight_indices]
            final_hidden_states.index_add_(
                0, token_indices, expert_output * expert_weights.unsqueeze(-1)
            )

        return final_hidden_states.view_as(hidden_states)

    def restore(self, original: torch.nn.Module) -> torch.nn.Module:
        return original
```

### Step 3: Import the calibration module at the call site

The `@MoECalibrationModule.register(...)` decorator only takes effect when the module is imported. Import it before calling `oneshot`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modeling.your_model import CalibrationYourModelMoE  # noqa: F401
from llmcompressor.modifiers.awq import AWQModifier

model_id = "your-org/your-moe-model"
model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

recipe = AWQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    num_calibration_samples=512,
    max_seq_length=2048,
)

model.save_pretrained("your-moe-model-W4A16", save_compressed=True)
tokenizer.save_pretrained("your-moe-model-W4A16")
```

## Tips

- **The register name must exactly match the original class name** (case-sensitive). Inspect `module.__class__.__name__` if unsure.
- **Copy only what you need from `original`.** Most implementations only need `experts`, `gate`, and any shared expert modules.
- **Match the original `forward` signature exactly**, including any additional arguments the transformer layer passes (e.g., `attention_mask`, `position_ids`).
- **Test with a small model or a few calibration samples first** to confirm all experts are being reached and no NaNs appear in the output.
- **`is_permanent=True` is rarely needed.** Use it only if the calibration-form module is compatible with inference and you intentionally want to skip restoration.
