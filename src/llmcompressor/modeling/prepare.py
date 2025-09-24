import contextlib
import warnings

import tqdm
from compressed_tensors.utils import replace_module
from transformers import PreTrainedModel

from llmcompressor.modeling.deepseek_v3 import replace as replace_deepseekv3
from llmcompressor.modeling.llama4 import replace as replace_llama4
from llmcompressor.modeling.moe_context import (
    MoECalibrationType,
    MoEModelConfig,
    get_moe_context,
    register_moe_model,
)
from llmcompressor.modeling.qwen3_moe import replace as replace_Qwen3MoE

__all__ = ["moe_calibration_context"]

# ---------------------- module replacements; permanent -------------------------
replacements = {
    "DeepseekV3MoE": replace_deepseekv3,
    "Llama4TextMoe": replace_llama4,
}


def replace_modules_for_calibration(
    model: PreTrainedModel,
    calibrate_all_experts: bool = True,
) -> PreTrainedModel:
    # This function is deprecated. Use moe_calibration_context instead.
    warnings.warn(
        "replace_modules_for_calibration is deprecated. "
        "Use moe_calibration_context instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    for name, module in tqdm.tqdm(list(model.named_modules())):
        cls_name = module.__class__.__name__
        if cls_name in replacements:
            new_module = replacements[cls_name](
                config=model.config,
                module=module,
                calibrate_all_experts=calibrate_all_experts,
            )
            replace_module(model, name, new_module)

    return model


# ------------------- module replacements; during calibration --------------------

# MoE model configurations - centralized registry
# Adding a new MoE model is now as simple as adding an entry here!
# This follows the same pattern as MAPPINGS_REGISTRY in SmoothQuant and AWQ
MOE_EXPERTS_REPLACEMENTS = {
    "Qwen3MoeForCausalLM": MoEModelConfig(
        calibration_type=MoECalibrationType.CONTEXTUAL,
        target_class_name="Qwen3MoeDecoderLayer",
        target_attribute="mlp",
        replace_function=replace_Qwen3MoE,
        description="Qwen3 MoE model with contextual calibration for MLP layers",
    ),
    "DeepseekV3ForCausalLM": MoEModelConfig(
        calibration_type=MoECalibrationType.PERMANENT,
        target_class_name="DeepseekV3MoE",
        replace_function=replace_deepseekv3,
        description="DeepSeek V3 MoE model with permanent calibration",
    ),
    "Llama4ForConditionalGeneration": MoEModelConfig(
        calibration_type=MoECalibrationType.PERMANENT,
        target_class_name="Llama4TextMoe",
        replace_function=replace_llama4,
        description=(
            "Llama4 MoE model with permanent calibration for vLLM compatibility"
        ),
    ),
}


# Register all MoE models automatically
for model_class_name, config in MOE_EXPERTS_REPLACEMENTS.items():
    register_moe_model(model_class_name, config)


@contextlib.contextmanager
def moe_calibration_context(
    model: PreTrainedModel,
    calibrate_all_experts: bool = True,
):
    """
    Context manager for MoE calibration that temporarily updates MoE modules.

    Args:
        model: The model to apply MoE calibration to
        calibrate_all_experts: Whether to calibrate all experts or only routed ones

    Yields:
        The model with MoE calibration applied
    """
    cls_name = model.__class__.__name__
    moe_context = get_moe_context(cls_name)

    if moe_context is None:
        # No MoE context registered for this model, yield unchanged
        yield model
        return

    # Apply MoE calibration
    moe_context.apply(model, calibrate_all_experts)

    try:
        yield model
    finally:
        # Restore original state
        moe_context.restore(model)
