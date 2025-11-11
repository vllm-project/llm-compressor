"""
MoE model preparation - imports and registration.

This module imports all MoE calibration modules to ensure they are registered
in the MOE_CALIBRATION_MODULES registry. The actual calibration logic is in
moe_context.py.
"""

import tqdm
from compressed_tensors.utils import deprecated, replace_module
from transformers import PreTrainedModel

# deprecated replacement functions
from llmcompressor.modeling.deepseek_v3 import replace as replace_deepseekv3
from llmcompressor.modeling.llama4 import replace as replace_llama4
from llmcompressor.modeling.qwen3_vl_moe import replace as replace_Qwen3VLMoE

__all__ = ["replace_modules_for_calibration"]

# ---------------------- module replacements; permanent -------------------------
replacements = {
    "DeepseekV3MoE": replace_deepseekv3,
    "Llama4TextMoe": replace_llama4,
    "Qwen3VLMoeTextSparseMoeBlock": replace_Qwen3VLMoE,
}


@deprecated(
    message=(
        "The function `replace_modules_for_calibration` has been deprecated. "
        "Please use `moe_calibration_context` instead. "
    )
)
def replace_modules_for_calibration(
    model: PreTrainedModel,
    calibrate_all_experts: bool = True,
) -> PreTrainedModel:
    """
    Deprecated function for backward compatibility.

    Use moe_calibration_context instead:
        with moe_calibration_context(model, calibrate_all_experts):
            # your code here

    Args:
        model: The model to modify
        calibrate_all_experts: Whether to calibrate all experts

    Returns:
        The modified model
    """
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
