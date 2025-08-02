from compressed_tensors.utils import replace_module
from transformers import PreTrainedModel

from llmcompressor.modeling.config import CalibrationConfig
from llmcompressor.modeling.deepseek_v3 import replace as replace_deepseekv3
from llmcompressor.modeling.llama4 import replace as replace_llama4
from llmcompressor.modeling.qwen3_moe import replace as replace_Qwen3MoE
from llmcompressor.utils.helpers import patch_attr

__all__ = ["replace_modules_for_calibration"]


# ---------------------- module replacements; permanent -------------------------
replacements = {
    "DeepseekV3MoE": replace_deepseekv3,
    "Llama4TextMoe": replace_llama4,
}


def replace_modules_for_calibration(
    model: PreTrainedModel,
    moe_calibrate_all_experts: bool = True,
    moe_calibrate_gated_acts: bool = True,
) -> PreTrainedModel:

    calib_config = CalibrationConfig(
        moe_calibrate_all_experts=moe_calibrate_all_experts,
        moe_calibrate_gated_acts=moe_calibrate_gated_acts,
    )

    for name, module in model.named_modules():
        cls_name = module.__class__.__name__
        if cls_name in replacements:
            new_module = replacements[cls_name](
                config=model.config, module=module, calib_config=calib_config
            )
            replace_module(model, name, new_module)

    return model


# ------------------- module replacements; during calibration --------------------


def update_qwen3_moe(model, stack, calib_config):
    for module in model.modules():
        cls_name = module.__class__.__name__
        if cls_name == "Qwen3MoeDecoderLayer":
            # Optionally update the model.config to pass in other arguments
            stack.enter_context(
                patch_attr(
                    module,
                    "mlp",
                    replace_Qwen3MoE(
                        config=model.config,
                        module=module.mlp,
                        calib_config=calib_config,
                    ),
                )
            )


moe_context = {
    "Qwen3MoeForCausalLM": update_qwen3_moe,
}


def moe_calibration_context(
    model: PreTrainedModel,
    stack,
    moe_calibrate_all_experts: bool = True,
    moe_calibrate_gated_acts: bool = True,
):
    calib_config = CalibrationConfig(
        moe_calibrate_all_experts=moe_calibrate_all_experts,
        moe_calibrate_gated_acts=moe_calibrate_gated_acts,
    )

    # Temporarily updates the MoE modules within the context
    # Once the context exists, parameter updates persist
    cls_name = model.__class__.__name__
    if cls_name in moe_context:
        moe_context.get(cls_name)(model, stack, calib_config)
