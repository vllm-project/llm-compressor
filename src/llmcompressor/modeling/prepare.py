from dataclasses import dataclass

from compressed_tensors.utils import replace_module
from transformers import PreTrainedModel

from llmcompressor.modeling.deepseek_v3 import replace as replace_DeepseekV3MoE

__all__ = ["prepare_for_calibration"]


replacements = {
    "DeepseekV3MoE": replace_DeepseekV3MoE,
}


def prepare_for_calibration(
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
            new_module = replacements[cls_name](module, calib_config)
            replace_module(model, name, new_module)

    return model


@dataclass
class CalibrationConfig:
    moe_calibrate_all_experts: bool
    moe_calibrate_gated_acts: bool
