from compressed_tensors.utils import replace_module
from transformers import PreTrainedModel

from llmcompressor.modeling.deepseek_v3 import replace as replace_DeepseekV3MoE

__all__ = ["prepare_for_calibration"]

replacements = {
    "DeepseekV3MoE": replace_DeepseekV3MoE,
}


def prepare_for_calibration(model: PreTrainedModel) -> PreTrainedModel:
    for name, module in model.named_modules():
        cls_name = module.__class__.__name__
        if cls_name in replacements:
            new_module = replacements[cls_name](module)
            replace_module(model, name, new_module)

    return model
