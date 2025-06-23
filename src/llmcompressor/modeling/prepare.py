import torch
from transformers import PreTrainedModel

from llmcompressor.modeling.deepseek_v3 import replace as replace_DeepseekV3MoE
from llmcompressor.utils.module import module_bfs

__all__ = ["prepare_for_calibration"]

replacements = {
    "DeepseekV3MoE": replace_DeepseekV3MoE,
}


def prepare_for_calibration(model: PreTrainedModel) -> PreTrainedModel:
    def replace(module: torch.nn.Module) -> torch.nn.Module:
        cls_name = module.__class__.__name__
        if cls_name in replacements:
            return replacements[cls_name](module)
        else:
            return module

    return module_bfs(model, replace, progress=True)
