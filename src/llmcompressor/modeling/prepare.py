import torch
from transformers import PreTrainedModel
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3MoE

from llmcompressor.modeling.deepseek_v3 import replace as replace_DeepseekV3MoE
from llmcompressor.utils.module import module_bfs

__all__ = ["prepare_for_quantization"]

replacements = {
    DeepseekV3MoE: replace_DeepseekV3MoE,
}


def prepare_for_quantization(model: PreTrainedModel) -> PreTrainedModel:
    def replace(module: torch.nn.Module) -> torch.nn.Module:
        if module.__class__ in replacements:
            return replacements[module.__class__](module)
        else:
            return module

    return module_bfs(model, replace, progress=True)
