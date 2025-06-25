from compressed_tensors.utils import replace_module
from transformers import PreTrainedModel

from llmcompressor.modeling.deepseek_v3 import replace as replace_deepseekv3
from llmcompressor.modeling.llama4 import replace as replace_llama4
from llmcompressor.utils.helpers import patch_attr

__all__ = ["prepare_for_calibration"]

replacements = {
    "DeepseekV3MoE": replace_deepseekv3,
    "Llama4TextMoe": replace_llama4,
}

def prepare_for_calibration(model: PreTrainedModel) -> PreTrainedModel:
    for name, module in model.named_modules():
        cls_name = module.__class__.__name__
        if cls_name in replacements:
            new_module = replacements[cls_name](config=model.config, module=module)
            replace_module(model, name, new_module)

    return model

def update_qwen3_moe(model, stack):
    for module in model.model.layers:
        stack.enter_context(
            patch_attr(module.mlp, "top_k", model.config.num_experts)
        )


moe_context = {
    "Qwen3MoeForCausalLM": update_qwen3_moe,
}

def calibrate_moe_context(model: PreTrainedModel, stack):
    cls_name = model.__class__.__name__
    moe_context.get(cls_name)(model, stack)
