from compressed_tensors.utils import replace_module
from transformers import PreTrainedModel
from llmcompressor.utils.helpers import patch_attr
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

def update_qwen3_moe(model, stack):
    for module in model.model.layers:
        stack.enter_context(
            patch_attr(module.mlp, "top_k", model.config.num_experts)
        )


def update_deepseekv3(model, stack):
    for i in range(len(model.model.layers)):
        if i > 2:
            stack.enter_context(
                patch_attr(model.model.layers[i], "mlp", replace_DeepseekV3MoE)
            )

moe_context = {
    "Qwen3MoeForCausalLM": update_qwen3_moe,
    "DeepseekV3ForCausalLM": update_deepseekv3
}

def calibrate_moe_context(model: PreTrainedModel, stack):
    cls_name = model.__class__.__name__
    moe_context.get(cls_name)(model, stack)
