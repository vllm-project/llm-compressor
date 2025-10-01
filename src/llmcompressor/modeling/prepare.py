import tqdm
from compressed_tensors.utils import replace_module
from transformers import PreTrainedModel

from llmcompressor.modeling.deepseek_v3 import replace as replace_deepseekv3
from llmcompressor.modeling.llama4 import replace as replace_llama4
from llmcompressor.modeling.qwen3_moe import replace as replace_Qwen3MoE
from llmcompressor.modeling.qwen3_next_moe import replace as replace_Qwen3NextMoE
from llmcompressor.modeling.qwen3_vl_moe import replace as replace_Qwen3VLMoE
from llmcompressor.utils.helpers import patch_attr

__all__ = ["replace_modules_for_calibration"]

# ---------------------- module replacements; permanent -------------------------
replacements = {
    "DeepseekV3MoE": replace_deepseekv3,
    "Llama4TextMoe": replace_llama4,
    "Qwen3VLMoeTextSparseMoeBlock": replace_Qwen3VLMoE,
}


def replace_modules_for_calibration(
    model: PreTrainedModel,
    calibrate_all_experts: bool = True,
) -> PreTrainedModel:
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


def update_qwen3_moe(model, module, stack, calibrate_all_experts):
    cls_name = module.__class__.__name__
    if (
        cls_name == "Qwen3MoeDecoderLayer"
        and module.mlp.__class__.__name__ == "Qwen3MoeSparseMoeBlock"
    ):
        stack.enter_context(
            patch_attr(
                module,
                "mlp",
                replace_Qwen3MoE(
                    config=model.config,
                    module=module.mlp,
                    calibrate_all_experts=calibrate_all_experts,
                ),
            )
        )


def update_qwen3_next_moe(model, module, stack, calibrate_all_experts):
    cls_name = module.__class__.__name__
    if (
        cls_name == "Qwen3NextDecoderLayer"
        and module.mlp.__class__.__name__ == "Qwen3NextSparseMoeBlock"
    ):
        stack.enter_context(
            patch_attr(
                module,
                "mlp",
                replace_Qwen3NextMoE(
                    config=model.config,
                    module=module.mlp,
                    calibrate_all_experts=calibrate_all_experts,
                ),
            )
        )


moe_context = {
    "Qwen3MoeForCausalLM": update_qwen3_moe,
    "Qwen3NextForCausalLM": update_qwen3_next_moe,
}


def moe_calibration_context(
    model: PreTrainedModel,
    stack,
    calibrate_all_experts: bool = True,
):
    # Temporarily updates the MoE modules within the context
    # Once the context exists, parameter updates persist
    model_name = model.__class__.__name__
    if model_name in moe_context:
        for module in model.modules():
            moe_context[model_name](model, module, stack, calibrate_all_experts)
