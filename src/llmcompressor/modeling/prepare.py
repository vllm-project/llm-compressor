import contextlib

import tqdm
from accelerate.big_modeling import attach_align_device_hook_on_blocks
from accelerate.hooks import AlignDevicesHook, named_module_tensors
from compressed_tensors.utils import replace_module
from compressed_tensors.utils.offload import offload_to_weights_map
from transformers import PreTrainedModel

from llmcompressor.modeling.deepseek_v3 import replace as replace_deepseekv3
from llmcompressor.modeling.gpt_oss import replace_gpt_oss
from llmcompressor.modeling.llama4 import replace as replace_llama4
from llmcompressor.modeling.qwen3_moe import replace as replace_Qwen3MoE
from llmcompressor.utils.helpers import patch_attr

__all__ = ["replace_modules_for_calibration"]

# ---------------------- module replacements; permanent -------------------------
replacements = {
    "DeepseekV3MoE": replace_deepseekv3,
    "Llama4TextMoe": replace_llama4,
    "GptOssExperts": replace_gpt_oss,
}


def replace_modules_for_calibration(model: PreTrainedModel) -> PreTrainedModel:
    modules = list(model.named_modules())
    for name, module in tqdm.tqdm(modules, desc="Converting modules"):
        cls_name = module.__class__.__name__
        if cls_name in replacements:
            new_module = replacements[cls_name](config=model.config, module=module)
            replace_module(model, name, new_module)

    return model


# ------------------- module replacements; during calibration --------------------


def update_qwen3_moe(model: PreTrainedModel, stack):
    modules = list(model.modules())
    for module in tqdm.tqdm(modules, desc="Converting modules"):
        cls_name = module.__class__.__name__
        if cls_name == "Qwen3MoeDecoderLayer":
            # Optionally update the model.config to pass in other arguments
            stack.enter_context(
                patch_attr(
                    module,
                    "mlp",
                    replace_Qwen3MoE(config=model.config, module=module.mlp),
                )
            )


def update_gpt_oss(model: PreTrainedModel, stack):
    @contextlib.contextmanager
    def replace(mod_name, module, name, original):
        hook: AlignDevicesHook = original._hf_hook

        replacement = replace_gpt_oss(model.config, original)
        replace_offload_module(module, name, hook, replacement)
        del original

        yield

        restored = replacement.to_original()
        delattr(module, name)
        module.register_module(name, restored)
        # replace_offload_module(module, name, hook, restored)
        del replacement

    modules = list(model.named_modules())
    for name, module in tqdm.tqdm(modules, desc="Converting modules"):
        for child_name, child in list(module.named_children()):
            if child.__class__.__name__ == "GptOssExperts":
                stack.enter_context(replace(name, module, child_name, child))


moe_context = {
    "Qwen3MoeForCausalLM": update_qwen3_moe,
    "GptOssForCausalLM": update_gpt_oss,
}


def moe_calibration_context(model: PreTrainedModel, stack):
    # Temporarily updates the MoE modules within the context
    # Once the context exists, parameter updates persist
    cls_name = model.__class__.__name__
    if cls_name in moe_context:
        moe_context.get(cls_name)(model, stack)


def replace_offload_module(base, name: str, hook: AlignDevicesHook, module):
    delattr(base, name)

    assert hook.offload
    assert hook.weights_map is not None

    # offload parameters to weights map
    offload_device = "cpu"
    for param_name, param in named_module_tensors(
        module, include_buffers=hook.offload_buffers, recurse=True
    ):
        offloaded = param.to(offload_device)
        if hook.tied_params_map is not None:
            hook.tied_params_map[offloaded.data_ptr()] = {}  # (1)
        offload_to_weights_map(hook.weights_map, param_name, offloaded)

    # attach hooks and offload weights
    attach_align_device_hook_on_blocks(
        module,
        hook.execution_device,
        hook.offload,
        hook.weights_map,
        hook.offload_buffers,
        "",
        hook.skip_keys,
        None,
        hook.tied_params_map,
    )

    base.register_module(name, module)
