import contextlib
import tqdm
from compressed_tensors.utils import replace_module, delete_offload_module, register_offload_module, get_offloaded_device
from compressed_tensors.utils.offload import offload_to_weights_map
from transformers import PreTrainedModel

from llmcompressor.modeling.deepseek_v3 import replace as replace_deepseekv3
from llmcompressor.modeling.llama4 import replace as replace_llama4
from llmcompressor.modeling.qwen3_moe import replace as replace_Qwen3MoE
from llmcompressor.modeling.gpt_oss import GptOssExpertsLinear, replace_gpt_oss
from llmcompressor.utils.helpers import patch_attr
from accelerate.hooks import add_hook_to_module, remove_hook_from_module, AlignDevicesHook, named_module_tensors, set_module_tensor_to_device, PrefixedDataset
from accelerate.big_modeling import attach_align_device_hook_on_blocks

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
        #replace_offload_module(module, name, hook, restored)
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


    # # offloading kwargs for submodule
    # place_submodules = False
    # offload_buffers = True

    # # copy device offloading arguments from parent
    # current_device = next(base.parameters()).device  # assume base has parameters
    # offload_device = get_offloaded_device(base)

    # # offload parameters to weights map
    # for param_name, param in named_module_tensors(
    #     module, include_buffers=offload_buffers, recurse=place_submodules
    # ):
    #     offloaded = param.to(offload_device)
    #     if hook.tied_params_map is not None:
    #         hook.tied_params_map[offloaded.data_ptr()] = {}  # (1)
    #     offload_to_weights_map(hook.weights_map, f"{name}.{param_name}", offloaded)

    #     # if the parent places submodules, offload here
    #     if hook.place_submodules:
    #         set_module_tensor_to_device(module, param_name, current_device)

    # if not hook.place_submodules:
    #     weights_map = PrefixedDataset(
    #         hook.weights_map.dataset, prefix=f"{hook.weights_map.prefix}{name}."
    #     )

    #     submodule_hook = AlignDevicesHook(
    #         execution_device=hook.execution_device,
    #         offload=hook.offload,
    #         io_same_device=False,
    #         weights_map=weights_map,
    #         offload_buffers=offload_buffers,
    #         place_submodules=place_submodules,
    #         skip_keys=None,
    #         tied_params_map=hook.tied_params_map,
    #     )
    #     add_hook_to_module(module, submodule_hook)

    # base.register_module(name, module)
    # for c_name, child in list(module.named_children()):
    #     register_offload_module(module, c_name, child)
    #     replace_offload_module(module, None, c_name, child, child)