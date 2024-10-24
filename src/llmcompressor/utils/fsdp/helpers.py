import contextlib
import operator
from pathlib import Path
from typing import Optional

from loguru import logger

from llmcompressor.utils.helpers import getattr_chain

try:
    from torch.distributed.fsdp import (
        FullStateDictConfig,
        FullyShardedDataParallel,
        StateDictType,
    )
except ImportError:
    FullyShardedDataParallel = None

import torch
from torch.nn import Module

from llmcompressor.core.state import State
from llmcompressor.pytorch.model_load.helpers import save_model_and_recipe
from llmcompressor.utils.pytorch import set_layer

__all__ = [
    "is_fsdp_model",
    "maybe_get_wrapped",
    "set_wrapped_model",
    "unwrap_and_export_model",
    "save_pretrained_fsdp",
    "get_fsdp_parent",
    "find_and_move_state_dicts_to_cpu",
]


def is_fsdp_model(model: Module) -> bool:
    """
    Check if a model instance is wrapped by FSDP

    :param model: pytorch model to check
    :return: True if module is wrapped, False otherwise
    """
    if not FullyShardedDataParallel:
        return False

    return isinstance(model, FullyShardedDataParallel)


def maybe_get_wrapped(model: Module) -> Module:
    """
    Given a model that may or may not have a distributed wrapper, return the underlying
    wrapped model.

    :param model: input model to get wrapped model from
    :returns: wrapped model
    """
    if is_fsdp_model(model=model):
        return model._fsdp_wrapped_module
    return model


def set_wrapped_model(state: State, wrapped_model: Module):
    """
    Given a state with a model that may or may not have a distributed wrapper, set
    the underlying wrapped model.

    :param state: state to update model of
    :param updated_wrapped: model to inject into input_model
    """
    if is_fsdp_model(state.model):
        state.model._fsdp_wrapped_module = wrapped_model
    else:
        state.model = wrapped_model


def unwrap_and_export_model(model, accelerator, output_dir, tokenizer):
    """
    Recursively unwraps an FSDP model, then saves the unwrapped model and the
    currently active recipe to disk

    :param model: model to unwrap
    :param accelerator: Accelerator instance used to perform unwrapping
    :param output_dir: where to save output model
    :param tokenizer: tokenizer used by the model
    """
    full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FullyShardedDataParallel.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        full_state_dict_config,
    ):
        unwrapped_model = accelerator.unwrap_model(model)
        for name, module in unwrapped_model.named_modules():
            if isinstance(module, FullyShardedDataParallel):
                set_layer(name, accelerator.unwrap_model(module), unwrapped_model)

        save_model_and_recipe(
            model=unwrapped_model,
            save_path=output_dir,
            tokenizer=tokenizer,
        )


def find_and_move_state_dicts_to_cpu(output_dir: str):
    """
    Looks for state dicts in the output directory and overwrites them
    with cpu state dicts.

    this is needed for quantized models trained with FSDP as the state dict
    contains device information, which can cause issues when loading the model
    using transformers AutoModel.from_pretrained(...) if the device information
    is not removed, assumes the state dicts are named pytorch_model*.bin
    """

    for model_file in Path(output_dir).rglob("pytorch_model*.bin"):
        loaded_dict = torch.load(model_file)
        for key, value in loaded_dict.items():
            if isinstance(value, torch.Tensor):
                loaded_dict[key] = value.cpu()

        torch.save(loaded_dict, model_file)
        logger.info(f"Moved state dict {model_file} to cpu")


def save_pretrained_fsdp(
    model,
    accelerator,
    output_dir,
    save_safetensors: bool = True,
    save_compressed: bool = False,
):
    full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    """
    Gathers the full FSDP state dict of the model onto rank0 GPU, then uses it to save
    the pretrained FSDP model to disk

    :param model: model to save
    :param accelerator: Accelerator instance used to perform unwrapping
    :param output_dir: where to save output model
    :param save_safetensors: True to safe in safetensors format, otherwise .bin
    :param save_compressed: whether to compress sparse weights on disk
    """
    with FullyShardedDataParallel.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, full_state_dict_config
    ):
        state_dict = accelerator.get_state_dict(model, unwrap=False)

    if accelerator.is_main_process:
        accelerator.unwrap_model(model).save_pretrained(
            output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=state_dict,
            save_compressed=save_compressed,
            safe_serialization=save_safetensors,
        )

    accelerator.wait_for_everyone()


def get_fsdp_parent(layer_name: str, model: Module) -> Optional[Module]:
    """
    Gets the closest parent of layer_name that is wrapped by FSDP. If no FSDP wrapper
    is found just return None

    :param layer_name: layer name in model to get parent of
    :model: pytorch module to search through
    :return: FSDP wrapped parent of layer_name if available, otherwise None
    """
    if not is_fsdp_model(model):
        return None

    parent_name = layer_name
    parent = operator.attrgetter(parent_name)(model)
    while not isinstance(parent, FullyShardedDataParallel):
        if len(parent_name) == 0:  # we've reached the root module and its not FSDP
            # this should never get hit because we check for an FSDP root above
            # but while statements without a backup are too scary
            return None
        parent_name = ".".join(parent_name.split(".")[:-1])
        parent = operator.attrgetter(parent_name)(model)

    return parent

def has_offloaded_params(module: torch.nn.Module) -> bool:
    """
    Checks if a module has offloaded parameters by checking if the given module
    has a AlignDevicesHook attached with offloading enabled
    Args:
        module (`torch.nn.Module`): The module to check for an offload hook.
    Returns:
        bool: `True` if the module has an offload hook and offloading is enabled,
        `False` otherwise.
    """
    from accelerate.hooks import AlignDevicesHook

    return (
        hasattr(module, "_hf_hook") and
        isinstance(module._hf_hook, AlignDevicesHook) and
        module._hf_hook.offload
    )

@contextlib.contextmanager
def align_module(
    module: torch.nn.Module,
    execution_device: Optional[torch.device] = None,
    args = tuple(), kwargs = dict()
):
    """
    Move a module's parameters to the execution device
    :param module: module with parameters to align
    :param execution_device: if provided, overrides module execution device
        within the context
    """
    if has_offloaded_params(module):
        if execution_device is not None:
            original_device = module._hf_hook.execution_device
            module._hf_hook.execution_device = original_device

        module._hf_hook.pre_forward(module, *args, **kwargs)
        yield
        module._hf_hook.post_forward(module, None)

        if execution_device is not None:
            module._hf_hook.execution_device = original_device

    elif execution_device is not None:
        devices = {}
        for name, param in module.named_parameters():
            devices[name] = param.device
            setattr(module, name, param.to(execution_device))

        yield

        for name, param_device in module.named_parameters:
            setattr(module, name, param.to(param_device))

    else:
        yield


def update_offload_parameter(
    module: torch.nn.Module,
    name: str,
    data: torch.Tensor,
    init_device: Optional[torch.device] = torch.device("cpu"),
):
    """
    :param module: module containing the parameter to update
    :param name: name of module parameter to update
    :param data: tensor to update parameter with
    :param init_device: offload device for newly registered parameters
    """
    param = getattr(module, name)
    param.data = data

    prefix_dict = getattr_chain(module, "module._hf_hook.weights_map.dataset", None)
    if prefix_dict is not None:
        prefix = module._hf_hook.weights_map.prefix
        key = f"{prefix}{name}"

        offload_device = prefix_dict[key].device if key in prefix_dict else init_device
        prefix_dict[key] = data.to(device=offload_device)


def register_offload_parameter(
    module: torch.nn.Module,
    name: str,
    data: torch.Tensor,
    offload_device: Optional[torch.device] = torch.device("cpu"), 
):
    module.register_parameter(name, torch.nn.Parameter(data))
    update_offload_parameter(module, name, data, offload_device)