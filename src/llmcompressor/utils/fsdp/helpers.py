import contextlib
from functools import wraps
import operator
from pathlib import Path
from typing import Optional
import warnings

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

try:
    from accelerate.hooks import AlignDevicesHook
    from accelerate.utils import OffloadedWeightsLoader, PrefixedDataset
    _has_accelerate = True
except ImportError:
    _has_accelerate = False
    AlignDevicesHook = None
    OffloadedWeightsLoader = None
    PrefixedDataset = None

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

# upstream candidate
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
    return (
        hasattr(module, "_hf_hook") and
        isinstance(module._hf_hook, AlignDevicesHook) and
        module._hf_hook.offload
    )


# depreciation candidate
@wraps(has_offloaded_params)
def is_module_offloaded(module: torch.nn.Module) -> bool:
    if not _has_accelerate:
        return False

    return has_offloaded_params(module)


# depreciation candidate
def get_execution_device(module: torch.nn.Module) -> torch.device:
    """
    :param module: module to check
    :return: device module is loaded onto during forward pass
    """
    if is_module_offloaded(module):
        return module._hf_hook.execution_device
    device = next(module.parameters()).device

    # offload only gets set for leaf modules, fallback to checking for device type
    if device.type == "meta":
        return module._hf_hook.execution_device

    return device


# upstream candidate
def _infer_offload_device(module: torch.nn.Module) -> torch.device:
    if not has_offloaded_params(module):
        raise ValueError("Cannot infer offload device from non-offloaded module")
    
    first_key = next(module._hf_hook.weights_map.keys(), None)
    if first_key is None:
        raise ValueError("Cannot infer offload device from empty weights map")

    prefix_dataset = module._hf_hook.weights_map.dataset
    return prefix_dataset[first_key].device

# depreciation candidate
def get_offloaded_device(module: torch.nn.Module) -> torch.device:
    """
    :param module: module to check
    :return: device module is offloaded to onto after forward pass
    """
    return _infer_offload_device(module)


# depreciation candidate
def update_prefix_dict(module: torch.nn.Module, key: str, data: torch.Tensor):
    """
    Updates the offloaded state dict for a given module. Parameter named key is replaced
    by data. This is neccesary because parameter updates for offloaded modules do not
    persist automatically between loads. This function only affects the offloaded
    state dict and not the current state of the loaded module.

    :param module: module containing the parameter to update
    :param key: name of parameter to update
    :param data: tensor to update parameter with in the offloaded state dict
    """
    if not is_module_offloaded(module):
        raise ValueError("Prefix dict is only applicable to offloaded modules")
    prefix_dict = module._hf_hook.weights_map
    prefix_dict.dataset[f"{prefix_dict.prefix}{key}"] = data


# upstream candidate?
def update_offload_parameter(
    module: torch.nn.Module,
    name: str,
    data: Optional[torch.Tensor] = None,
    offload_device: Optional[torch.device] = None,
):
    """
    :param module: module containing the parameter to update
    :param name: name of module parameter to update
    :param data: tensor to update parameter with
    :param offload_device: offload device for newly registered parameters
    """
    param = getattr(module, name)
    if data is not None:
        if data.device == "meta":
            raise ValueError("Cannot copy data from meta device. Consider calling with align_module(module) context")
    
        if param.data.dtype != data.dtype:
            warnings.warn("TODO")
            
        param.data.copy_(data)

    if has_offloaded_params(module):
        weights_map = module._hf_hook.weights_map

        # for upstreaming, probably better to modify the weight map types so that they can be written to?
        if isinstance(weights_map, PrefixedDataset):
            prefix_dict = getattr_chain(module, "module._hf_hook.weights_map.dataset", None)
            if prefix_dict is not None:
                prefix = module._hf_hook.weights_map.prefix
                key = f"{prefix}{name}"

                offload_device = (
                    prefix_dict[key].device if key in prefix_dict
                    else offload_device if offload_device is not None
                    else _infer_offload_device(module)
                )
                prefix_dict[key] = param.data.to(device=offload_device)
            
        if isinstance(weights_map, OffloadedWeightsLoader):
            raise NotImplementedError()
        
        else:
            raise NotImplementedError()

# depreciation candidate
def update_parameter_data(
    module: torch.nn.Module, new_param_data: torch.Tensor, param_name: str
):
    param = getattr(module, param_name)
    new_param_data = new_param_data.to(device=param.device, dtype=param.dtype)
    update_offload_parameter(module, param_name, new_param_data)


# upstream candidate
@contextlib.contextmanager
def align_module(module: torch.nn.Module, execution_device: Optional[torch.device] = None):
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

        module._hf_hook.pre_forward(module)
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


@contextlib.contextmanager
def modify_offload_module(
    module: torch.nn.Module,
    execution_device: Optional[torch.device] = None,
    offload_device: Optional[torch.device] = None,
):
    with align_module(module, execution_device):
        yield

        # there is little performance gain from checking if a parameter's data
        # has been modified before copying since the new data must be copied
        # to the offload device anyways; just update all module parameters
        for name, param in module.named_parameters():
            update_offload_parameter(module, name, param.data, offload_device)


# upstream candidate?
def register_offload_parameter(
    module: torch.nn.Module,
    name: str,
    parameter: torch.nn.Parameter,
    offload_device: Optional[torch.device] = None,
):
    module.register_parameter(name, parameter)
    update_offload_parameter(module, name, parameter.data, offload_device)


# upstream candidate?
def deregister_offload_parameter():
    raise NotImplementedError()