# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Utilities associated with offloading functionality provided by `accelerate`.

| ----------------------------------------------------------------------------------------------------- | # noqa: E501
| Operation | Without offloading support             | With offloading support                          | # noqa: E501
| --------- | -------------------------------------- | ------------------------------------------------ | # noqa: E501
| Add       | module.register_parameter(name, param) | register_offload_parameter(module, name, param)  | # noqa: E501
| Check     | N/A                                    | has_offloaded_params(module)                     | # noqa: E501
| Onload    | N/A                                    | with align_module_device(module)                 | # noqa: E501
| Update    | module.name.data.copy_(new_data)       | update_offload_parameter(module, name, new_data) | # noqa: E501
| Delete    | del module.name                        | delete_offload_parameter(module, name)           | # noqa: E501
| ----------------------------------------------------------------------------------------------------- | # noqa: E501
"""

import contextlib
import warnings
from functools import wraps
from typing import Any, Callable, Dict, Literal, Optional, Union

import torch


try:
    from accelerate.hooks import (
        AlignDevicesHook,
        add_hook_to_module,
        remove_hook_from_module,
    )
    from accelerate.utils import (
        OffloadedWeightsLoader,
        PrefixedDataset,
        set_module_tensor_to_device,
    )

    _has_accelerate = True
except ImportError:
    _has_accelerate = False
    AlignDevicesHook = None
    add_hook_to_module = None
    remove_hook_from_module = None
    OffloadedWeightsLoader = None
    PrefixedDataset = None
    set_module_tensor_to_device = None


__all__ = [
    "is_module_offloaded",
    "get_execution_device",
    "get_offloaded_device",
    "update_prefix_dict",
    "update_parameter_data",
    "register_offload_parameter",
    "update_offload_parameter",
    "delete_offload_parameter",
    "has_offloaded_params",
    "disable_hf_hook",
    "align_module_device",
]


def check_accelerate(fallback: Any):
    def decorator(func: Callable[[Any], Any]):
        if not _has_accelerate:

            @wraps(func)
            def fallback_fn(*args, **kwargs):
                return fallback

            return fallback_fn

        return func

    return decorator


""" Candidates for Depreciation """


@check_accelerate(fallback=False)
def is_module_offloaded(module: torch.nn.Module) -> bool:
    return has_offloaded_params(module)


def get_execution_device(module: torch.nn.Module) -> torch.device:
    """
    :param module: module to check
    :return: device module is loaded onto during forward pass
    """
    if has_offloaded_params(module):
        return module._hf_hook.execution_device
    device = next(module.parameters()).device

    # offload only gets set for leaf modules, fallback to checking for device type
    if device.type == "meta":
        return module._hf_hook.execution_device

    return device


def get_offloaded_device(module: torch.nn.Module) -> torch.device:
    """
    :param module: module to check
    :return: device module is offloaded to onto after forward pass
    """
    if has_offloaded_params(module):
        first_key = list(module._hf_hook.weights_map.keys())[0]
        prefix_dataset = module._hf_hook.weights_map.dataset
        return prefix_dataset[first_key].device
    return next(module.parameters()).device


@check_accelerate(fallback=None)
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
    if not has_offloaded_params(module):
        raise ValueError("Prefix dict is only applicable to offloaded modules")

    weights_map = module._hf_hook.weights_map
    offload_to_weights_map(weights_map, key, data)


def update_parameter_data(
    module: torch.nn.Module, new_param_data: torch.Tensor, param_name: str
):
    """
    Update the data of an existing parameter and its offload dict. Supports both
    parameters of offloaded modules and non-offloaded modules

    :param module: module containing the parameter to update
    :param new_param_data: tensor to update parameter with
    :param param_name: name of module parameter to update
    """
    update_offload_parameter(module, param_name, new_param_data)


""" Candidates for Upstreaming """


def register_offload_parameter(
    module: torch.nn.Module,
    name: str,
    parameter: torch.nn.Parameter,
    offload_device: Optional[Union[torch.device, Literal["disk"]]] = None,
):
    """
    Register a parameter to the given module which may be offloaded

    :param module: maybe offloaded module
    :param name: name of newly registered parameter
    :param parameter: parameter being registered
    :param offload_device: device on which weight will be offloaded to. If None is
        provided, then infer device from parameters on module
    """
    has_onload = any(p.device != torch.device("meta") for p in module.parameters())
    module.register_parameter(name, parameter)

    if has_offloaded_params(module):
        weights_map = module._hf_hook.weights_map
        offload_to_weights_map(weights_map, name, parameter.data, offload_device)
        if not has_onload:
            set_module_tensor_to_device(module, name, "meta")


def update_offload_parameter(
    module: torch.nn.Module,
    name: str,
    data: Optional[torch.Tensor],
    offload_device: Optional[Union[torch.device, Literal["disk"]]] = None,
):
    """
    Update the data of an existing parameter and its offload dict. Supports both
    parameters of offloaded modules and non-offloaded modules

    :param module: module containing the parameter to update
    :param name: name of module parameter to update
    :param data: tensor to update parameter with
    :param offload_device: device on which weight will be offloaded to. If None is
        provided, then infer device from parameters on module
    """
    param = getattr(module, name)
    data = data.to(param.dtype)
    if param.data.shape != data.shape:
        warnings.warn(
            f"Shape of parameter being updated {param.data.shape} does not match shape "
            f"of update data {data.shape}"
        )

    # copy data into onloaded parameter if applicable
    if param.device != torch.device("meta"):
        param.data.copy_(data)

    # update offload dict
    if has_offloaded_params(module):
        weights_map = module._hf_hook.weights_map
        offload_to_weights_map(weights_map, name, data, offload_device)


def delete_offload_parameter(module: torch.nn.Module, name: str):
    """
    Delete a parameter from a module which may be offloaded

    :param module: maybe offloaded module
    :param name: name of parameter being deleted
    """
    delattr(module, name)

    if has_offloaded_params(module):
        weights_map = module._hf_hook.weights_map
        delete_from_weights_map(weights_map, name)


@check_accelerate(fallback=contextlib.nullcontext())
@contextlib.contextmanager
def disable_hf_hook(module: torch.nn.Module):
    hooks = {}

    def collect_hooks(module):
        nonlocal hooks
        if hasattr(module, "_hf_hook"):
            hooks[module] = module._hf_hook
            remove_hook_from_module(module)

    module.apply(collect_hooks)

    yield

    for submodule, hook in hooks.items():
        add_hook_to_module(submodule, hook)


@check_accelerate(fallback=None)
def offload_to_weights_map(
    weights_map: Union[PrefixedDataset, Dict, OffloadedWeightsLoader],
    key: str,
    value: torch.Tensor,
    offload_device: Optional[Union[torch.device, Literal["disk"]]] = None,
):
    """
    Helper function which implements offloaded item assignment for PrefixedDataset,
    OffloadedWeightsLoader, and Dict types.

    :param weights_map: weight map to be updated with offload information
    :param key: key used to identify weight location
    :param value: weight being offloaded
    :param offload_device: device on which weight will be offloaded to. If None is
        provided, then infer device from parameters in weights_map
    """
    if isinstance(weights_map, PrefixedDataset):
        if offload_device == "disk":
            raise ValueError(f"Cannot offload to disk with type {type(weights_map)}")

        dataset = weights_map.dataset
        key = f"{weights_map.prefix}{key}"
        offload_to_weights_map(dataset, key, value, offload_device)

    elif isinstance(weights_map, OffloadedWeightsLoader):
        if key not in weights_map.all_keys:
            weights_map.all_keys.append(key)

        if len(weights_map.index) <= 0 and offload_device != "disk":
            offload_to_weights_map(weights_map.state_dict, key, value, offload_device)

        else:
            raise NotImplementedError(
                "Updating weights_map with disk offloading is not implemented yet"
            )

    elif isinstance(weights_map, dict):
        if offload_device == "disk":
            raise ValueError(f"Cannot offload to disk with type {type(weights_map)}")

        # infer offload device
        if offload_device is None:
            if key in weights_map:
                offload_device = weights_map[key].device
            else:
                tens = next(iter(weights_map.values()), None)
                if tens is None:
                    raise ValueError(
                        "Cannot infer offload device from empty weights_map"
                    )
                offload_device = tens.device

        weights_map[key] = value.to(device=offload_device)

    else:
        raise NotImplementedError(
            "Updating offload data not implemented for weights_map of type "
            f"{type(weights_map)}"
        )


@check_accelerate(fallback=None)
def delete_from_weights_map(
    weights_map: Union[PrefixedDataset, Dict, OffloadedWeightsLoader],
    key: str,
):
    if isinstance(weights_map, PrefixedDataset):
        dataset = weights_map.dataset
        key = f"{weights_map.prefix}{key}"
        delete_from_weights_map(dataset, key)

    elif isinstance(weights_map, OffloadedWeightsLoader):
        if len(weights_map.index) <= 0:
            delete_from_weights_map(weights_map.state_dict, key)

        else:
            raise NotImplementedError(
                "Delete from weights_map with disk offloading is not implemented yet"
            )

    elif isinstance(weights_map, dict):
        del weights_map[key]

    else:
        raise NotImplementedError(
            "Updating offload data not implemented for weights_map of type "
            f"{type(weights_map)}"
        )


""" Upstreamed Functions """


# introduced in accelerate v1.1.0
@check_accelerate(fallback=False)
def has_offloaded_params(module: torch.nn.Module) -> bool:
    """
    Checks if a module has offloaded parameters by checking if the given module has a
    AlignDevicesHook attached with offloading enabled

    Args:
        module (`torch.nn.Module`): The module to check for an offload hook.

    Returns:
        bool: `True` if the module has an offload hook and offloading is enabled,
        `False` otherwise.
    """
    return (
        hasattr(module, "_hf_hook")
        and isinstance(module._hf_hook, AlignDevicesHook)
        and module._hf_hook.offload
    )


# introduced in accelerate v1.1.0
@check_accelerate(fallback=contextlib.nullcontext())
@contextlib.contextmanager
def align_module_device(
    module: torch.nn.Module, execution_device: Optional[torch.device] = None
):
    """
    Context manager that moves a module's parameters to the specified execution device.

    Args:
        module (`torch.nn.Module`):
            Module with parameters to align.
        execution_device (`torch.device`, *optional*):
            If provided, overrides the module's execution device within the context.
            Otherwise, use hook execution device or pass
    """
    if has_offloaded_params(module):
        if execution_device is not None:
            original_device = module._hf_hook.execution_device
            module._hf_hook.execution_device = execution_device

        try:
            module._hf_hook.pre_forward(module)
            yield
        finally:
            module._hf_hook.post_forward(module, None)
            if execution_device is not None:
                module._hf_hook.execution_device = original_device

    elif execution_device is not None:
        devices = {
            name: param.device for name, param in module.named_parameters(recurse=False)
        }
        try:
            for name in devices:
                set_module_tensor_to_device(module, name, execution_device)
            yield
        finally:
            for name, device in devices.items():
                set_module_tensor_to_device(module, name, device)

    else:
        yield
