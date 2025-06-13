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

| ------------------------------------------------------------------------------------------------------ | # noqa: E501
| Operation  | Without offloading support             | With offloading support                          | # noqa: E501
| ---------- | -------------------------------------- | ------------------------------------------------ | # noqa: E501
| Add        | module.register_parameter(name, param) | register_offload_parameter(module, name, param)  | # noqa: E501
| Check      | N/A                                    | has_offloaded_params(module)                     | # noqa: E501
| Onload     | N/A                                    | with align_module_device(module)                 | # noqa: E501
| Update     | module.name.data.copy_(new_data)       | update_offload_parameter(module, name, new_data) | # noqa: E501
| Delete     | del module.name                        | delete_offload_parameter(module, name)           | # noqa: E501
| Add Module | module.register_module(name, child)    | register_offload_module(name, child)             | # noqa: E501
| Del Module | del module.name                        | delete_offload_module(module, name)              | # noqa: E501
| ------------------------------------------------------------------------------------------------------ | # noqa: E501
"""

import contextlib
import warnings
from functools import wraps
from operator import attrgetter
from typing import Any, Callable, Dict, Iterable, Literal, Optional, Tuple, Union

import torch
from compressed_tensors.utils import patch_attr


try:
    from accelerate.hooks import (
        AlignDevicesHook,
        add_hook_to_module,
        attach_align_device_hook,
        named_module_tensors,
        remove_hook_from_module,
    )
    from accelerate.utils import (
        OffloadedWeightsLoader,
        PrefixedDataset,
        find_tied_parameters,
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
    named_module_tensors = None
    attach_align_device_hook = None
    find_tied_parameters = None


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
    "disable_offload",
    "align_modules",
    "align_module_device",
    "register_offload_module",
    "delete_offload_module",
    "offloaded_dispatch",
    "disable_offloading",
]


def check_accelerate(fallback: Any):
    def decorator(func: Callable[[Any], Any]):
        if not _has_accelerate:
            if fallback == "error":

                @wraps(func)
                def fallback_fn(*args, **kwargs):
                    raise ValueError(
                        "Please install `accelerate` in order to use this function"
                    )

            else:

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


def get_execution_device(module: torch.nn.Module) -> torch.device:
    """
    Get the device which inputs should be moved to before module execution

    :param module: module to check, may be offloaded
    :return: onload device of module
    """
    if has_offloaded_params(module):
        return module._hf_hook.execution_device

    first_param = next(module.parameters(), None)
    if first_param is None:
        warnings.warn(
            f"Unable able to infer execution device of {module}, falling back to CPU"
        )
        return torch.device("cpu")

    return first_param.device


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
    data: torch.Tensor,
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
    param: torch.nn.Parameter = getattr(module, name)
    if param.data.shape != data.shape:
        warnings.warn(
            f"Shape of parameter being updated {param.data.shape} does not match shape "
            f"of update data {data.shape}"
        )

    # copy data into onloaded parameter if applicable
    if param.device != torch.device("meta") and data is not param.data:
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


@check_accelerate(fallback=contextlib.nullcontext())
@contextlib.contextmanager
def disable_offload(module: torch.nn.Module):
    """
    Context manager to disable module onloading and offloading. Parameters will stay on
    their current device

    :param module: module to disable offloading for
    """
    if has_offloaded_params(module):
        module._hf_hook.offload = False
        yield
        module._hf_hook.offload = True
    else:
        yield


@check_accelerate(fallback=contextlib.nullcontext())
@contextlib.contextmanager
def align_modules(
    modules: Union[torch.nn.Module, Iterable[torch.nn.Module]],
    execution_device: Optional[torch.device] = None,
):
    """
    Context manager for onloading modules to a device, and disabling onload and offload
    attempts triggered by forward calls. Used for sequential onloading of layers

    :param modules: `torch.nn.Module` or iterable of `torch.nn.Module`s to onload
    :param execution_device: device to onload to
    """
    modules = (modules,) if isinstance(modules, torch.nn.Module) else modules

    with contextlib.ExitStack() as stack:
        for module in modules:
            stack.enter_context(align_module_device(module, execution_device))
            stack.enter_context(disable_offload(module))  # disable redundant onloading
        yield


def register_offload_module(base: torch.nn.Module, name: str, module: torch.nn.Module):
    """
    Register a submodule with offloading if the parent module is offloaded

    :param base: module to attach submodule to
    :param name: name of submodule
    :param module: submodule to attach
    """

    if has_offloaded_params(base):
        hook: AlignDevicesHook = base._hf_hook
        assert hook.offload
        assert hook.weights_map is not None
        assert hook.tied_params_map is not None

        # offloading kwargs for submodule
        place_submodules = False
        offload_buffers = True

        # copy device offloading arguments from parent
        current_device = next(base.parameters()).device  # assume base has parameters
        offload_device = get_offloaded_device(base)

        # offload parameters to weights map
        for param_name, param in named_module_tensors(
            module, include_buffers=offload_buffers, recurse=place_submodules
        ):
            offloaded = param.to(offload_device)
            hook.tied_params_map[offloaded.data_ptr()] = {}  # (1)
            offload_to_weights_map(hook.weights_map, f"{name}.{param_name}", offloaded)

            # if the parent places submodules, offload here
            if hook.place_submodules:
                set_module_tensor_to_device(module, param_name, current_device)

        # if the parent does not place submodules, then add a hook
        # parameters are offloaded by `add_hook_to_module`
        if not hook.place_submodules:
            weights_map = PrefixedDataset(
                hook.weights_map.dataset, prefix=f"{hook.weights_map.prefix}{name}."
            )

            submodule_hook = AlignDevicesHook(
                execution_device=hook.execution_device,
                offload=hook.offload,
                io_same_device=False,
                weights_map=weights_map,
                offload_buffers=offload_buffers,
                place_submodules=place_submodules,
                skip_keys=None,
                tied_params_map=hook.tied_params_map,
            )
            add_hook_to_module(module, submodule_hook)

    base.register_module(name, module)

    # (1): Since we cannot know which pointers are shared when we add parameters in an
    # online way, assume that all pointers are shared. This comes at no runtime cost


def delete_offload_module(base: torch.nn.Module, name: str):
    """
    Delete a submodule from a model which may contain offloading
    :param base: parent module to delete submodule from
    :param name: name of submodule on parent
    """
    module: torch.nn.Module = getattr(base, name)

    for param_name, _ in list(module.named_parameters()):
        delete_offload_parameter(module, param_name)

    delattr(base, name)


@check_accelerate(fallback="error")
def offloaded_dispatch(
    module: torch.nn.Module,
    execution_device: torch.device,
    offload_device: Union[torch.device, Literal["disk"]] = torch.device("cpu"),
) -> torch.nn.Module:
    """
    Unlike `dispatch_model`, this function forces a module (and its submodules) to
    offload all parameters and replace them with meta tensors, utiliizing the
    `AlignDevicesHook` to control onloading and offloading.

    :param module: module containing parameters to offload
    :param execution_device: device that modules will onload and execute on
    :param offload_device: device that module parameters will offload to
    :return: module with offloading device hooks
    """
    if offload_device == "disk":
        raise NotImplementedError("Disk offloading is not currently supported")

    # create weights map
    state_dict = module.state_dict()
    state_dict = {key: val.to(offload_device) for key, val in state_dict.items()}
    weights_map = OffloadedWeightsLoader(state_dict=state_dict, device=offload_device)

    # create tied params map
    tied_params = find_tied_parameters(module)
    tied_params_map = {}
    for group in tied_params:
        for param_name in group:
            data_ptr = attrgetter(param_name)(module).data_ptr()
            tied_params_map[data_ptr] = {}

    # recursively attaches hooks to all submodules
    attach_align_device_hook(
        module,
        execution_device=execution_device,
        offload=True,
        weights_map=weights_map,
        tied_params_map=tied_params_map,
    )
    return module


@contextlib.contextmanager
def disable_offloading():
    """
    Keep modules onloaded and disable offloading until this context exits.
    Affects modules which have been hooked with accelerate's `AlignDevicesHook`
    """
    original_pre_forward = AlignDevicesHook.pre_forward
    onloaded_modules: Dict[torch.nn.Module, Tuple[AlignDevicesHook, bool]] = dict()

    # onload once and disable any future onloading/offloading steps
    def keep_onload_pre_forward(self: AlignDevicesHook, module, *args, **kwargs):
        ret = original_pre_forward(self, module, *args, **kwargs)
        if module not in onloaded_modules:
            onloaded_modules[module] = (self, self.offload)
            self.offload = False
        return ret

    # use the patched pre_forward function within the context
    with patch_attr(AlignDevicesHook, "pre_forward", keep_onload_pre_forward):
        yield

    # manually offload all modules that were onloaded
    # update any parameters which may have changed
    for module, (hook, offload) in onloaded_modules.items():
        hook.offload = offload
        for name, param in module.named_parameters():
            update_offload_parameter(module, name, param.data)
        hook.post_forward(module, None)


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
