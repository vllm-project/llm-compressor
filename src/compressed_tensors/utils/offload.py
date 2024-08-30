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

import torch
from torch.nn import Module


__all__ = [
    "is_module_offloaded",
    "get_execution_device",
    "get_offloaded_device",
    "update_prefix_dict",
    "update_parameter_data",
]


def is_module_offloaded(module: Module) -> bool:
    """
    :param module: layer to check
    :return: True if layer is offloaded from GPU, False otherwise
    """
    return hasattr(module, "_hf_hook") and module._hf_hook.offload


def get_execution_device(module: Module) -> torch.device:
    """
    :param module: layer to check
    :return: device layer is loaded onto during forward pass
    """
    if is_module_offloaded(module):
        return module._hf_hook.execution_device
    device = next(module.parameters()).device

    # offload only gets set for leaf modules, fallback to checking for device type
    if device.type == "meta":
        return module._hf_hook.execution_device

    return device


def get_offloaded_device(module: Module) -> torch.device:
    """
    :param module: layer to check
    :return: device layer is offloaded to onto after forward pass
    """
    if is_module_offloaded(module):
        first_key = list(module._hf_hook.weights_map.keys())[0]
        prefix_dataset = module._hf_hook.weights_map.dataset
        return prefix_dataset[first_key].device
    return next(module.parameters()).device


def update_prefix_dict(module: Module, key: str, data: torch.Tensor):
    """
    Updates the offloaded state dict for a given module. Parameter named key is replaced
    by data. This is neccesary because parameter updates for offloaded modules do not
    persist automatically between loads. This function only affects the offloaded
    state dict and not the current state of the loaded module.

    :param module: layer containing the parameter to update
    :param key: name of parameter to update
    :param data: tensor to update parameter with in the offloaded state dict
    """
    if not is_module_offloaded(module):
        raise ValueError("Prefix dict is only applicable to offloaded modules")
    prefix_dict = module._hf_hook.weights_map
    prefix_dict.dataset[f"{prefix_dict.prefix}{key}"] = data


def update_parameter_data(
    module: Module, new_param_data: torch.Tensor, param_name: str
):
    """
    Updates the paramter value named param_name for a given module. This function
    updates both the current loaded module state and the offloaded state dict if
    the module is offloaded. This is neccesary because parameter updates for offloaded
    modules do not persist automatically between loads.

    :param module: layer containing the parameter to update
    :param new_param_data: tensor to update parameter with
    :param param_name: name of layer parameter to update
    """
    if not hasattr(module, param_name):
        return

    device = next(module.parameters()).device

    offloaded = False
    if is_module_offloaded(module):
        offload_device = get_offloaded_device(module)
        offloaded = True

    parameter = getattr(module, param_name, None)
    if parameter is None:
        raise ValueError("Attempted to update uninitialized parameter")

    dtype = parameter.dtype
    parameter.data = new_param_data.to(device).to(dtype)

    if offloaded:
        prefix_dict = module._hf_hook.weights_map.dataset
        prefix = module._hf_hook.weights_map.prefix
        prefix_dict[f"{prefix}{param_name}"] = new_param_data.to(offload_device).to(
            dtype
        )
