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
Miscelaneous helpers for the quantization lifecycle
"""

from typing import Optional

import torch
from torch.nn import Module


__all__ = [
    "update_layer_weight_quant_params",
    "enable_quantization",
    "disable_quantization",
]


def update_layer_weight_quant_params(
    layer: Module,
    weight: Optional[torch.Tensor] = None,
    g_idx: Optional[torch.Tensor] = None,
    reset_obs: bool = False,
):
    """
    Update quantization parameters on layer

    :param layer: input layer
    :param weight: weight to update quant params with, defaults to layer weight
    :param g_idx: optional mapping from column index to group index
    :param reset_obs: reset the observer before calculating quant params,
        defaults to False
    """
    attached_weight = getattr(layer, "weight", None)

    if weight is None:
        weight = attached_weight
    scale = getattr(layer, "weight_scale", None)
    zero_point = getattr(layer, "weight_zero_point", None)
    if g_idx is None:
        g_idx = getattr(layer, "weight_g_idx", None)
    observer = getattr(layer, "weight_observer", None)

    if weight is None or observer is None or scale is None or zero_point is None:
        # scale, zp, or observer not calibratable or weight not available
        return

    if reset_obs:
        observer.reset()

    if attached_weight is not None:
        weight = weight.to(attached_weight.dtype)

    updated_scale, updated_zero_point = observer(weight)

    # update scale and zero point
    device = next(layer.parameters()).device
    scale.data = updated_scale.to(device)
    zero_point.data = updated_zero_point.to(device)


def enable_quantization(module: Module):
    module.quantization_enabled = True


def disable_quantization(module: Module):
    module.quantization_enabled = False
