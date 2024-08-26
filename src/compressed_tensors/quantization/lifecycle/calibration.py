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


import logging

from compressed_tensors.quantization.quant_config import QuantizationStatus
from compressed_tensors.utils import is_module_offloaded, update_parameter_data
from torch.nn import Module


__all__ = [
    "set_module_for_calibration",
]


_LOGGER = logging.getLogger(__name__)


def set_module_for_calibration(module: Module, quantize_weights_upfront: bool = True):
    """
    marks a layer as ready for calibration which activates observers
    to update scales and zero points on each forward pass

    apply to full model with `model.apply(set_module_for_calibration)`

    :param module: module to set for calibration
    :param quantize_weights_upfront: whether to automatically
       run weight quantization at the start of calibration
    """
    if not getattr(module, "quantization_scheme", None):
        # no quantization scheme nothing to do
        return
    status = getattr(module, "quantization_status", None)
    if not status or status != QuantizationStatus.INITIALIZED:
        _LOGGER.warning(
            f"Attempting set module with status {status} to calibration mode. "
            f"but status is not {QuantizationStatus.INITIALIZED} - you may "
            "be calibrating an uninitialized module which may fail or attempting "
            "to re-calibrate a frozen module"
        )

    if quantize_weights_upfront and module.quantization_scheme.weights is not None:
        # set weight scale and zero_point up front, calibration data doesn't affect it
        observer = module.weight_observer
        g_idx = getattr(module, "weight_g_idx", None)

        offloaded = False
        if is_module_offloaded(module):
            module._hf_hook.pre_forward(module)
            offloaded = True

        scale, zero_point = observer(module.weight, g_idx=g_idx)
        update_parameter_data(module, scale, "weight_scale")
        update_parameter_data(module, zero_point, "weight_zero_point")

        if offloaded:
            module._hf_hook.post_forward(module, None)

    module.quantization_status = QuantizationStatus.CALIBRATION
