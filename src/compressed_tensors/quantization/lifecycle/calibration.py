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
from torch.nn import Module


__all__ = [
    "set_module_for_calibration",
]


_LOGGER = logging.getLogger(__name__)


def set_module_for_calibration(module: Module):
    """
    marks a layer as ready for calibration which activates observers
    to update scales and zero points on each forward pass

    apply to full model with `model.apply(set_module_for_calibration)`

    :param module: module to set for calibration
    """
    if not getattr(module, "quantization_scheme", None):
        # no quantization scheme nothing to do
        return
    status = getattr(module, "quantization_status", None)
    if not status or status != QuantizationStatus.INITIALIZED:
        raise _LOGGER.warning(
            f"Attempting set module with status {status} to calibration mode. "
            f"but status is not {QuantizationStatus.INITIALIZED} - you may "
            "be calibrating an uninitialized module which may fail or attempting "
            "to re-calibrate a frozen module"
        )

    module.quantization_status = QuantizationStatus.CALIBRATION
