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


from compressed_tensors.quantization.quant_config import QuantizationStatus
from torch.nn import Module


__all__ = [
    "freeze_module_quantization",
]


def freeze_module_quantization(module: Module):
    """
    deletes observers so static quantization is completed.

    apply to full model with `model.apply(freeze_module_quantization)`

    :param module: module to freeze quantization for
    """
    scheme = getattr(module, "quantization_scheme", None)
    if not scheme:
        # no quantization scheme nothing to do
        return

    if module.quantization_status == QuantizationStatus.FROZEN:
        # nothing to do, already frozen
        return

    # delete observers from module if not dynamic
    if hasattr(module, "input_observer") and not scheme.input_activations.dynamic:
        delattr(module, "input_observer")
    if hasattr(module, "weight_observer") and not scheme.weights.dynamic:
        delattr(module, "weight_observer")
    if hasattr(module, "output_observer") and not scheme.output_activations.dynamic:
        delattr(module, "output_observer")

    module.quantization_status = QuantizationStatus.FROZEN
