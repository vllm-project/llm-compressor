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

import torch
from compressed_tensors.quantization.lifecycle.forward import quantize
from compressed_tensors.quantization.quant_config import QuantizationStatus
from torch.nn import Module


__all__ = [
    "compress_quantized_weights",
]


_LOGGER = logging.getLogger(__name__)


def compress_quantized_weights(module: Module):
    """
    Quantizes the module weight representation to use fewer bits in memory

    apply to full model with `model.apply(compress_quantized_weights)`

    :param module: module to compress to quantized representation
    """
    scheme = getattr(module, "quantization_scheme", None)
    if not scheme or not scheme.weights:
        # no quantization scheme or weights not quantized, nothing to do
        return

    if scheme is QuantizationStatus.COMPRESSED:
        # module is already compressed, nothing to do
        return

    weight = getattr(module, "weight", None)
    scale = getattr(module, "weight_scale", None)
    zero_point = getattr(module, "weight_zero_point", None)

    if weight is None or scale is None or zero_point is None:
        # no weight, scale, or ZP, nothing to do

        # mark as compressed here to maintain consistent status throughout the model
        module.quantization_status = QuantizationStatus.COMPRESSED
        return

    module.weight.requires_grad = False  # cannot use auto grad after compression
    module.weight.data = quantize(
        x=weight,
        scale=scale,
        zero_point=zero_point,
        args=scheme.weights,
        dtype=torch.int8,
    )

    module.quantization_status = QuantizationStatus.COMPRESSED
