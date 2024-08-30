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
from compressed_tensors.compressors.base import Compressor
from compressed_tensors.quantization import (
    QuantizationScheme,
    QuantizationStatus,
    initialize_module_for_quantization,
)
from torch import Tensor
from torch.nn import Parameter
from torch.nn.functional import linear
from torch.nn.modules import Linear


class CompressedLinear(Linear):
    """
    Wrapper module for running a compressed forward pass of a quantized Linear module.
    The wrapped layer will decompressed on each forward call.

    :param module: dense linear module to replace
    :param quantization_scheme: quantization config for the module to wrap
    :param quantization_format: compression format module is stored as
    """

    @classmethod
    @torch.no_grad()
    def from_linear(
        cls,
        module: Linear,
        quantization_scheme: QuantizationScheme,
        quantization_format: str,
    ):
        module.__class__ = CompressedLinear
        module.compressor = Compressor.load_from_registry(quantization_format)
        device = next(module.parameters()).device

        # this will initialize all the scales and zero points
        initialize_module_for_quantization(
            module, quantization_scheme, force_zero_point=False
        )

        # get the shape and dtype of compressed parameters
        compression_params = module.compressor.compression_param_info(
            module.weight.shape, quantization_scheme.weights
        )

        # no need for this once quantization is initialized, will be replaced
        # with the compressed parameter
        delattr(module, "weight")

        # populate compressed weights and quantization parameters
        for name, (shape, dtype) in compression_params.items():
            param = Parameter(
                torch.empty(shape, device=device, dtype=dtype), requires_grad=False
            )
            module.register_parameter(name, param)

        # mark module as compressed
        module.quantization_status = QuantizationStatus.COMPRESSED

        # handles case where forward is wrapped in new_forward by accelerate hooks
        if hasattr(module, "_old_forward"):
            module._old_forward = CompressedLinear.forward.__get__(
                module, CompressedLinear
            )

        return module

    def forward(self, input: Tensor) -> Tensor:
        """
        Decompresses the weight, then runs the wrapped forward pass
        """
        uncompressed_weight = self.compressor.decompress_module(self)
        return linear(input, uncompressed_weight, self.bias)
