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

from typing import List, Optional

from compressed_tensors.quantization.quant_args import QuantizationArgs
from pydantic import BaseModel


__all__ = ["QuantizationScheme"]


class QuantizationScheme(BaseModel):
    """
    Set of QuantizationArgs defining how the weights, inputs and outputs of target list
    of modules should be quantized

    :param targets: list of modules to apply the QuantizationArgs to, can be layer
    names, layer types or a regular expression
    :param weights: quantization config for layer weights
    :param input_activations: quantization config for layer inputs
    :param output_activations: quantization config for layer outputs
    """

    targets: List[str]
    weights: Optional[QuantizationArgs] = None
    input_activations: Optional[QuantizationArgs] = None
    output_activations: Optional[QuantizationArgs] = None
    
    @classmethod
    def default_scheme(
        cls,
        targets: Optional[List[str]] = None,
    ):
    
        if targets is None:
            # default to quantizing all Linear layers
            targets = ["Linear"]
        
        # default to 8 bit integer symmetric quantization
        # for weights
        weights = QuantizationArgs(num_bits=8, symmetric=True)
        
        # default to 8 bit integer asymmetric quantization
        input_activations = QuantizationArgs(num_bits=8, symmetric=True)
        
        # Do not quantize the output activations
        # by default
        output_activations = None
        
        return cls(
            targets=targets,
            weights=weights,
            input_activations=input_activations,
            output_activations=output_activations,)
        
        
