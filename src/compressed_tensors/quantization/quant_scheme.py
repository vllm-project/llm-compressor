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

from copy import deepcopy
from typing import List, Optional

from compressed_tensors.quantization.quant_args import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from pydantic import BaseModel


__all__ = [
    "QuantizationScheme",
    "preset_name_to_scheme",
    "is_preset_scheme",
]


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
            output_activations=output_activations,
        )


"""
Pre-Set Quantization Scheme Args
"""


def preset_name_to_scheme(name: str, targets: List[str]) -> QuantizationScheme:
    """
    :param name: preset quantization settings name. must exist in upper case in
        PRESET_SCHEMES
    :param targets: list of quantization targets to be passed to the Scheme
    :return: new QuantizationScheme for a given name with the given targets
    """
    name = name.upper()

    if name not in PRESET_SCHEMES:
        raise KeyError(
            f"Unknown preset scheme name {name}, "
            f"available names: {list(PRESET_SCHEMES.keys())}"
        )

    scheme_args = deepcopy(PRESET_SCHEMES[name])  # deepcopy to avoid args references
    return QuantizationScheme(
        targets=targets,
        **scheme_args,
    )


def is_preset_scheme(name: str) -> bool:
    """
    :param name: preset quantization settings name
    :return: True if the name is a preset scheme name
    """
    return name.upper() in PRESET_SCHEMES


W8A8 = dict(
    weights=QuantizationArgs(
        num_bits=8,
        symmetric=True,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.CHANNEL,
    ),
    input_activations=QuantizationArgs(
        num_bits=8,
        symmetric=True,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.TOKEN,
        dynamic=True,
    ),
)

W8A16 = dict(
    weights=QuantizationArgs(
        num_bits=8,
        symmetric=True,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.CHANNEL,
    )
)

W4A16 = dict(
    weights=QuantizationArgs(
        num_bits=4,
        symmetric=True,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.GROUP,
        group_size=128,
    )
)

FP8 = dict(
    weights=QuantizationArgs(
        num_bits=8,
        symmetric=True,
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.TENSOR,
    ),
    input_activations=QuantizationArgs(
        num_bits=8,
        symmetric=True,
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.TENSOR,
        dynamic=False,
    ),
)

PRESET_SCHEMES = {"W8A8": W8A8, "W8A16": W8A16, "W4A16": W4A16, "FP8": FP8}
