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

import torch
from compressed_tensors.config import CompressionFormat, SparsityStructure
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from compressed_tensors.quantization.utils import is_module_quantized
from loguru import logger


__all__ = ["infer_and_set_per_module_quantization_format"]


def _get_quant_compression_format(
    input_args: Optional[QuantizationArgs],
    weight_args: Optional[QuantizationArgs],
    sparsity_structure: Optional[str] = None,
) -> CompressionFormat:
    """
    Using the weight and input quantization args as well as an optional
    sparsity structure, determine the compression format that should be
    applied to a given module

    :param input_args: input quantization parameters
    :param weight_args: weight quantization parameters
    :param sparsity_structure: optional (global) modle sparsity
        structure
    :return CompresssionFormat for the module
    """
    is_24_structure = (
        SparsityStructure(sparsity_structure) == SparsityStructure.TWO_FOUR
    )
    is_weight_only = weight_args is not None and input_args is None

    if weight_args.num_bits == 4 and weight_args.type == QuantizationType.FLOAT.value:
        return CompressionFormat.nvfp4_pack_quantized

    if is_weight_only:  # w4a16 and w8a16
        is_valid_pack = (
            weight_args.num_bits in [4, 8]
            and weight_args.type == QuantizationType.INT.value
        )
        if not is_valid_pack:  # packing only valid for int4 and int 8
            return CompressionFormat.naive_quantized

        if is_24_structure and weight_args.strategy in (
            QuantizationStrategy.CHANNEL.value,
            QuantizationStrategy.GROUP.value,
        ):
            # marlin24 kernel only applicable for channel/group quantization
            # Note: vLLM may only support group quant for marlin24
            return CompressionFormat.marlin_24
        return CompressionFormat.pack_quantized

    else:  # w8a8 float and int
        if (
            weight_args.type == QuantizationType.FLOAT.value
            and weight_args.num_bits == 8
        ):
            return CompressionFormat.float_quantized
        if weight_args.type == QuantizationType.INT.value:
            return CompressionFormat.int_quantized

        return CompressionFormat.naive_quantized


def set_per_module_format(
    module: torch.nn.Module,
    sparsity_structure: Optional[str] = None,
    quantization_format: Optional[str] = None,
):
    """
    Determine and set the per module quantization format given quantization args
    and sparsity structure.

    :param module: module which has its quantization inferred
    :param sparsity_structure: optional sparsity applied to the module
    :param quantization_format: optional global format to override
        the per module formats

    """
    weight_scheme = module.quantization_scheme.weights
    input_scheme = module.quantization_scheme.input_activations
    if weight_scheme is None:
        return  # no weight quant - nothing to compress
    compression_format = _get_quant_compression_format(
        input_scheme, weight_scheme, sparsity_structure
    )

    # Check if a global format was provided first
    # This will override any per module format
    if quantization_format is not None:
        if quantization_format != compression_format.value:
            logger.warning(
                "The provided format for the module does not match the "
                "inferred format. Compression may fail "
            )
        module.quantization_scheme.format = quantization_format
    # If a per module format is not provided, we check if it matches our inferred one
    elif module.quantization_scheme.format is not None:
        # If it does not, warn the user
        if module.quantization_scheme.format != compression_format.value:
            logger.warning(
                "The provided format for the module does not match the "
                "inferred format. Compression may fail "
            )
    # If neither provided, set ours
    else:
        module.quantization_scheme.format = compression_format.value


def infer_and_set_per_module_quantization_format(
    model: torch.nn.Module,
    sparsity_structure: Optional[str] = None,
    quantization_format: Optional[str] = None,
) -> List[str]:
    """
    Infers the quantization format for a model based on its state and provided
    compression arguments. Updates thhe quantization_scheme.format value
    based on the inferred format. Returns the unique list of formats in the model.
    All None formats are mapped to CompressionFormat.dense.value

    For a summary of the formats, see `docs/guides/compression_formats.md`.

    :param model: model to check for quantization
    :param sparsity_structure: optional sparsity applied to the module
    :param quantization_format: optional global format to override
        the per module formats
    :return compression format appropriate for the model
    """
    unique_formats = []
    for submodule in model.modules():
        if is_module_quantized(submodule):
            assert hasattr(submodule, "quantization_scheme")
            set_per_module_format(submodule, sparsity_structure, quantization_format)
            if (
                submodule.quantization_scheme.format
                and submodule.quantization_scheme.format not in unique_formats
            ):
                unique_formats.append(submodule.quantization_scheme.format)

    if len(unique_formats) > 0:
        return unique_formats
    return [CompressionFormat.dense.value]
