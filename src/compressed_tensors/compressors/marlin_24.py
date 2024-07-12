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
from typing import Dict, Generator, Tuple

import numpy as np
import torch
from compressed_tensors.compressors import Compressor
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy
from compressed_tensors.quantization.lifecycle.forward import quantize
from compressed_tensors.utils import (
    get_permutations_24,
    is_quantization_param,
    merge_names,
    sparse_semi_structured_from_dense_cutlass,
    tensor_follows_mask_structure,
)
from torch import Tensor
from tqdm import tqdm


_LOGGER: logging.Logger = logging.getLogger(__name__)


@Compressor.register(name=CompressionFormat.marlin_24.value)
class Marlin24Compressor(Compressor):
    """
    Compresses a quantized model with 2:4 sparsity structure for inference with the
    Marlin24 kernel. Decompression is not implemented for this compressor.
    """

    COMPRESSION_PARAM_NAMES = ["weight_packed", "scale_packed", "meta"]

    @staticmethod
    def validate_quant_compatability(
        model_quant_args: Dict[str, QuantizationArgs]
    ) -> bool:
        """
        Checks if every quantized module in the model is compatible with Marlin24
        compression. Quantization must be channel or group strategy with group_size
        of 128. Only symmetric quantization is supported

        :param model_quant_args: dictionary of mapping module names to their
            quantization configuration
        :return: True if all modules are compatible with Marlin24 compression, raises
            a ValueError otherwise
        """
        for name, quant_args in model_quant_args.items():
            strategy = quant_args.strategy
            group_size = quant_args.group_size
            symmetric = quant_args.symmetric
            if (
                strategy is not QuantizationStrategy.GROUP.value
                and strategy is not QuantizationStrategy.CHANNEL.value
            ):
                raise ValueError(
                    f"Marlin24 Compressor is only valid for group and channel "
                    f"quantization strategies, got {strategy} in {name}"
                )

            if group_size is not None and group_size != 128:
                raise ValueError(
                    f"Marlin24 Compressor is only valid for group size 128, "
                    f"got {group_size} in {name}"
                )

            if not symmetric:
                raise ValueError(
                    f"Marlin24 Compressor is only valid for symmetric quantzation, "
                    f"got symmetric={symmetric} in {name}"
                )

        return True

    @staticmethod
    def validate_sparsity_structure(name: str, weight: Tensor) -> bool:
        """
        Checks if a tensor fits the required 2:4 sparsity structure

        :param name: name of the tensor to check
        :param weight: tensor to check for sparsity structure
        :return: True if all rows match the 2:4 sparsity structure, raises
            ValueError otherwise
        """

        if not tensor_follows_mask_structure(weight):
            raise ValueError(
                "Marlin24 Compressor is only compatible with weights that have "
                f"a 2:4 sparsity structure. Found segments in {name} "
                "that do not match the expected structure."
            )

        return True

    def compress(
        self,
        model_state: Dict[str, Tensor],
        names_to_scheme: Dict[str, QuantizationArgs],
        **kwargs,
    ) -> Dict[str, Tensor]:
        """
        Compresses a quantized state_dict with 2:4 sparsity structure for inference
        with the Marlin24 kernel

        :param model_state: state dict of uncompressed model
        :param names_to_scheme: quantization args for each quantized weight, needed for
           quantize function to calculate bit depth
        :return: compressed state dict
        """
        self.validate_quant_compatability(names_to_scheme)

        compressed_dict = {}
        weight_suffix = ".weight"
        _LOGGER.debug(
            f"Compressing model with {len(model_state)} parameterized layers..."
        )

        for name, value in tqdm(model_state.items(), desc="Compressing model"):
            if name.endswith(weight_suffix):
                prefix = name[: -(len(weight_suffix))]
                scale = model_state.get(merge_names(prefix, "weight_scale"), None)
                zp = model_state.get(merge_names(prefix, "weight_zero_point"), None)
                if scale is not None:  # weight is quantized, compress it

                    # Marlin24 kernel requires float16 inputs
                    scale = scale.to(torch.float16)
                    value = value.to(torch.float16)

                    # quantize weight, keeping it as a float16 for now
                    quant_args = names_to_scheme[prefix]
                    value = quantize(
                        x=value, scale=scale, zero_point=zp, args=quant_args
                    )

                    # compress based on sparsity structure
                    self.validate_sparsity_structure(prefix, value)
                    value, meta = compress_weight_24(value)
                    meta = meta.cpu()

                    # Marlin24 kernel expects input dim first
                    value = value.t().contiguous().cpu()
                    scale = scale.t().contiguous().cpu()
                    og_weight_shape = value.shape

                    # Marlin24 kernel expects unsigned values, shift zero-point
                    value += (1 << quant_args.num_bits) // 2

                    # pack quantized weight and scale
                    value = pack_weight_24(value, quant_args)
                    packed_scale = pack_scales_24(scale, quant_args, og_weight_shape)
                    meta = meta.resize_(meta.shape[1] // 2, meta.shape[0] * 2)

                    # save compressed values
                    compressed_dict[merge_names(prefix, "scale_packed")] = packed_scale
                    compressed_dict[merge_names(prefix, "weight_packed")] = value
                    compressed_dict[merge_names(prefix, "meta")] = meta
                    continue

            if not is_quantization_param(name):
                # export unquantized parameters without modifying
                compressed_dict[name] = value.to("cpu")

        return compressed_dict

    def decompress(
        self, path_to_model_or_tensors: str, device: str = "cpu", **kwargs
    ) -> Generator[Tuple[str, Tensor], None, None]:
        raise NotImplementedError(
            "Decompression is not implemented for the Marlin24 Compressor."
        )


def compress_weight_24(weight: Tensor):
    weight = weight.contiguous()
    w_comp, meta = sparse_semi_structured_from_dense_cutlass(weight)
    w_comp = w_comp.contiguous()
    return w_comp, meta


def marlin_permute_weights(q_w, size_k, size_n, perm, tile):
    assert q_w.shape == (size_k, size_n)
    assert size_k % tile == 0, f"size_k = {size_k}, tile = {tile}"
    assert size_n % tile == 0, f"size_k = {size_n}, tile = {tile}"

    # Permute weights to 16x64 marlin tiles
    q_w = q_w.reshape((size_k // tile, tile, size_n // tile, tile))
    q_w = q_w.permute((0, 2, 1, 3))
    q_w = q_w.reshape((size_k // tile, size_n * tile))

    q_w = q_w.reshape((-1, perm.numel()))[:, perm].reshape(q_w.shape)

    return q_w


def pack_weight_24(
    weight: Tensor,
    quantization_args: QuantizationArgs,
    tile: int = 16,
):
    size_k = weight.shape[0]
    size_n = weight.shape[1]
    num_bits = quantization_args.num_bits
    pack_factor = 32 // num_bits

    # Reshuffle to marlin_24 format
    perm, _, _ = get_permutations_24(num_bits)
    q_w = marlin_permute_weights(weight, size_k, size_n, perm, tile)

    q_w = q_w.cpu().numpy().astype(np.uint32)

    q_packed = np.zeros((q_w.shape[0], q_w.shape[1] // pack_factor), dtype=np.uint32)
    for i in range(pack_factor):
        q_packed |= q_w[:, i::pack_factor] << num_bits * i

    q_packed = torch.from_numpy(q_packed.astype(np.int32))

    return q_packed


def pack_scales_24(scales, quantization_args, w_shape):
    size_k = w_shape[0]
    size_n = w_shape[1]
    num_bits = quantization_args.num_bits

    _, scale_perm_2_4, scale_perm_single_2_4 = get_permutations_24(num_bits)

    if (
        quantization_args.strategy is QuantizationStrategy.GROUP
        and quantization_args.group_size < size_k
    ):
        scales = scales.reshape((-1, len(scale_perm_2_4)))[:, scale_perm_2_4]
    else:  # channelwise
        scales = scales.reshape((-1, len(scale_perm_single_2_4)))[
            :, scale_perm_single_2_4
        ]
    scales = scales.reshape((-1, size_n)).contiguous()

    return scales
