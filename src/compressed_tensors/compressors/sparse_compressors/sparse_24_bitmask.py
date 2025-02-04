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

from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import torch
from compressed_tensors.compressors.base import BaseCompressor
from compressed_tensors.compressors.sparse_compressors.base import BaseSparseCompressor
from compressed_tensors.config import CompressionFormat, SparsityStructure
from compressed_tensors.quantization import FP8_DTYPE
from compressed_tensors.utils import merge_names, pack_bitmasks, unpack_bitmasks
from torch import Tensor


__all__ = [
    "Sparse24BitMaskCompressor",
    "Sparse24BitMaskTensor",
    "sparse24_bitmask_compress",
    "sparse24_bitmask_decompress",
    "get_24_bytemasks",
]


@BaseCompressor.register(name=CompressionFormat.sparse_24_bitmask.value)
class Sparse24BitMaskCompressor(BaseSparseCompressor):
    """
    Compression for sparse models using bitmasks. Non-zero weights are stored in a 2d
    values tensor, with their locations stored in a 2d bitmask
    """

    @property
    def compression_param_names(self) -> Tuple[str]:
        """
        Returns a tuple of compression parameter names introduced by
        the compressor during compression
        """
        return (
            "shape",
            "compressed",
            "bitmask",
        )

    def compress_weight(self, name, value):
        bitmask_tensor = Sparse24BitMaskTensor.from_dense(
            value, self.config.sparsity_structure
        )
        bitmask_dict = bitmask_tensor.dict(name_prefix=name, device="cpu")
        return bitmask_dict

    def decompress_weight(self, weight_data):
        data = Sparse24BitMaskTensor.from_compressed_data(**weight_data)
        decompressed = data.decompress()
        return decompressed


@dataclass
class Sparse24BitMaskTensor:
    """
    Owns compressions and decompression for a single 2:4 sparse
    bitmask compressed tensor.

    :param shape: shape of dense tensor
    :param compressed: 2d tensor of non-zero values
    :param bitmask: 2d bitmask of non-zero values
    """

    shape: List[int]
    compressed: Tensor
    bitmask: Tensor

    @staticmethod
    def from_dense(
        tensor: Tensor,
        sparsity_structure: Union[SparsityStructure, str] = SparsityStructure.TWO_FOUR,
    ) -> "Sparse24BitMaskTensor":
        """
        :param tensor: dense tensor to compress
        :return: instantiated compressed tensor
        """
        shape = list(tensor.shape)
        compressed, bitmask = sparse24_bitmask_compress(
            tensor.cpu(), sparsity_structure=sparsity_structure
        )
        return Sparse24BitMaskTensor(
            shape=shape,
            compressed=compressed,
            bitmask=bitmask,
        )

    @staticmethod
    def from_compressed_data(
        shape: Union[List[int], Tensor], compressed: Tensor, bitmask: Tensor
    ) -> "Sparse24BitMaskTensor":
        """
        :param shape: shape of the dense tensor (can be a list or a tensor)
        :param compressed: 2d tensor of non-zero values
        :param bitmask: 2d bitmask of non-zero values
        :return: instantiated Sparse24BitMaskTensor
        """
        if isinstance(shape, list):
            shape = torch.tensor(shape)
        if isinstance(shape, torch.Tensor):
            shape = shape.flatten().tolist()
        return Sparse24BitMaskTensor(
            shape=shape, compressed=compressed, bitmask=bitmask
        )

    def decompress(self) -> Tensor:
        """
        :return: reconstructed dense tensor
        """
        return sparse24_bitmask_decompress(self.compressed, self.bitmask, self.shape)

    def curr_memory_size_bytes(self) -> int:
        """
        :return: size in bytes required to store compressed tensor on disk
        """

        def sizeof_tensor(a: Tensor) -> int:
            return a.element_size() * a.nelement()

        return sizeof_tensor(self.compressed) + sizeof_tensor(self.bitmask)

    def dict(self, name_prefix: str, device: str = "cpu") -> Dict[str, Tensor]:
        """
        :param name_prefix: name of original tensor to store compressed weight as
        :return: dict of compressed data for the stored weight
        """
        if name_prefix.endswith(".weight"):
            name_prefix = name_prefix[: -len(".weight")]
        return {
            merge_names(name_prefix, "shape"): torch.tensor(
                self.shape, device=device
            ).reshape(-1, 1),
            merge_names(name_prefix, "compressed"): self.compressed.to(device),
            merge_names(name_prefix, "bitmask"): self.bitmask.to(device),
        }

    def __repr__(self) -> str:
        return f"BitMaskTensor(shape={self.shape}, compressed=True)"


def sparse24_bitmask_compress(
    tensor: Tensor,
    sparsity_structure: Union[SparsityStructure, str] = SparsityStructure.TWO_FOUR,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Compresses a dense tensor using bitmask compression

    :param tensor: dense 2D tensor to compress
    :param sparsity_structure: structure of sparsity in the tensor, defaults
        to unstructured, can also be set to `2:4`
    :return: tuple of compressed data representing tensor
    """
    assert len(tensor.shape) == 2, "Only 2D tensors are supported"
    assert (
        SparsityStructure(sparsity_structure) == SparsityStructure.TWO_FOUR
    ), "Only 2:4 sparsity is supported"

    bytemasks = get_24_bytemasks(tensor=tensor)

    if tensor.dtype == FP8_DTYPE:
        # acces raw bytes of the tensor
        tensor_view = tensor.view(torch.int8)
        values = tensor_view[bytemasks]
        values = values.view(FP8_DTYPE)
    else:
        values = tensor[bytemasks]

    num_rows, num_cols = tensor.shape
    compressed_values = values.reshape(num_rows, num_cols // 2)
    bitmasks_packed = pack_bitmasks(bytemasks)
    return compressed_values, bitmasks_packed


def sparse24_bitmask_decompress(
    values: Tensor, bitmasks: Tensor, original_shape: torch.Size
) -> Tensor:
    """
    Reconstructs a dense tensor from a compressed one

    :param values: 1d tensor of non-zero values
    :param bitmasks: 2d int8 tensor flagging locations of non-zero values in the
    tensors original shape
    :param original_shape: shape of the dense tensor
    :return: decompressed dense tensor
    """
    bytemasks_unpacked = unpack_bitmasks(bitmasks, original_shape)

    decompressed_tensor = torch.zeros(original_shape, dtype=values.dtype)
    decompressed_tensor = decompressed_tensor.to(values.device)
    values = values.flatten()
    if decompressed_tensor.dtype == FP8_DTYPE:
        decompressed_tensor[bytemasks_unpacked] = values
        decompressed_tensor = decompressed_tensor.cuda()
    else:
        decompressed_tensor[bytemasks_unpacked] = values
    return decompressed_tensor


def get_24_bytemasks(tensor):
    """
    Generate a 2:4 sparsity mask for the given tensor.

    This function creates a mask where exactly 2 out of every 4 elements are
    preserved based on their magnitudes. The preserved elements are the ones
    with the highest absolute values in each group of 4 elements.

    :param tensor: The input tensor for which the 2:4 sparsity mask is to be created.
                   The tensor can be of any shape but its total number of elements
                   must be a multiple of 4.
    :return: A boolean tensor of the same shape as the input tensor, where `True`
             indicates the preserved elements and `False` indicates the pruned elements.
    :raises ValueError: If the total number of elements in the tensor is not a
                        multiple of 4.
    """
    original_dtype = tensor.dtype
    if tensor.dtype == FP8_DTYPE:
        tensor = tensor.view(torch.int8)
    original_shape = tensor.shape
    num_elements = tensor.numel()

    if num_elements % 4 != 0:
        raise ValueError("Tensor size must be a multiple of 4 for TWO_FOUR sparsity")

    reshaped_tensor = tensor.view(-1, 4)
    abs_tensor = reshaped_tensor.abs()
    topk_indices = abs_tensor.topk(2, dim=1).indices
    mask = torch.zeros_like(reshaped_tensor, dtype=torch.bool)
    mask.scatter_(1, topk_indices, True)
    mask = mask.view(original_shape)
    tensor = tensor.view(original_dtype)

    return mask
