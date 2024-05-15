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

from enum import Enum
from typing import Optional

from compressed_tensors.registry import RegistryMixin
from pydantic import BaseModel


__all__ = ["SparsityCompressionConfig", "CompressionFormat"]


class CompressionFormat(Enum):
    dense = "dense"
    sparse_bitmask = "sparse-bitmask"
    int_quantized = "int-quantized"
    pack_quantized = "pack-quantized"


class SparsityCompressionConfig(RegistryMixin, BaseModel):
    """
    Base data class for storing sparsity compression parameters

    :param format: name of compression format
    :param global_sparsity: average sparsity of the entire model
    :param sparsity_structure: structure of the sparsity, such as
    "unstructured", "2:4", "8:16" etc
    """

    format: str
    global_sparsity: Optional[float] = 0.0
    sparsity_structure: Optional[str] = "unstructured"
