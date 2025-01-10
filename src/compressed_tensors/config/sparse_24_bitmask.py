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

from typing import Optional

from compressed_tensors.config import (
    CompressionFormat,
    SparsityCompressionConfig,
    SparsityStructure,
)


__all__ = ["Sparse24BitMaskConfig"]


@SparsityCompressionConfig.register(name=CompressionFormat.sparse_24_bitmask.value)
class Sparse24BitMaskConfig(SparsityCompressionConfig):
    """
    Configuration for storing a 24 sparse model using
    bytemask compression

    :param global_sparsity: average sparsity of the entire model
    :param sparsity_structure: structure of the sparsity, should always be
        "2:4" for this compression format
    """

    format: str = CompressionFormat.sparse_24_bitmask.value
    global_sparsity: Optional[float] = 0.0
    sparsity_structure: Optional[str] = SparsityStructure.TWO_FOUR.value
