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

from typing import TYPE_CHECKING, Dict, Generator, Tuple

import torch
from compressed_tensors.compressors.base import BaseCompressor
from compressed_tensors.config import CompressionFormat
from torch import Tensor


if TYPE_CHECKING:
    from compressed_tensors.quantization import QuantizationScheme


@BaseCompressor.register(name=CompressionFormat.dense.value)
class DenseCompressor(BaseCompressor):
    """
    Identity compressor for dense models, returns the original state_dict
    """

    @property
    def compression_param_names(self) -> Tuple[str]:
        """
        Returns a tuple of compression parameter names introduced by
        the compressor during compression
        """
        return ()

    def compress(self, model_state: Dict[str, Tensor], **kwargs) -> Dict[str, Tensor]:
        return model_state

    def decompress(
        self, path_to_model_or_tensors: str, device: str = "cpu", **kwargs
    ) -> Generator[Tuple[str, Tensor], None, None]:
        return iter([])

    def decompress_from_state_dict(
        self,
        state_dict: Dict[str, Tensor],
    ) -> Generator[Tuple[str, Dict[str, Tensor]], None, None]:
        for key, value in state_dict.items():
            yield key, value

    def decompress_module_from_state_dict(
        self,
        prefix: str,
        state_dict: Dict[str, torch.Tensor],
        scheme: "QuantizationScheme",
    ) -> Dict[str, torch.Tensor]:
        """
        This function is implemented as a workaround because of how
        `ModelCompressor.quantization_compressor` can be set to either
        an instance of `BaseQuantizationCompressor` or `DenseCompressor`.
        """
        return state_dict.copy()
