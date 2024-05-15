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

from typing import Dict, Generator, Tuple, Union

from compressed_tensors.config import SparsityCompressionConfig
from compressed_tensors.quantization import QuantizationConfig
from compressed_tensors.registry import RegistryMixin
from torch import Tensor


__all__ = ["Compressor"]


class Compressor(RegistryMixin):
    """
    Base class representing a model compression algorithm

    :param config: config specifying compression parameters
    """

    def __init__(
        self, config: Union[SparsityCompressionConfig, QuantizationConfig, None] = None
    ):
        self.config = config

    def compress(self, model_state: Dict[str, Tensor], **kwargs) -> Dict[str, Tensor]:
        """
        Compresses a dense state dict

        :param model_state: state dict of uncompressed model
        :return: compressed state dict
        """
        raise NotImplementedError()

    def decompress(
        self, path_to_model_or_tensors: str, device: str = "cpu"
    ) -> Generator[Tuple[str, Tensor], None, None]:
        """
        Reads a compressed state dict located at path_to_model_or_tensors
        and returns a generator for sequentially decompressing back to a
        dense state dict

        :param model_path: path to compressed safetensors model (directory with
            one or more safetensors files) or compressed tensors file
        :param device: optional device to load intermediate weights into
        :return: compressed state dict
        """
        raise NotImplementedError()
