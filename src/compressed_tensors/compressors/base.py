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

import operator
from typing import Dict, Generator, Optional, Tuple

from compressed_tensors.base import SPARSITY_CONFIG_NAME
from compressed_tensors.config import CompressionConfig
from compressed_tensors.registry import RegistryMixin
from compressed_tensors.utils import get_safetensors_folder
from torch import Tensor
from torch.nn import Module, Parameter
from tqdm import tqdm
from transformers import AutoConfig


__all__ = ["ModelCompressor"]


class ModelCompressor(RegistryMixin):
    """
    Base class representing a model compression algorithm.

    :param config: config specifying compression parameters
    """

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str
    ) -> Optional["ModelCompressor"]:
        """
        Given a path to a model config, extract a sparsity config if it exists and
        return the associated ModelCompressor

        :param pretrained_model_name_or_path: path to model config on disk or HF hub
        :return: matching compressor if config contains a sparsity config
        """
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        sparsity_config = getattr(config, SPARSITY_CONFIG_NAME, None)
        if sparsity_config is None:
            return None

        format = sparsity_config.get("format")
        sparsity_config = CompressionConfig.load_from_registry(
            format, **sparsity_config
        )
        compressor = cls.load_from_registry(format, config=sparsity_config)
        return compressor

    def __init__(self, config: Optional[CompressionConfig] = None):
        self.config = config

    def compress(self, model_state: Dict[str, Tensor]) -> Dict[str, Tensor]:
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
        :return: compressed state dict
        """
        raise NotImplementedError()

    def overwrite_weights(self, model_path: str, model: Module):
        """
        Overwrites the weights in model with weights decompressed from model_path

        :param model_path: path to compressed weights
        :param model: pytorch model to load decompressed weights into
        """
        model_path = get_safetensors_folder(model_path)
        dense_gen = self.decompress(model_path)
        for name, data in tqdm(dense_gen, desc="Decompressing model"):
            # loading the decompressed weights into the model
            model_device = operator.attrgetter(name)(model).device
            data_new = Parameter(data.to(model_device))
            data_old = operator.attrgetter(name)(model)
            data_old.data = data_new.data

        setattr(model, SPARSITY_CONFIG_NAME, self.config)
