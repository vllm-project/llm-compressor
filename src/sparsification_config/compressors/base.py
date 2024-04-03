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
from typing import Dict, Generator, Tuple

from sparsezoo.utils.registry import RegistryMixin
from torch import Tensor
from torch.nn import Module, Parameter
from tqdm import tqdm

__all__ = ["ModelCompressor"]


class ModelCompressor(RegistryMixin):
    """
    Base class representing a model compression algorithm.

    :param config: config specifying compression parameters
    """

    def __init__(self, config: "CompressionConfig"):  # noqa
        self.config = config

    def compress(self, model_state: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Compresses a dense state dict

        :param model_state: state dict of uncompressed model
        :return: compressed state dict
        """
        raise NotImplementedError()

    def decompress(self, model_path: str) -> Generator[Tuple[str, Tensor], None, None]:
        """
        Reads a compressed state dict located at model_path and returns a
        generator for sequentially decompressing back to a dense state dict

        :param model_path: path to compressed safetensors model
        :return: compressed state dict
        """
        raise NotImplementedError()

    @staticmethod
    def replace_layer(param_name: str, data: Tensor, model: Module):
        """
        Overwrites a parameterized layer with a new tensor, maintaining the device of
        the original parameter

        :param param_name: name of parameterized layer to replace
        :param data: tensor to insert into model
        :param model: pytorch model to insert data into
        """
        model_device = operator.attrgetter(param_name)(model).device
        set_layer(param_name, Parameter(data.to(model_device)), model)

    def overwrite_weights(self, pretrained_model_name_or_path: str, model: Module):
        """
        Overwrites the weights in model with weights decompressed from
        pretrained_model_name_or_path

        :param pretrained_model_name_or_path: path to compressed weights
        :param model: pytorch model to load decompressed weights into
        """
        dense_gen = self.decompress(pretrained_model_name_or_path)
        for name, data in tqdm(dense_gen, desc="Decompressing model"):
            ModelCompressor.replace_layer(name, data, model)
        setattr(model, SPARSITY_CONFIG_NAME, self.config)
