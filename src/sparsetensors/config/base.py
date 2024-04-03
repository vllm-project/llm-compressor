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

from pydantic import BaseModel
from sparsezoo.utils.registry import ModuleSparsificationInfo, RegistryMixin
from torch.nn import Module


__all__ = ["CompressionConfig"]


class CompressionConfig(RegistryMixin, BaseModel):
    """
    Base data class for storing compression parameters

    :param format: name of compression format
    :param global_sparsity: average sparsity of the entire model
    :param sparsity_structure: structure of the sparsity, such as
    "unstructured", "2:4", "8:16" etc
    """

    format: str
    global_sparsity: Optional[float] = 0.0
    sparsity_structure: Optional[str] = "unstructured"

    @staticmethod
    def infer_global_sparsity(model: Module) -> float:
        """
        Calculates the global percentage of sparse zero weights in the model

        :param model: pytorch model to infer sparsity of
        :return: global sparsity of model
        """
        info = ModuleSparsificationInfo(model)
        global_sparsity = info.params_sparse_percent
        return global_sparsity

    # TODO: Move infer_sparsity_structure to sparseml

    @staticmethod
    def infer_config_from_model(
        model: Module, compress: bool = False
    ) -> Optional["CompressionConfig"]:
        """
        Determines compression type and informational parameters for a given model

        :param model: pytorch model to calculate sparsity config for
        :param compress: whether or not to compress the model on disk
        :return: compression config inferred from the model
        """

        global_sparsity = CompressionConfig.infer_global_sparsity(model)

        if global_sparsity < 0.05:
            return None

        sparsity_structure = CompressionConfig.infer_sparsity_structure()
        if compress:
            format = "sparse_bitmask"
        else:
            format = "dense_sparsity"

        return CompressionConfig.load_from_registry(
            format,
            global_sparsity=global_sparsity,
            sparsity_structure=sparsity_structure,
        )

    def fill_config_details(self, model: Module):
        """
        Fills in informational sparsity parameters from a given model

        :param model: pytorch model to infer config parameters from
        """
        self.global_sparsity = CompressionConfig.infer_global_sparsity(model)
        self.sparsity_structure = CompressionConfig.infer_sparsity_structure()
