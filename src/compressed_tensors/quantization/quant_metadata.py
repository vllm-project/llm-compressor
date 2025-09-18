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

from compressed_tensors.utils import delete_offload_parameter
from torch.nn import Module


__all__ = ["QuantizationMetadata", "KVCacheScaleType"]


class KVCacheScaleType(Enum):
    KEY = "k_scale"
    VALUE = "v_scale"


class QuantizationMetadata:
    """
    Container class for metadata related to quantization
    """

    @staticmethod
    def all_qparam_names():
        """
        All quantization parameter names that might be registered
        onto a module during lifecycle (excluding serialized parameters)
        """
        return [KVCacheScaleType.KEY.value, KVCacheScaleType.VALUE.value] + [
            f"{base_name}_{suffix}"
            for base_name in ("input", "weight", "output")
            for suffix in (
                "global_scale",
                "scale",
                "zero_point",
                "g_idx",
            )
        ]

    @classmethod
    def clear_all_qparams(cls, module: Module):
        """
        Remove all parameters related to quantization that might have
        been registered onto a module previously in lifecycle (excluding
        serialized parameters)

        :param module: Module to clear
        """
        for key in cls.all_qparam_names():
            if hasattr(module, key):
                delete_offload_parameter(module, key)
