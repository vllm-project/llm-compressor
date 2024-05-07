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
from typing import Dict, List, Optional

from compressed_tensors.base import QUANTIZATION_CONFIG_NAME
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization.quant_scheme import QuantizationScheme
from compressed_tensors.quantization.utils import (
    calculate_compression_ratio,
    is_module_quantized,
    iter_named_leaf_modules,
    module_type,
)
from pydantic import BaseModel, Field
from torch.nn import Module
from transformers import AutoConfig


__all__ = [
    "QuantizationStatus",
    "QuantizationConfig",
    "LIFECYCLE_ORDER",
]


class QuantizationStatus(str, Enum):
    """
    Enum storing the different states a quantized layer can be in

    Initialized: scale, zero points and observers have been attached to the layer but
    are set to dummy values (not yet calibrated)
    Calibration: scale and zero points have been calibrated through OBCQ or similar
    algorithm, observers are still attached
    Frozen: scale and zero points are finalized, observers have been deleted, weights
    are still in their original precision
    Compressed: weights have been converted to their target type or compressed to
    their closed approximation
    """

    INITIALIZED = "initialized"
    CALIBRATION = "calibration"
    FROZEN = "frozen"
    COMPRESSED = "compressed"

    @classmethod
    def lifecycle_order(cls) -> List["QuantizationStatus"]:
        """
        :return: list of correct quantization lifecycle order
        """
        return

    def __ge__(self, other):
        if other is None:
            return True
        if not isinstance(other, self.__class__):
            raise NotImplementedError
        return LIFECYCLE_ORDER.index(self) >= LIFECYCLE_ORDER.index(other)

    def __gt__(self, other):
        if other is None:
            return True
        if not isinstance(other, self.__class__):
            raise NotImplementedError
        return LIFECYCLE_ORDER.index(self) > LIFECYCLE_ORDER.index(other)

    def __lt__(self, other):
        if other is None:
            return False
        if not isinstance(other, self.__class__):
            raise NotImplementedError
        return LIFECYCLE_ORDER.index(self) < LIFECYCLE_ORDER.index(other)

    def __le__(self, other):
        if other is None:
            return False
        if not isinstance(other, self.__class__):
            raise NotImplementedError
        return LIFECYCLE_ORDER.index(self) <= LIFECYCLE_ORDER.index(other)


LIFECYCLE_ORDER = [
    QuantizationStatus.INITIALIZED,
    QuantizationStatus.CALIBRATION,
    QuantizationStatus.FROZEN,
    QuantizationStatus.COMPRESSED,
]


class QuantizationConfig(BaseModel):
    """
    Full configuration specifying how a model is quantized. Each quantized layer is
    mapped to a QuantizationScheme in config_groups.

    :param config_groups: dict of QuantizationSchemes specifying the quantization
    settings for each quantized layer
    :param quant_method: a constant used to differentiate sparseML quantization from
    other quantization configs
    :param format: specifies how the quantized model is stored on disk
    :quantization_status: specifies the current status of all quantized layers. It is
    assumed all layers are in the same state.
    :global_compression_ratio: optional informational config to report the model
    compression ratio acheived by the quantization config
    :ignore: optional list of layers to ignore from config_groups. Layers in this list
    are not quantized even if they match up with a target in config_groups
    """

    config_groups: Dict[str, QuantizationScheme]
    quant_method: str = "sparseml"
    format: str = "fakequant"
    quantization_status: QuantizationStatus = QuantizationStatus.INITIALIZED
    global_compression_ratio: Optional[float] = None
    ignore: Optional[List[str]] = Field(default_factory=list)

    @staticmethod
    def from_model_config(model_name_or_path) -> "QuantizationConfig":
        """
        Given a path to a model config, extract a quantization config if it exists

        :param pretrained_model_name_or_path: path to model config on disk or HF hub
        :return: instantiated QuantizationConfig if config contains a quant config
        """
        config = AutoConfig.from_pretrained(model_name_or_path)
        quantization_config = getattr(config, QUANTIZATION_CONFIG_NAME, None)
        if quantization_config is None:
            return None

        return QuantizationConfig.parse_obj(quantization_config)

    @staticmethod
    def from_pretrained(
        model: Module, format: Optional[str] = None
    ) -> Optional["QuantizationConfig"]:
        """
        Converts a model into its associated QuantizationConfig based on the
        QuantizationScheme attached to each quanitzed module

        :param model: model to calculate quantization scheme of
        :return: filled out QuantizationScheme for the input model
        """
        quant_scheme_to_layers = []
        quantization_status = None
        ignore = {}
        quantization_type_names = set()
        for name, submodule in iter_named_leaf_modules(model):
            layer_type = module_type(submodule)
            if not is_module_quantized(submodule):
                if layer_type not in ignore:
                    ignore[layer_type] = []
                ignore[layer_type].append(name)
            else:
                quantization_status = submodule.quantization_status
                scheme = submodule.quantization_scheme
                quantization_type_names.add(layer_type)

                match_found = False
                for existing_scheme in quant_scheme_to_layers:
                    if scheme == existing_scheme:
                        match_found = True
                        break
                if not match_found:
                    quant_scheme_to_layers.append(scheme)

        if len(quant_scheme_to_layers) == 0:  # No quantized layers
            return None

        # clean up ignore list, we can leave out layers types if none of the
        # instances are quantized
        consolidated_ignore = []
        for layer_type, ignore_names in ignore.items():
            if layer_type in quantization_type_names:
                # specific layers of a quantized type are ignored
                consolidated_ignore += ignore_names
            # else we leave it off the ignore list, doesn't fall under any of the
            # existing quantization schemes so it won't be quantized

        config_groups = {}
        for idx, scheme in enumerate(quant_scheme_to_layers):
            group_name = "group_" + str(idx)
            config_groups[group_name] = scheme

        # TODO: this is incorrect in compressed mode, since we are overwriting the
        # original weight we lose the uncompressed bit_depth indo
        compression_ratio = calculate_compression_ratio(model)

        if format is None:
            if quantization_status == QuantizationStatus.COMPRESSED:
                format = CompressionFormat.int_quantized.value
            else:
                format = CompressionFormat.dense.value

        return QuantizationConfig(
            config_groups=config_groups,
            quantization_status=quantization_status,
            global_compression_ratio=compression_ratio,
            format=format,
            ignore=consolidated_ignore,
        )
