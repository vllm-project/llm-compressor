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
from typing import Dict, List, Optional, Union

from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization.quant_args import QuantizationArgs
from compressed_tensors.quantization.quant_scheme import (
    QuantizationScheme,
    preset_name_to_scheme,
)
from compressed_tensors.quantization.utils import (
    calculate_compression_ratio,
    is_module_quantized,
    iter_named_quantizable_modules,
    module_type,
    parse_out_kv_cache_args,
)
from pydantic import BaseModel, Field
from torch.nn import Module


__all__ = [
    "QuantizationStatus",
    "QuantizationConfig",
    "LIFECYCLE_ORDER",
    "DEFAULT_QUANTIZATION_METHOD",
    "DEFAULT_QUANTIZATION_FORMAT",
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

DEFAULT_QUANTIZATION_METHOD = "compressed-tensors"
DEFAULT_QUANTIZATION_FORMAT = "fakequant"


class QuantizationConfig(BaseModel):
    """
    Full configuration specifying how a model is quantized. Each quantized layer is
    mapped to a QuantizationScheme in config_groups.

    :param config_groups: dict of QuantizationSchemes specifying the quantization
    settings for each quantized layer. A group could also be a reference to
    a predefined scheme name, mapped to a list of its target layers/classes
    :param quant_method: a constant used to differentiate sparseML quantization from
    other quantization configs
    :param format: specifies how the quantized model is stored on disk
    :quantization_status: specifies the current status of all quantized layers. It is
        assumed all layers are in the same state.
    :param kv_cache_scheme: optional QuantizationArgs, that specify the
        quantization of the kv cache. If None, kv cache is not quantized.
        When applying kv cache quantization to transformer AutoModelForCausalLM,
        the kv_cache_scheme gets converted into a QuantizationScheme that:
            - targets the `q_proj` and `k_proj` modules of the model. The outputs
              of those modules are the keys and values that might be cached
            - quantizes the outputs of the aformentioned layers, so that
              keys and values are compressed before storing them in the cache
        There is an explicit assumption that the model contains modules with
        `k_proj` and `v_proj` in their names. If this is not the case
        and kv_cache_scheme != None, the quantization of kv cache will fail
    :global_compression_ratio: optional informational config to report the model
    compression ratio acheived by the quantization config
    :ignore: optional list of layers to ignore from config_groups. Layers in this list
    are not quantized even if they match up with a target in config_groups
    """

    config_groups: Dict[str, Union[QuantizationScheme, List[str]]]
    quant_method: str = DEFAULT_QUANTIZATION_METHOD
    kv_cache_scheme: Optional[QuantizationArgs] = None
    format: str = DEFAULT_QUANTIZATION_FORMAT
    quantization_status: QuantizationStatus = QuantizationStatus.INITIALIZED
    global_compression_ratio: Optional[float] = None
    ignore: Optional[List[str]] = Field(default_factory=list)

    def model_post_init(self, __context):
        """
        updates any quantization schemes defined as presets to be fully loaded
        schemes
        """
        for group_name, targets_or_scheme in self.config_groups.items():
            if isinstance(targets_or_scheme, QuantizationScheme):
                continue  # scheme already defined
            self.config_groups[group_name] = preset_name_to_scheme(
                name=group_name,
                targets=targets_or_scheme,
            )

    def to_dict(self):
        # for compatibility with HFQuantizer
        return self.dict()

    @staticmethod
    def from_pretrained(
        model: Module, format: Optional[str] = None
    ) -> Optional["QuantizationConfig"]:
        """
        Converts a model into its associated QuantizationConfig based on the
        QuantizationScheme attached to each quantized module

        :param model: model to calculate quantization scheme of
        :return: filled out QuantizationScheme for the input model
        """
        quant_scheme_to_layers = []
        quantization_status = None
        ignore = {}
        quantization_type_names = set()
        for name, submodule in iter_named_quantizable_modules(
            model, include_children=True, include_attn=True
        ):
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

        # kv-cache only, no weight/activation quantization
        if (
            len(quantization_type_names) == 1
            and "attention" in list(quantization_type_names)[0].lower()
        ):
            quantization_type_names.add("Linear")

        # clean up ignore list, we can leave out layers types if none of the
        # instances are quantized
        consolidated_ignore = []
        for layer_type, ignore_names in ignore.items():
            if layer_type in quantization_type_names:
                # specific layers of a quantized type are ignored
                consolidated_ignore += ignore_names
            # else we leave it off the ignore list, doesn't fall under any of the
            # existing quantization schemes so it won't be quantized

        kv_cache_args, quant_scheme_to_layers = parse_out_kv_cache_args(
            quant_scheme_to_layers
        )
        kv_cache_scheme = (
            kv_cache_args.model_dump() if kv_cache_args is not None else kv_cache_args
        )

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
            kv_cache_scheme=kv_cache_scheme,
            global_compression_ratio=compression_ratio,
            format=format,
            ignore=consolidated_ignore,
        )

    def requires_calibration_data(self):
        if self.kv_cache_scheme is not None:
            return True

        for _, scheme in self.config_groups.items():
            if scheme.input_activations is not None:
                if not scheme.input_activations.dynamic:
                    return True
            if scheme.output_activations is not None:
                if not scheme.output_activations.dynamic:
                    return True

        return False
