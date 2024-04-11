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

from pydantic import BaseModel
from sparsetensors.quantization.quant_scheme import QuantizationScheme


__all__ = ["QuantizationStatus", "QuantizationConfig"]


class QuantizationStatus(Enum):
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
    ignore: Optional[List[str]] = None
