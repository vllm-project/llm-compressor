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

from dataclasses import dataclass
from typing import Any

__all__ = ["ModelParameterizedLayer"]


@dataclass
class ModelParameterizedLayer:
    """
    A dataclass for holding a parameter and its layer

    :param layer_name: the name of the layer
    :param layer: the layer object
    :param param_name: the name of the parameter
    :param param: the parameter object
    """

    layer_name: str
    layer: Any
    param_name: str
    param: Any
