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
from typing import List

from pydantic import BaseModel, Field, field_validator


__all__ = ["TransformArgs", "TransformLocation"]


class TransformLocation(str, Enum):
    """
    Enum representing which parameters/activations a transform weight should be applied
    to on a given module.

    | -------------------------------------------------------------------------------------------------------- |  # noqa: E501
    | Name            | Runtime     | Values        | Locations Where Inverse Could Be Applied                 |  # noqa: E501
    | --------------- | ----------- | ------------- | -------------------------------------------------------- |  # noqa: E501
    | `INPUT`         | online      | activations   | `prev.WEIGHT_OUTPUT`, `prev.OUTPUT`, `this.WEIGHT_INPUT` |  # noqa: E501
    | `WEIGHT_INPUT`  | offline     | weight        | `prev.WEIGHT_OUTPUT`, `prev.OUTPUT`, `this.INPUT`        |  # noqa: E501
    | `WEIGHT_OUTPUT` | offline     | weight        | `this.OUTPUT`, `next.INPUT`, `next.WEIGHT_INPUT`         |  # noqa: E501
    | `OUTPUT`        | online      | activations   | `this.WEIGHT_OUTPUT`, `next.INPUT`, `next.WEIGHT_INPUT`  |  # noqa: E501
    | `K_CACHE`       | online      | key_values    | `q_proj.Q_ATTN`                                          |  # noqa: E501
    | `Q_ATTN`        | online      | query_values  | `k_proj.K_CACHE`                                         |  # noqa: E501
    | -------------------------------------------------------------------------------------------------------- |  # noqa: E501
    """

    INPUT = "input"
    WEIGHT_INPUT = "weight_input"
    WEIGHT_OUTPUT = "weight_output"
    OUTPUT = "output"
    K_CACHE = "k_cache"
    Q_ATTN = "q_attn"


class TransformArgs(BaseModel):
    """
    Arguments which define *how* and where a transform should be applied to a model

    :param targets: list of modules to apply transforms to
    :param location: where to apply transform on module, one of (`input`, `weight`,
        `output`, `k_cache`, `q_attn`)
    :param inverse: whether or not to apply the inverse of a transform
    :param ignore: any modules which should be ignored from the targets list
    """

    targets: List[str]
    location: TransformLocation
    inverse: bool = Field(default=False)
    ignore: List[str] = Field(default_factory=list)

    @field_validator("targets", "ignore", mode="before")
    @classmethod
    def wrap_singleton(cls, value):
        if isinstance(value, str):
            return [value]
        return value
