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

from typing import List

from compressed_tensors.transform import TransformArgs
from pydantic import BaseModel, Field


__all__ = ["TransformScheme"]


class TransformScheme(BaseModel):
    """
    Scheme used to parameterize a particular transform type and specify how and where it
    should be applied to the model

    :param type: string indicating the particular transform type that should be created
        and applied. This should be one of the registered transform types
        (see `Transforms.registered_names()`)
    :param apply: list of TransformationArgs containing the information about the
        modules that should be targeted by the specified transform
    :param randomize: True if uniquely randomized transform weights should be used,
        otherwise use identical transform weights where applicable
    :param requires_grad: True if weights include gradients for training
    """

    type: str
    apply: List[TransformArgs] = Field(default_factory=list)
    randomize: bool = Field(default=False)
    requires_grad: bool = Field(default=False)
