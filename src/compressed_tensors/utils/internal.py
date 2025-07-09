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

import torch


__all__ = ["InternalModule"]


class InternalModule(torch.nn.Module):
    """
    Abstract base class for modules which are not a part of the the model definition.
    `torch.nn.Module`s which inherit from this class will not be targeted by configs

    This is typically used to skip apply configs to `Observers` and `Transforms`
    """

    pass
