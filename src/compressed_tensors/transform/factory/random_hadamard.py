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

from compressed_tensors.transform import HadamardFactory, TransformFactory
from compressed_tensors.transform.utils.hadamard import random_hadamard_matrix
from torch import device, dtype
from torch.nn import Parameter


@TransformFactory.register("random-hadamard")
class RandomHadamardFactory(HadamardFactory):
    """
    Factory used to apply random hadamard transforms to a model

    :param name: name associated with transform scheme
    :param scheme: transform scheme which defines how transforms should be created
    :param seed: random seed used to transform weight randomization
    """

    def _create_weight(self, size: int, dtype: dtype, device: device) -> Parameter:
        data = random_hadamard_matrix(size, dtype, device, self.generator)
        data = data.to(dtype=dtype, device=device)
        return Parameter(data, requires_grad=self.scheme.requires_grad)
