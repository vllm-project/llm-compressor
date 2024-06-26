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

from typing import Dict, Generator, Tuple

from compressed_tensors.compressors import Compressor
from compressed_tensors.config import CompressionFormat
from torch import Tensor


@Compressor.register(name=CompressionFormat.dense.value)
class DenseCompressor(Compressor):
    """
    Identity compressor for dense models, returns the original state_dict
    """

    def compress(self, model_state: Dict[str, Tensor], **kwargs) -> Dict[str, Tensor]:
        return model_state

    def decompress(
        self, path_to_model_or_tensors: str, device: str = "cpu", **kwargs
    ) -> Generator[Tuple[str, Tensor], None, None]:
        return iter([])
