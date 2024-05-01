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


from typing import Optional

from compressed_tensors.base import SPARSITY_CONFIG_NAME
from compressed_tensors.compressors import ModelCompressor
from compressed_tensors.config import CompressionConfig
from transformers import AutoConfig


__all__ = ["infer_compressor_from_model_config"]


def infer_compressor_from_model_config(
    pretrained_model_name_or_path: str,
) -> Optional[ModelCompressor]:
    """
    Given a path to a model config, extract a sparsity config if it exists and return
    the associated ModelCompressor

    :param pretrained_model_name_or_path: path to model config on disk or HF hub
    :return: matching compressor if config contains a sparsity config
    """
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
    sparsity_config = getattr(config, SPARSITY_CONFIG_NAME, None)
    if sparsity_config is None:
        return None

    format = sparsity_config.get("format")
    sparsity_config = CompressionConfig.load_from_registry(format, **sparsity_config)
    compressor = ModelCompressor.load_from_registry(format, config=sparsity_config)
    return compressor
