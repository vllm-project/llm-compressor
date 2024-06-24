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

from transformers import AutoConfig


__all__ = ["infer_compressor_from_model_config", "fix_fsdp_module_name"]

FSDP_WRAPPER_NAME = "_fsdp_wrapped_module"


def infer_compressor_from_model_config(
    pretrained_model_name_or_path: str,
) -> Optional["ModelCompressor"]:  # noqa: F821
    """
    Given a path to a model config, extract a sparsity config if it exists and return
    the associated ModelCompressor

    :param pretrained_model_name_or_path: path to model config on disk or HF hub
    :return: matching compressor if config contains a sparsity config
    """
    from compressed_tensors.compressors import ModelCompressor
    from compressed_tensors.config import CompressionConfig

    config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
    sparsity_config = ModelCompressor.parse_sparsity_config(config)
    if sparsity_config is None:
        return None

    format = sparsity_config.get("format")
    sparsity_config = CompressionConfig.load_from_registry(format, **sparsity_config)
    compressor = ModelCompressor.load_from_registry(format, config=sparsity_config)
    return compressor


# TODO: There is already the same function in
# SparseML, should be moved to a shared location
# in the future
def fix_fsdp_module_name(name: str) -> str:
    """
    Remove FSDP wrapper prefixes from a module name
    Accounts for scenario where FSDP_WRAPPER_NAME is
    at the end of the name, as well as in the middle.
    :param name: name to strip
    :return: stripped name
    """
    return name.replace(FSDP_WRAPPER_NAME + ".", "").replace(
        "." + FSDP_WRAPPER_NAME, ""
    )
