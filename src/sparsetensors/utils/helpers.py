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

from sparsetensors.compressors import ModelCompressor
from sparsetensors.config import CompressionConfig
from sparsetensors.utils.helpers import SPARSITY_CONFIG_NAME
from torch.nn import Module
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


def set_layer(target: str, layer: Module, module: Module) -> Module:
    target = fix_fsdp_module_name(target)  # noqa TODO
    with summon_full_params_context(module):  # noqa TODO
        # importing here to avoid circular import
        from sparseml.utils.fsdp.helpers import maybe_get_wrapped  # noqa TODO

        parent_target = ".".join(target.split(".")[:-1])
        if parent_target != "":
            parent_layer = get_layer(parent_target, module)[1]  # noqa TODO
        else:
            parent_layer = maybe_get_wrapped(module)
        old_layer = getattr(parent_layer, target.split(".")[-1])
        setattr(parent_layer, target.split(".")[-1], layer)

    return old_layer
