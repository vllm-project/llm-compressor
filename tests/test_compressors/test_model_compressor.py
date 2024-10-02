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

from copy import deepcopy

import pytest
from compressed_tensors.compressors import ModelCompressor
from compressed_tensors.config.base import SparsityCompressionConfig
from compressed_tensors.quantization.quant_config import QuantizationConfig
from tests.testing_utils import requires_hf_quantizer


def sparsity_config():
    return {
        "format": "sparse-bitmask",  # dense format is ignored by ModelCompressor
        "global_sparsity": 19.098103233975568,
        "registry_requires_subclass": False,
        "sparsity_structure": "unstructured",
    }


def quantization_config():
    return {
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 4,
                    "strategy": "channel",
                    "symmetric": True,
                    "type": "int",
                },
            }
        },
        "format": "pack-quantized",
        "global_compression_ratio": 1.891791164021256,
        "ignore": ["lm_head"],
        "quant_method": "compressed-tensors",
        "quantization_status": "frozen",
    }


def _get_combined_config(s_config, q_config):
    combined = {}

    if q_config is not None:
        combined = deepcopy(q_config)

    if s_config is not None:
        combined["sparsity_config"] = s_config

    return combined


@pytest.mark.parametrize("s_config", [sparsity_config(), None])
@pytest.mark.parametrize("q_config", [quantization_config(), None])
def test_config_format(s_config, q_config):
    combined_config = _get_combined_config(s_config, q_config)
    assert ModelCompressor.parse_sparsity_config(combined_config) == s_config
    assert ModelCompressor.parse_quantization_config(combined_config) == q_config


@requires_hf_quantizer()
@pytest.mark.parametrize(
    "s_config,q_config",
    [
        (sparsity_config(), quantization_config()),
        (sparsity_config(), None),
        (None, quantization_config()),
        (None, None),
    ],
)
def test_hf_compressor_tensors_config(s_config, q_config, tmp_path):
    from transformers.utils.quantization_config import CompressedTensorsConfig

    combined_config = _get_combined_config(s_config, q_config)
    compression_config = CompressedTensorsConfig(**combined_config)
    compressor = ModelCompressor.from_compression_config(compression_config)

    if s_config is q_config is None:
        assert compressor is None
        return

    s_config = (
        SparsityCompressionConfig.load_from_registry(s_config.get("format"), **s_config)
        if s_config is not None
        else None
    )
    q_config = QuantizationConfig(**q_config) if q_config is not None else None

    s_config_dict = s_config.dict() if s_config is not None else None
    q_config_dict = q_config.dict() if q_config is not None else None

    assert compressor.sparsity_config == s_config
    assert compressor.quantization_config == q_config

    assert ModelCompressor.parse_sparsity_config(compression_config) == s_config_dict
    assert (
        ModelCompressor.parse_quantization_config(compression_config) == q_config_dict
    )
