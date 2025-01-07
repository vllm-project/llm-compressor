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

import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from compressed_tensors.compressors import ModelCompressor
from compressed_tensors.config.base import SparsityCompressionConfig
from compressed_tensors.quantization.quant_config import QuantizationConfig
from safetensors.torch import save_file
from tests.testing_utils import induce_sparsity, requires_hf_quantizer


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

    s_config_dict = s_config.model_dump() if s_config is not None else None
    q_config_dict = q_config.model_dump() if q_config is not None else None

    assert compressor.sparsity_config == s_config
    assert compressor.quantization_config == q_config

    assert ModelCompressor.parse_sparsity_config(compression_config) == s_config_dict
    assert (
        ModelCompressor.parse_quantization_config(compression_config) == q_config_dict
    )


@pytest.fixture
def fake_model_class():
    import torch.nn as nn

    class CustomLinearModel(nn.Module):
        def __init__(self, weights, weight_scale=None, weight_zero_point=None):
            super(CustomLinearModel, self).__init__()
            out_features, in_features = weights.shape

            # Define a linear layer without bias
            self.linear = nn.Linear(in_features, out_features, bias=False)

            # Set the weights of the linear layer
            self.linear.weight = nn.Parameter(weights, requires_grad=False)

            # Attach weight_scale and weight_zero_point as parameters
            if weight_scale is not None:
                self.linear.weight_scale = nn.Parameter(
                    torch.tensor(weight_scale), requires_grad=False
                )
            if weight_zero_point is not None:
                self.linear.weight_zero_point = nn.Parameter(
                    torch.tensor(weight_zero_point), requires_grad=False
                )

        def forward(self, x):
            return self.linear(x)

    return CustomLinearModel


def get_bitmask_sparsity_config():
    from compressed_tensors import BitmaskConfig

    return BitmaskConfig(
        format="sparse-bitmask",
        global_sparsity=0.7,
        targets=["Linear"],
        sparsity_structure="unstructured",
    )


def create_quantization_config(bits=8, type="int", strategy="tensor"):

    config_dict = {
        "format": "int-quantized",
        "global_compression_ratio": 1.0,
        "quant_method": "compressed-tensors",
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": bits,
                    "strategy": strategy,
                    "symmetric": True,
                    "type": type,
                },
            }
        },
    }

    return QuantizationConfig.model_validate(config_dict)


@pytest.mark.parametrize("sparsity_config", [get_bitmask_sparsity_config()])
@pytest.mark.parametrize(
    "quantization_config",
    [
        create_quantization_config(bits=8, type="int", strategy="channel"),
        create_quantization_config(bits=8, type="float", strategy="channel"),
    ],
)
def test_composability(
    tmp_path, fake_model_class, sparsity_config, quantization_config
):
    from compressed_tensors.quantization.lifecycle.forward import quantize

    weights = torch.rand(10, 5)
    sparse_weights = induce_sparsity(weights, sparsity_config.global_sparsity)

    quantization_args = quantization_config.config_groups["group_0"].weights

    if quantization_args.strategy == "channel":
        scale = torch.ones((weights.shape[0], 1))
    elif quantization_args.strategy == "tensor":
        scale = torch.tensor([1.0])

    zero_point = torch.zeros_like(scale)

    quantized_weights = quantize(
        sparse_weights,
        scale=scale,
        zero_point=zero_point,
        args=quantization_args,
    )

    fake_oneshot_model = fake_model_class(quantized_weights, scale, zero_point)
    fake_oneshot_model.linear.quantization_scheme = quantization_config.config_groups[
        "group_0"
    ]
    model_compressor = ModelCompressor(
        sparsity_config=sparsity_config, quantization_config=quantization_config
    )
    # does both sparse and quantization compression
    compressed_state_dict = model_compressor.compress(fake_oneshot_model)

    save_dir = tmp_path / "model"
    save_dir = _create_dummy_checkpoint(
        compressed_state_dict, save_dir, model_compressor
    )

    decompressed_model = fake_model_class(torch.zeros_like(weights))
    model_compressor.decompress(model=decompressed_model, model_path=save_dir)

    # check that the decompressed model is the same as the original model
    _check_state_dicts(fake_oneshot_model.state_dict(), decompressed_model.state_dict())


def _create_dummy_checkpoint(state_dict, save_dir, model_compressor):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_file(state_dict, save_dir / "model.safetensors")

    config_file_path = save_dir / "config.json"
    with open(config_file_path, "w") as config_file:
        json.dump({}, config_file, indent=2, sort_keys=True)

    model_compressor.update_config(save_dir)
    return save_dir


def _check_state_dicts(state_dict1, state_dict2):
    for key in state_dict1.keys():
        assert key in state_dict2, f"Missing tensor: {key}"
        if key.endswith("weight"):
            original_tensor = state_dict1[key]
            decompressed_tensor = state_dict2[key].to(original_tensor.dtype)
            diff = torch.abs(original_tensor - decompressed_tensor)
            assert not torch.any(diff > 0.01), f"Max diff: {torch.max(diff)}"
