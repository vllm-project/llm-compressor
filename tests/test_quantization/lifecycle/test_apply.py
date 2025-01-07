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

import re
from typing import Optional
from unittest.mock import MagicMock

import pytest
import torch
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import (
    DEFAULT_QUANTIZATION_METHOD,
    QuantizationConfig,
    QuantizationStatus,
)
from compressed_tensors.quantization.lifecycle import (
    apply_quantization_config,
    apply_quantization_status,
    expand_sparse_target_names,
    is_sparse_target,
)
from compressed_tensors.quantization.utils import iter_named_leaf_modules
from tests.testing_utils import requires_accelerate
from transformers import AutoModelForCausalLM


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.named_modules.return_value = [
        ("layer1", MagicMock()),
        ("layer2", MagicMock()),
        ("layer3", MagicMock()),
    ]
    return model


@pytest.fixture
def mock_module():
    return MagicMock()


@pytest.fixture
def llama_stories_model():
    return AutoModelForCausalLM.from_pretrained(
        "Xenova/llama2.c-stories15M",
        torch_dtype="auto",
    )


def test_target_prioritization(mock_frozen):
    # tests that the config_groups are applied in the correct order
    # of priority, where exact layer name > regex > module name
    config = {
        "quant_method": "compressed-tensors",
        "format": "fakequant",
        "config_groups": {
            "group_1": {
                "weights": {
                    "num_bits": 8,
                },
                "targets": ["Linear"],
            },
            "group_2": {
                "weights": {
                    "num_bits": 4,
                },
                "targets": ["re:.*down_proj"],
            },
            "group_3": {
                "weights": {
                    "num_bits": 2,
                },
                "targets": ["model.layers.0.mlp.down_proj"],
            },
        },
    }

    model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceM4/tiny-random-LlamaForCausalLM", torch_dtype="auto"
    )
    model.eval()

    config = QuantizationConfig(**config)
    config.quantization_status = QuantizationStatus.CALIBRATION
    apply_quantization_config(model, config)
    mock_frozen(model)

    for name, module in iter_named_leaf_modules(model):
        if name == "model.layers.0.mlp.down_proj":
            assert module.quantization_scheme.weights.num_bits == 2
        elif re.match(".*down_proj", name):
            assert module.quantization_scheme.weights.num_bits == 4
        elif isinstance(module, torch.nn.Linear):
            assert module.quantization_scheme.weights.num_bits == 8


def test_apply_quantization_config_tinyllama():
    quant_config = get_sample_tinyllama_quant_config(status="calibration")
    model = get_tinyllama_model()

    # check that model is not already quantized
    for module in model.modules():
        _test_layer_quantization_status(module, inputs=False, weights=False)

    # apply quant config to model
    apply_quantization_config(model, quant_config)

    # check for correct application of quant config
    num_linears = 0
    num_embeddings = 0
    num_rotary_embeddings = 0
    for name, module in model.named_modules():
        if name in quant_config.ignore:
            continue
        module_type = module.__class__.__name__
        if module_type == "Linear":
            num_linears += 1
            _test_layer_quantization_status(module, inputs=True, weights=True)
        elif module_type == "Embedding":
            num_embeddings += 1
            _test_layer_quantization_status(module, inputs=False, weights=True)
        elif module_type == "LlamaRotaryEmbedding":
            num_rotary_embeddings += 1
            _test_layer_quantization_status(module, inputs=False, weights=False)

    # sanity check correct number of layers targeted
    assert num_linears == 154  # 155 Linear layers - 1 that gets ignored
    assert num_embeddings == 1
    assert num_rotary_embeddings == 23  # model updated, now has model.rotary_embedding

    # test quantization compression
    # sample forward pass to fill scales, zps
    model(torch.zeros((1, 1), dtype=int), torch.zeros((1, 1), dtype=int))
    apply_quantization_status(model, QuantizationStatus.COMPRESSED)
    for name, module in model.named_modules():
        if name in quant_config.ignore:
            continue
        module_type = module.__class__.__name__
        if module_type == "Linear":
            _test_layer_quantization_status(
                module,
                inputs=True,
                weights=True,
                expected_status=QuantizationStatus.COMPRESSED,
                expected_dtype=torch.int8,
            )


def test_serialize_config_tinyllama():
    quant_config = get_sample_tinyllama_quant_config()
    model = get_tinyllama_model()

    # check that model is not already quantized
    for module in model.modules():
        _test_layer_quantization_status(module, inputs=False, weights=False)

    # apply quant config to model
    apply_quantization_config(model, quant_config)

    serialized_config = QuantizationConfig.from_pretrained(model)
    assert len(serialized_config.config_groups) == 2
    assert serialized_config.config_groups["group_0"].targets == ["Embedding"]
    assert serialized_config.config_groups["group_0"].input_activations is None
    assert serialized_config.config_groups["group_1"].targets == ["Linear"]
    assert serialized_config.config_groups["group_1"].input_activations is not None
    assert serialized_config.format == CompressionFormat.dense.value
    assert serialized_config.quant_method == DEFAULT_QUANTIZATION_METHOD
    assert serialized_config.ignore == ["model.layers.1.mlp.down_proj"]
    assert serialized_config.global_compression_ratio > 1.0
    assert serialized_config.global_compression_ratio < 8.0


def _test_layer_quantization_status(
    module,
    inputs: bool,
    weights: bool,
    expected_status: Optional[QuantizationStatus] = None,
    expected_dtype: Optional[torch.dtype] = None,
):
    # check if quantization is applied at all (true if inputs or weights targeted)
    quantized = inputs or weights
    assert hasattr(module, "quantization_scheme") == quantized
    assert hasattr(module, "quantization_status") == quantized
    if expected_status is not None:
        assert module.quantization_status is expected_status

    # check inputs matches expected
    assert hasattr(module, "input_scale") == inputs
    assert hasattr(module, "input_zero_point") == inputs

    # check weights matches expected
    assert hasattr(module, "weight_scale") == weights
    assert hasattr(module, "weight_zero_point") == weights
    if weights and expected_dtype is not None:
        assert module.weight.dtype is expected_dtype


def get_tinyllama_model():
    return AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        torch_dtype="auto",
    )


def get_sample_tinyllama_quant_config(status: str = "frozen"):
    config_dict = {
        "quant_method": "compressed-tensors",
        "format": "fakequant",
        "quantization_status": status,
        "global_compression_ratio": None,
        "config_groups": {
            "group_1": {
                "weights": {
                    "num_bits": 8,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "tensor",
                },
                "input_activations": {
                    "num_bits": 8,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "tensor",
                },
                "targets": ["Linear"],
            },
            "group_2": {
                "weights": {
                    "num_bits": 8,
                    "type": "int",
                    "symmetric": False,
                    "strategy": "tensor",
                },
                "input_activations": None,
                "targets": ["Embedding"],
            },
        },
        "ignore": ["LlamaRotaryEmbedding", "model.layers.1.mlp.down_proj"],
    }
    return QuantizationConfig.model_validate(config_dict)


@requires_accelerate()
@pytest.mark.parametrize(
    "ignore,should_raise_warning",
    [
        [("lm_head", "re:.*gate"), False],
        [("lm_head", "re:.*foobarbaz"), True],
    ],
)
def test_apply_quantization_status(caplog, ignore, should_raise_warning):
    import logging

    # load a dense, unquantized tiny llama model
    model = get_tinyllama_model()
    quantization_config_dict = {
        "quant_method": "sparseml",
        "format": "pack-quantized",
        "global_compression_ratio": None,
        "config_groups": {
            "group_1": {
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "symmetric": False,
                    "strategy": "tensor",
                },
                "targets": ["Linear"],
            }
        },
    }
    quantization_config_dict["ignore"] = ignore

    config = QuantizationConfig(**quantization_config_dict)
    config.quantization_status = QuantizationStatus.CALIBRATION

    # mismatch in the ignore key of quantization_config_dict
    with caplog.at_level(logging.WARNING):
        apply_quantization_config(model, config)
        if should_raise_warning:
            assert len(caplog.text) > 0
        else:
            assert len(caplog.text) == 0


@pytest.mark.parametrize(
    "targets, ignore, expected_targets",
    [
        ([], [], set()),
        (["layer1", "layer2"], [], {"layer1", "layer2"}),
        ([], ["layer1"], set()),
        (["layer1", "layer2"], ["layer2"], {"layer1"}),
        (["re:layer.*"], ["layer3"], {"layer1", "layer2"}),
    ],
)
def test_expand_targets_with_mock(mock_model, targets, ignore, expected_targets):
    expanded_targets = expand_sparse_target_names(mock_model, targets, ignore)
    assert expanded_targets == expected_targets


@pytest.mark.parametrize(
    "targets, ignore, expected_targets",
    [
        (
            ["re:model.layers.[01].self_attn.q_proj"],
            ["re:model.layers.1.self_attn.q_proj"],
            set(["model.layers.0.self_attn.q_proj"]),
        ),
        (
            ["re:model.layers.[01].self_attn.q_proj"],
            [],
            set(["model.layers.0.self_attn.q_proj", "model.layers.1.self_attn.q_proj"]),
        ),
        (
            ["re:model.layers.[0-2].self_attn.q_proj"],
            ["re:model.layers.1.self_attn.q_proj"],
            set(["model.layers.0.self_attn.q_proj", "model.layers.2.self_attn.q_proj"]),
        ),
        (
            ["model.layers.0.self_attn.q_proj"],
            ["model.layers.0.self_attn.q_proj"],
            set(),
        ),
        (
            ["re:model.layers.*.self_attn.q_proj"],
            ["re:model.layers.[01].self_attn.q_proj"],
            set(
                f"model.layers.{layer_idx}.self_attn.q_proj"
                for layer_idx in range(2, 6)
            ),
        ),
    ],
)
def test_expand_targets_with_llama_stories(
    llama_stories_model, targets, ignore, expected_targets
):
    expanded_targets = expand_sparse_target_names(llama_stories_model, targets, ignore)
    assert expanded_targets == expected_targets


@pytest.mark.parametrize(
    "name, targets, ignore, expected",
    [
        ("layer1", ["layer1"], [], True),
        ("layer1", ["layer1"], ["layer1"], False),
        ("layer1", ["layer2"], [], False),
        ("layer1", ["re:layer.*"], [], True),
        ("layer1", ["re:layer.*"], ["re:layer1"], False),
    ],
)
def test_is_target_with_mock(mock_module, name, targets, ignore, expected):
    result = is_sparse_target(name, mock_module, targets, ignore)
    assert result == expected
