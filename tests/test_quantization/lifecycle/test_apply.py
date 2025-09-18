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
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
    QuantizationStrategy,
    QuantizationType,
)
from compressed_tensors.quantization.lifecycle import apply_quantization_config
from compressed_tensors.utils import is_match, match_named_modules
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

    for name, module in model.named_modules():
        if name == "model.layers.0.mlp.down_proj":
            assert module.quantization_scheme.weights.num_bits == 2
        elif re.match(".*down_proj", name):
            assert module.quantization_scheme.weights.num_bits == 4
        elif isinstance(module, torch.nn.Linear):
            assert module.quantization_scheme.weights.num_bits == 8


def test_apply_quantization_config_tinyllama():
    quant_config = get_sample_tinyllama_quant_config(
        status=QuantizationStatus.INITIALIZED
    )
    model = get_tinyllama_model()

    # check that model is not already quantized
    for module in model.modules():
        _test_layer_quantization_status(module, inputs=False, weights=False)

    # apply quant config to model
    apply_quantization_config(model, quant_config)

    # check for correct application of quant config
    for quant_scheme in quant_config.config_groups.values():
        for name, module in match_named_modules(
            model, quant_scheme.targets, quant_config.ignore
        ):
            _test_layer_quantization_status(
                module,
                inputs=quant_scheme.input_activations is not None,
                weights=quant_scheme.weights is not None,
                expected_status=QuantizationStatus.INITIALIZED,
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
    if serialized_config.global_compression_ratio is not None:
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


def get_sample_tinyllama_quant_config(
    status: QuantizationStatus = QuantizationStatus.FROZEN,
):
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
    "target,should_raise_warning",
    [
        [("Linear",), False],
        [("Linear", "re:.*foobarbaz"), True],
    ],
)
def test_apply_quantization_config(caplog, target, should_raise_warning):
    import logging

    # load a dense, unquantized tiny llama model
    model = get_tinyllama_model()
    quantization_config_dict = {
        "quant_method": "compressed-tensors",
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
                "targets": target,
            }
        },
        "ignore": ["lm_head", "re:.*gate"],
    }

    config = QuantizationConfig(**quantization_config_dict)
    config.quantization_status = QuantizationStatus.CALIBRATION

    # mismatch in the ignore key of quantization_config_dict
    with caplog.at_level(logging.WARNING):
        apply_quantization_config(model, config)
        if should_raise_warning:
            assert len(caplog.text) > 0
        else:
            assert len(caplog.text) == 0


def test_multi_apply_quantization_config():
    """
    Ensure that multiple quantization configs are applied correctly
    If quantization config was previously applied to a module,
    those changes should be reset for newly applied quantization config
    """
    model = get_tinyllama_model()

    # FP8 applied to self_attn
    qconfig1 = QuantizationConfig(
        config_groups={
            "group_0": QuantizationScheme(
                targets=[
                    r"re:.*self_attn\.(k|q|o|v)_proj$",
                ],
                weights=QuantizationArgs(
                    num_bits=8,
                    type=QuantizationType.FLOAT,
                    strategy=QuantizationStrategy.TENSOR,
                    symmetric=True,
                    dynamic=False,
                ),
                input_activations=QuantizationArgs(
                    num_bits=8,
                    type=QuantizationType.FLOAT,
                    strategy=QuantizationStrategy.TENSOR,
                    symmetric=True,
                    dynamic=False,
                ),
            )
        },
        ignore=["lm_head"],
    )
    # W4A16_ASYM applied to mlp and self_attn.o_proj to validate overwriting
    qconfig2 = QuantizationConfig(
        config_groups={
            "group_0": QuantizationScheme(
                targets=[
                    r"re:.*mlp\.(down|gate|up)_proj$",
                    r"re:.*self_attn\.o_proj$",
                ],
                weights=QuantizationArgs(
                    num_bits=4,
                    type=QuantizationType.INT,
                    strategy=QuantizationStrategy.GROUP,
                    group_size=128,
                    symmetric=False,
                    dynamic=False,
                ),
            )
        },
        ignore=["lm_head"],
    )

    apply_quantization_config(model, qconfig1)
    apply_quantization_config(model, qconfig2)
    for name, module in model.named_modules():
        if is_match(
            name, module, qconfig2.config_groups["group_0"].targets, qconfig2.ignore
        ):
            # assert W4A16_ASYM parameters are present with correct shape
            # and FP8 parameters have been removed
            assert not hasattr(module, "input_scale")
            assert not hasattr(module, "input_zero_point")
            weight_scale = getattr(module, "weight_scale", None)
            assert (
                weight_scale is not None
                and weight_scale.shape[:-1] == module.weight.shape[:-1]
                and weight_scale.shape[-1] == module.weight.shape[-1] / 128
            )
            weight_zero_point = getattr(module, "weight_zero_point", None)
            assert (
                weight_zero_point is not None
                and weight_zero_point.shape[:-1] == module.weight.shape[:-1]
                and weight_zero_point.shape[-1] == module.weight.shape[-1] / 128
            )

        elif is_match(
            name, module, qconfig1.config_groups["group_0"].targets, qconfig1.ignore
        ):
            # assert FP8 scheme parameters are present with correct shape
            input_scale = getattr(module, "input_scale", None)
            assert input_scale is not None and input_scale.shape == torch.Size([1])
            input_zero_point = getattr(module, "input_zero_point", None)
            assert (
                input_zero_point is not None
                and input_zero_point.shape == torch.Size([1])
            )
            weight_scale = getattr(module, "weight_scale", None)
            assert weight_scale is not None and weight_scale.shape == torch.Size([1])
            weight_zero_point = getattr(module, "weight_zero_point", None)
            assert (
                weight_zero_point is not None
                and weight_zero_point.shape == torch.Size([1])
            )
