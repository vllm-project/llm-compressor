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
from compressed_tensors.quantization.lifecycle import (
    apply_quantization_config,
    freeze_module_quantization,
)
from compressed_tensors.quantization.quant_config import QuantizationConfig
from transformers import AutoModelForCausalLM


def test_apply_tinyllama_dynamic_activations():
    quant_config = get_sample_dynamic_tinyllama_quant_config()
    model = get_tinyllama_model()

    # check that model is not already quantized
    for module in model.modules():
        _test_layer_dynamic_quantization_status(module, inputs=False, weights=False)

    # apply quant config to model
    apply_quantization_config(model, quant_config)

    # test linears are dynamically quantized for calibration
    _test_linears_dynamic_quantization_status(model, quant_config, frozen=False)
    # verify forward works w/ dynamic during calibration
    model(torch.zeros((1, 1), dtype=int), torch.zeros((1, 1), dtype=int))

    # freeze and test that only weight observers are deleted
    model.apply(freeze_module_quantization)
    _test_linears_dynamic_quantization_status(model, quant_config, frozen=True)
    # verify forward works w/ dynamic after freeze
    model(torch.zeros((1, 1), dtype=int), torch.zeros((1, 1), dtype=int))


def _test_linears_dynamic_quantization_status(model, quant_config, frozen: bool):
    # check for correct application of quant config
    num_linears = 0
    for name, module in model.named_modules():
        if name in quant_config.ignore:
            continue
        module_type = module.__class__.__name__
        if module_type == "Linear":
            num_linears += 1
            _test_layer_dynamic_quantization_status(
                module, inputs=True, weights=True, frozen=frozen
            )

    # sanity check correct number of layers targeted
    assert num_linears == 154  # 155 Linear layers - 1 that gets ignored


def _test_layer_dynamic_quantization_status(
    module, inputs: bool, weights: bool, frozen: bool = False
):
    # check if quantization is applied at all (true if inputs or weights targeted)
    quantized = inputs or weights
    assert hasattr(module, "quantization_scheme") == quantized
    assert hasattr(module, "quantization_status") == quantized

    # check inputs always have an observer if quantized but never scale/zp
    assert not hasattr(module, "input_scale")
    assert not hasattr(module, "input_zero_point")
    assert hasattr(module, "input_observer") == inputs

    # check weights always have scale/zp and observer only if not frozen
    assert hasattr(module, "weight_scale") == weights
    assert hasattr(module, "weight_zero_point") == weights
    assert hasattr(module, "weight_observer") == (weights and not frozen)


def get_tinyllama_model():
    return AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    )


def get_sample_dynamic_tinyllama_quant_config():
    config_dict = {
        "quant_method": "sparseml",
        "format": "fakequant",
        "quantization_status": "calibration",
        "global_compression_ratio": None,
        "config_groups": {
            "group_1": {
                "weights": {
                    "num_bits": 8,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "tensor",
                    "dynamic": False,
                },
                "input_activations": {
                    "num_bits": 8,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "tensor",
                    "dynamic": True,
                },
                "targets": ["Linear"],
            },
        },
        "ignore": ["LlamaRotaryEmbedding", "model.layers.1.mlp.down_proj"],
    }
    return QuantizationConfig.parse_obj(config_dict)
