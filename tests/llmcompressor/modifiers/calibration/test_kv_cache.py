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
import pytest
import torch
from compressed_tensors.quantization import (
    QuantizationConfig,
    QuantizationStatus,
    apply_quantization_config,
    is_attention_module,
)
from transformers import AutoModelForCausalLM

from llmcompressor.modifiers.quantization.calibration import (
    calibrate_kv_cache_input_hook,
    calibrate_kv_cache_output_hook,
    freeze_module_quantization,
    initialize_quantized_kv_cache,
)

config = {
    "quant_method": "compressed-tensors",
    "format": "fakequant",
    "kv_cache_scheme": {
        "num_bits": 8,
        "type": "int",
        "symmetric": True,
        "strategy": "tensor",
    },
    "config_groups": {
        "group_1": {
            "weights": {
                "num_bits": 4,
                "type": "int",
                "symmetric": True,
                "strategy": "tensor",
            },
            "targets": ["Linear"],
        },
    },
}


def _prep_for_calibration(module: torch.nn.Module):
    if is_attention_module(module):
        module.register_forward_pre_hook(
            calibrate_kv_cache_input_hook, with_kwargs=True
        )
        module.register_forward_hook(calibrate_kv_cache_output_hook)
    module.quantization_status = QuantizationStatus.CALIBRATION


@pytest.mark.parametrize("config", [config])
def test_kv_cache_quantization(config):
    sample = {
        name: torch.ones((1, 32)).long()
        for name in ["input_ids", "attention_mask", "labels"]
    }
    model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceM4/tiny-random-LlamaForCausalLM",
        torch_dtype="auto",
    )
    model.eval()

    config = QuantizationConfig(**config)
    config.quantization_status = QuantizationStatus.CALIBRATION
    apply_quantization_config(model, config)
    model.apply(initialize_quantized_kv_cache)
    model.apply(_prep_for_calibration)

    with torch.no_grad():
        _ = model(**sample)

    model.apply(freeze_module_quantization)

    reloaded_config = QuantizationConfig.from_pretrained(model)

    assert (
        config.kv_cache_scheme.model_dump().keys()
        == reloaded_config.kv_cache_scheme.model_dump().keys()
    )
    assert list(config.kv_cache_scheme.model_dump().values()) == list(
        reloaded_config.kv_cache_scheme.model_dump().values()
    )
