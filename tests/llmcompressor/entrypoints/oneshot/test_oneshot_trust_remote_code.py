import json

import pytest
from transformers import LlamaConfig, LlamaForCausalLM

from llmcompressor import oneshot
from llmcompressor.transformers.utils import is_model_ct_quantized_from_path

CONFIGURATION_CODE = """
from transformers import LlamaConfig


class CustomConfig(LlamaConfig):
    model_type = "custom_llama"
"""

MODELING_CODE = """
from transformers import LlamaForCausalLM

from .configuration_custom import CustomConfig


class CustomForCausalLM(LlamaForCausalLM):
    config_class = CustomConfig
"""


def _save_tiny_model(path, **config_updates):
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    LlamaForCausalLM(config).save_pretrained(path)

    config_path = path / "config.json"
    config_dict = json.loads(config_path.read_text())
    config_dict.update(config_updates)
    config_path.write_text(json.dumps(config_dict))


@pytest.fixture
def custom_code_model(tmp_path):
    (tmp_path / "configuration_custom.py").write_text(CONFIGURATION_CODE)
    (tmp_path / "modeling_custom.py").write_text(MODELING_CODE)
    _save_tiny_model(
        tmp_path,
        model_type="custom_llama",
        architectures=["CustomForCausalLM"],
        auto_map={
            "AutoConfig": "configuration_custom.CustomConfig",
            "AutoModelForCausalLM": "modeling_custom.CustomForCausalLM",
        },
    )
    return tmp_path


@pytest.mark.unit
def test_oneshot_loads_custom_code_model(custom_code_model):
    model = oneshot(
        model=str(custom_code_model), recipe=[], trust_remote_code_model=True
    )

    assert type(model).__name__ == "CustomForCausalLM"


@pytest.mark.unit
def test_is_model_ct_quantized_from_path_trust_remote_code(custom_code_model):
    assert not is_model_ct_quantized_from_path(
        str(custom_code_model), trust_remote_code=True
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "quantization_config, expected",
    [
        ({"quant_method": "compressed-tensors"}, True),
        ({"quant_method": "fp8"}, False),
        ({}, False),
        (None, False),
    ],
)
def test_is_model_ct_quantized_from_path(tmp_path, quantization_config, expected):
    _save_tiny_model(tmp_path, quantization_config=quantization_config)

    assert is_model_ct_quantized_from_path(str(tmp_path)) == expected
