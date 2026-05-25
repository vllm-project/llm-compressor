import os
from pathlib import Path

import pytest
import torch
from compressed_tensors.utils import patch_attr
from safetensors import safe_open
from transformers import AutoModelForCausalLM
from transformers import initialization as init
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4Experts,
    DeepseekV4PreTrainedModel,
)
from transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe import (
    Qwen3VLMoeTextConfig,
)
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeTextExperts

from llmcompressor.modeling.moe.context import (
    moe_calibration_context,
)
from llmcompressor.modeling.moe.helpers import FusedExpertsProtocol, MoEConfig
from llmcompressor.modeling.moe.linearize import linearize_moe, load_quantizable_moe
from tests.testing_utils import requires_gpu

NUM_TEST_TOKENS = 64
MODEL_MSE = 1e-2
MODULE_MSE = 1e-10


@pytest.fixture
def patch_deepseek_fp32_modules():
    """
    Monkey patch to force DeepseekV4 models to load in bfloat16.

    BUG: norms should be loaded in float32, but usually aren't due to the base
    model having a quant_config which overrides this. Loading in float32 actually
    breaks the model definition (it expects bfloat16). Let's force load in bfloat16.
    """
    with patch_attr(DeepseekV4PreTrainedModel, "_keep_in_fp32_modules_strict", set()):
        yield


@torch.no_grad()
@requires_gpu
@pytest.mark.parametrize(
    "model_stub,exp_keys",
    [
        (
            "inference-optimization/DSV4-tiny-empty",
            [
                "model.layers.0.mlp.experts.2.up_proj.weight",
                "model.layers.1.mlp.experts.0.gate_proj.weight",
                "model.layers.2.mlp.experts.1.down_proj.weight",
            ],
        )
    ],
)
def test_load_quantizable_moe(
    model_stub, exp_keys, tmp_path, patch_deepseek_fp32_modules
):
    save_dir = tmp_path / "offload_dir"
    os.mkdir(save_dir)

    input_ids = torch.randint(1024, size=(1, NUM_TEST_TOKENS), device="cuda")
    model = AutoModelForCausalLM.from_pretrained(model_stub, device_map="cuda")
    true_outputs = model(input_ids=input_ids).logits
    del model

    with load_quantizable_moe():
        model2 = AutoModelForCausalLM.from_pretrained(model_stub, device_map="cuda")

    select_exp_outputs = model2(input_ids=input_ids).logits

    with moe_calibration_context():
        all_exp_outputs = model2(input_ids=input_ids).logits

    assert torch.any(true_outputs != 0), "Bad test setup, output is all zeros"
    assert torch.nn.functional.mse_loss(true_outputs, select_exp_outputs) < MODEL_MSE
    assert torch.nn.functional.mse_loss(true_outputs, all_exp_outputs) < MODEL_MSE

    model2.save_pretrained(save_dir)
    assert keys_exist(save_dir, exp_keys)


def keys_exist(model_path: Path, keys: list[str]) -> bool:
    """
    Utility to check that expected expert keys exist in a saved model.

    Args:
        model_path: Path to the saved model directory
        expected_patterns: List of key patterns to check for

    Returns:
        True if all expected patterns are found in the model checkpoint
    """
    safetensor_files = list(model_path.glob("*.safetensors"))
    all_keys = set()
    keys = set(keys)

    for st_file in safetensor_files:
        with safe_open(st_file, framework="pt", device="cpu") as f:
            all_keys.update(f.keys())

    return keys <= all_keys


class DummyModel(torch.nn.Module):
    def __init__(self, module, config):
        super().__init__()
        self.config = config
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


@torch.no_grad()
@requires_gpu
@pytest.mark.parametrize(
    "config_cls,experts_cls",
    [
        (DeepseekV4Config, DeepseekV4Experts),
        (Qwen3VLMoeTextConfig, Qwen3VLMoeTextExperts),
    ],
)
def test_linearize_moe(config_cls, experts_cls):
    with torch.device("cuda"):
        config = config_cls()
        experts = experts_cls(config)
        assert isinstance(experts, FusedExpertsProtocol)
        init.normal_(experts.gate_up_proj, mean=0.0, std=config.initializer_range)
        init.normal_(experts.down_proj, mean=0.0, std=config.initializer_range)

        mock_model = DummyModel(experts, config)
        linearize_moe(mock_model)

        moe_config = MoEConfig.from_config(config)

        hidden_states = torch.randn(
            NUM_TEST_TOKENS, moe_config.hidden_dim, dtype=moe_config.dtype
        )
        top_k_index = torch.randint(
            0,
            moe_config.num_experts,
            size=(NUM_TEST_TOKENS, moe_config.num_experts_per_tok),
        )
        top_k_weights = torch.randn(
            NUM_TEST_TOKENS, moe_config.num_experts_per_tok, dtype=moe_config.dtype
        )
        true_outputs = experts(hidden_states, top_k_index, top_k_weights)
        outputs = mock_model(hidden_states, top_k_index, top_k_weights)

        assert torch.any(true_outputs != 0), "Bad test setup, output is all zeros"
        print(torch.nn.functional.mse_loss(outputs, true_outputs))
        assert torch.nn.functional.mse_loss(outputs, true_outputs) < MODULE_MSE
