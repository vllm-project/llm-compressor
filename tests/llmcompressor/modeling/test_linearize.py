import os
from pathlib import Path

import pytest
import torch
from compressed_tensors.utils import patch_attr
from safetensors import safe_open
from transformers import AutoModelForCausalLM
from transformers import initialization as init
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4PreTrainedModel,
)

from llmcompressor.modeling.moe.context import moe_calibration_context
from llmcompressor.modeling.moe.conversion_mappings import ARCH_TO_IMPORT_PATHS
from llmcompressor.modeling.moe.helpers import (
    FusedExpertsProtocol,
    MoEConfig,
    _getattr_fallbacks,
    import_or_none,
)
from llmcompressor.modeling.moe.linearize import linearize_moe, load_quantizable_moe
from tests.testing_utils import requires_gpu

NUM_TEST_TOKENS = 64
MODEL_MSE = 1e-2
MODULE_MSE = 1e-10
CONFIG_OVERRIDES = {
    "deepseek_ocr2": {"num_experts_per_tok": 16},
    "deepseek_v3": {"hidden_size": 512, "moe_intermediate_size": 1024},
    "cohere2_moe": {"hidden_size": 256, "intermediate_size": 256},
    "gemma4": {"num_experts": 16, "top_k_experts": 4, "moe_intermediate_size": 2304},
    "glm_moe_dsa": {"hidden_size": 512},
    "hy_v3": {"hidden_size": 256, "moe_intermediate_size": 256, "num_experts": 16},
    "jamba": {"hidden_size": 256, "intermediate_size": 256, "num_experts": 16},
    "nemotron_h": {"hidden_size": 32, "moe_intermediate_size": 64},
}


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
        ),
        (
            "inference-optimization/Qwen3-1.6B-A0.9B",
            [
                "model.layers.0.mlp.experts.2.up_proj.weight",
                "model.layers.1.mlp.experts.0.gate_proj.weight",
                "model.layers.2.mlp.experts.1.down_proj.weight",
            ],
        ),
        (
            "inference-optimization/GLM-5.2-0.8B-A0.8B",
            [
                "model.layers.2.mlp.experts.2.up_proj.weight",
                "model.layers.3.mlp.experts.0.gate_proj.weight",
                "model.layers.4.mlp.experts.1.down_proj.weight",
            ],
        ),
    ],
)
def test_load_quantizable_moe(
    model_stub, exp_keys, tmp_path, patch_deepseek_fp32_modules
):
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

    save_dir = tmp_path / "save_path"
    os.mkdir(save_dir)
    model2.save_pretrained(save_dir)
    assert_keys_exist(save_dir, exp_keys)


def assert_keys_exist(model_path: Path, keys: list[str]):
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

    assert keys <= all_keys, all_keys


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
    "model_type", list(ARCH_TO_IMPORT_PATHS.keys() - {"llama4", "granitemoe"})
)
def test_linearize_moe(model_type):
    config_path, experts_path = ARCH_TO_IMPORT_PATHS[model_type]
    config_cls = import_or_none(config_path)
    experts_cls = import_or_none(experts_path)

    assert config_cls is not None, f"Could not import config for {model_type}"
    assert experts_cls is not None, f"Could not import experts for {model_type}"

    with torch.device("cuda"):
        config = config_cls(**CONFIG_OVERRIDES.get(model_type, {}))
        experts = experts_cls(config)
        assert isinstance(experts, FusedExpertsProtocol)
        up_proj = _getattr_fallbacks(experts, ["gate_up_proj", "up_proj"])
        init.normal_(up_proj, mean=0.0, std=config.initializer_range)
        init.normal_(experts.down_proj, mean=0.0, std=config.initializer_range)

        mock_model = DummyModel(experts, config)
        linearize_moe(mock_model)
        assert mock_model.module is not experts

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
        with moe_calibration_context():
            calib_outputs = mock_model(hidden_states, top_k_index, top_k_weights)

        assert torch.any(true_outputs != 0), "Bad test setup, output is all zeros"
        assert torch.nn.functional.mse_loss(outputs, true_outputs) < MODULE_MSE
        assert torch.nn.functional.mse_loss(calib_outputs, true_outputs) < MODULE_MSE


def test_linearize_moe_granite():
    try:
        from transformers.models.granitemoe.configuration_granitemoe import (
            GraniteMoeConfig,
        )
        from transformers.models.granitemoe.modeling_granitemoe import (
            GraniteMoeParallelExperts,
        )
    except ImportError:
        pytest.skip("GraniteMoeParallelExperts has been removed")

    config = GraniteMoeConfig(hidden_size=512, intermediate_size=1024)
    experts = GraniteMoeParallelExperts(
        config.num_local_experts, config.hidden_size, config.intermediate_size
    )
    init.normal_(experts.weight, mean=0.0, std=config.initializer_range)

    mock_model = DummyModel(experts, config)
    linearize_moe(mock_model)
    assert mock_model.module is not experts

    hidden_states = torch.randn(NUM_TEST_TOKENS, config.hidden_size, dtype=config.dtype)
    expert_size = [
        (NUM_TEST_TOKENS // config.num_local_experts)
        for _ in range(config.num_local_experts)
    ]
    expert_size[-1] += NUM_TEST_TOKENS % config.num_local_experts
    true_outputs = experts(hidden_states, expert_size)
    outputs = mock_model(hidden_states, expert_size)
    with moe_calibration_context():
        calib_outputs = mock_model(hidden_states, expert_size)

    assert torch.any(true_outputs != 0), "Bad test setup, output is all zeros"
    assert torch.nn.functional.mse_loss(outputs, true_outputs) < MODULE_MSE
    assert torch.nn.functional.mse_loss(calib_outputs, true_outputs) < MODULE_MSE


def test_linearize_moe_llama4():
    from transformers.models.llama4.configuration_llama4 import (
        Llama4Config,
        Llama4TextConfig,
    )
    from transformers.models.llama4.modeling_llama4 import Llama4TextExperts

    text_config = Llama4TextConfig(hidden_size=512, intermediate_size=1024)
    config = Llama4Config(text_config=text_config)
    experts = Llama4TextExperts(config.text_config)
    init.normal_(experts.gate_up_proj, mean=0.0, std=text_config.initializer_range)
    init.normal_(experts.down_proj, mean=0.0, std=text_config.initializer_range)

    mock_model = DummyModel(experts, config)
    linearize_moe(mock_model)
    assert mock_model.module is not experts

    moe_config = MoEConfig.from_config(text_config)
    hidden_states = torch.randn(
        NUM_TEST_TOKENS, moe_config.hidden_dim, dtype=moe_config.dtype
    )
    true_outputs = experts(hidden_states)
    outputs = mock_model(hidden_states)
    with moe_calibration_context():
        calib_outputs = mock_model(hidden_states)

    assert torch.any(true_outputs != 0), "Bad test setup, output is all zeros"
    assert torch.nn.functional.mse_loss(outputs, true_outputs) < MODULE_MSE
    assert torch.nn.functional.mse_loss(calib_outputs, true_outputs) < MODULE_MSE
