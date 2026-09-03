import os
from pathlib import Path

import pytest
import torch
from compressed_tensors.utils import patch_attr
from huggingface_hub.errors import StrictDataclassError
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedConfig
from transformers import initialization as init
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4PreTrainedModel,
)

from llmcompressor.modeling.moe.context import moe_calibration_context
from llmcompressor.modeling.moe.conversion_mappings import ARCH_TO_IMPORT_PATHS
from llmcompressor.modeling.moe.helpers import (
    FusedExpertsProtocol,
    MoEConfig,
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
    "gpt_oss": {"hidden_size": 256, "intermediate_size": 256, "num_local_experts": 16},
    "hy_v3": {"hidden_size": 256, "moe_intermediate_size": 256, "num_experts": 16},
    "jamba": {"hidden_size": 256, "intermediate_size": 256, "num_experts": 16},
    "nemotron_h": {"hidden_size": 32, "moe_intermediate_size": 64},
    "deepseek_v4": {
        "hidden_size": 512,
        "moe_intermediate_size": 64,
        "n_routed_experts": 16,
    },
}


@pytest.fixture
def patch_deepseek_fp32_modules():
    """
    Monkey patch to force DeepseekV4 models to load in bfloat16.

    BUG: norms should be loaded in float32, but usually aren't due to the base
    model having a quant_config which overrides this. Loading in float32 actually
    breaks the model definition (it expects bfloat16). Let's force load in bfloat16.
    # Fixed upstream by: https://github.com/huggingface/transformers/pull/47486
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
                "model.layers.0.ffn.experts.2.w3.weight",
                "model.layers.1.ffn.experts.0.w1.weight",
                "model.layers.2.ffn.experts.1.w2.weight",
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
    try:
        AutoConfig.from_pretrained(model_stub)
    except StrictDataclassError:
        pytest.skip("Could not import model, please upgrade your transformers version")

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


def init_experts(experts: FusedExpertsProtocol, config: PreTrainedConfig):
    """
    Initialize every parameter of an experts module.

    Experts modules allocate their parameters with `torch.empty`, so a parameter which
    is not initialized here holds whatever the allocator hands back. Architectures with
    expert biases (such as `gpt_oss`) then read that memory during forward, which
    produces non-finite outputs for whichever tokens are routed to the affected expert.

    :param experts: fused experts module to initialize in place
    :param config: config of the model the experts module belongs to
    """
    for parameter in experts.parameters():
        init.normal_(parameter, mean=0.0, std=config.initializer_range)


@torch.no_grad()
@requires_gpu
@pytest.mark.parametrize("model_type", list(ARCH_TO_IMPORT_PATHS.keys() - {"llama4"}))
def test_linearize_moe(model_type):
    config_path, experts_path = ARCH_TO_IMPORT_PATHS[model_type]
    config_cls = import_or_none(config_path)
    experts_cls = import_or_none(experts_path)

    if config_cls is None or experts_cls is None:
        pytest.skip(
            f"Could not import {model_type}, please upgrade your transformers version"
        )

    with torch.device("cuda"):
        config = config_cls(**CONFIG_OVERRIDES.get(model_type, {}))
        experts = experts_cls(config)
        assert isinstance(experts, FusedExpertsProtocol)
        init_experts(experts, config)

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
        assert torch.isfinite(
            true_outputs
        ).all(), "Bad test setup, output is not finite"
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
    init_experts(experts, text_config)

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


def test_linearize_moe_gpt_oss():
    """
    `gpt_oss` is an architecture whose experts carry biases, which are the parameters
    most easily left uninitialized by a test. Cover it on cpu so that a forward through
    uninitialized memory, which yields non-finite outputs and a NaN mse, is caught
    without requiring a gpu.
    """
    config_path, experts_path = ARCH_TO_IMPORT_PATHS["gpt_oss"]
    config_cls = import_or_none(config_path)
    experts_cls = import_or_none(experts_path)

    if config_cls is None or experts_cls is None:
        pytest.skip(
            "Could not import gpt_oss, please upgrade your transformers version"
        )

    config = config_cls(**CONFIG_OVERRIDES["gpt_oss"])
    experts = experts_cls(config)
    init_experts(experts, config)
    assert all(torch.isfinite(parameter).all() for parameter in experts.parameters())

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
    assert torch.isfinite(true_outputs).all(), "Bad test setup, output is not finite"
    assert torch.nn.functional.mse_loss(outputs, true_outputs) < MODULE_MSE
    assert torch.nn.functional.mse_loss(calib_outputs, true_outputs) < MODULE_MSE
