import os
from pathlib import Path

import pytest
import torch
from compressed_tensors.utils import patch_attr
from safetensors import safe_open
from transformers import AutoModelForCausalLM
from transformers import initialization as init
from transformers.models.afmoe.configuration_afmoe import AfmoeConfig
from transformers.models.afmoe.modeling_afmoe import AfmoeExperts
from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3Experts
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4Experts,
    DeepseekV4PreTrainedModel,
)
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextExperts
from transformers.models.glm4_moe.configuration_glm4_moe import Glm4MoeConfig
from transformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeExperts
from transformers.models.glm4_moe_lite.configuration_glm4_moe_lite import (
    Glm4MoeLiteConfig,
)
from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import Glm4MoeLiteExperts
from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaExperts
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts
from transformers.models.granitemoe.configuration_granitemoe import GraniteMoeConfig
from transformers.models.granitemoe.modeling_granitemoe import GraniteMoeExperts
from transformers.models.hy_v3.configuration_hy_v3 import HYV3Config
from transformers.models.hy_v3.modeling_hy_v3 import HYV3Experts
from transformers.models.llama4.configuration_llama4 import (
    Llama4Config,
    Llama4TextConfig,
)
from transformers.models.llama4.modeling_llama4 import Llama4TextExperts
from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig
from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHExperts
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeExperts
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts
from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextExperts
from transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe import (
    Qwen3VLMoeTextConfig,
)
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeTextExperts

from llmcompressor.modeling.moe.context import (
    moe_calibration_context,
)
from llmcompressor.modeling.moe.helpers import (
    FusedExpertsProtocol,
    MoEConfig,
    _getattr_fallbacks,
)
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
        ),
        (
            "inference-optimization/Qwen3-1.6B-A0.9B",
            [
                "model.layers.0.mlp.experts.2.up_proj.weight",
                "model.layers.1.mlp.experts.0.gate_proj.weight",
                "model.layers.2.mlp.experts.1.down_proj.weight",
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
    "config_cls,experts_cls,kwargs",
    [
        (
            AfmoeConfig,
            AfmoeExperts,
            {"hidden_size": 512, "intermediate_size": 1024, "num_experts": 16},
        ),
        (
            DeepseekV3Config,
            DeepseekV3Experts,
            {"hidden_size": 512, "moe_intermediate_size": 1024, "n_routed_experts": 16},
        ),
        (
            DeepseekV4Config,
            DeepseekV4Experts,
            {"hidden_size": 512, "moe_intermediate_size": 1024, "n_routed_experts": 16},
        ),
        (
            Gemma4TextConfig,
            Gemma4TextExperts,
            {"num_experts": 16, "top_k_experts": 4, "moe_intermediate_size": 2304},
        ),
        (Glm4MoeConfig, Glm4MoeExperts, {}),
        (Glm4MoeLiteConfig, Glm4MoeLiteExperts, {}),
        (GlmMoeDsaConfig, GlmMoeDsaExperts, {"hidden_size": 512}),
        (
            GraniteMoeConfig,
            GraniteMoeExperts,
            {"hidden_size": 512, "intermediate_size": 1024, "num_local_experts": 4},
        ),
        (Qwen3_5MoeTextConfig, Qwen3_5MoeExperts, {}),
        (Qwen3MoeConfig, Qwen3MoeExperts, {}),
        (Qwen3NextConfig, Qwen3NextExperts, {}),
        (Qwen3VLMoeTextConfig, Qwen3VLMoeTextExperts, {}),
        (GptOssConfig, GptOssExperts, {}),
        (HYV3Config, HYV3Experts, {"hidden_size": 512, "moe_intermediate_size": 1024}),
        (
            NemotronHConfig,
            NemotronHExperts,
            {"hidden_size": 32, "moe_intermediate_size": 64},
        ),
    ],
)
def test_linearize_moe(config_cls, experts_cls, kwargs):
    with torch.device("cuda"):
        config = config_cls(**kwargs)
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


def test_linearize_moe_llama4():
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


def test_linearize_quant_config():
    from compressed_tensors.quantization import (
        QuantizationConfig,
        QuantizationScheme,
        apply_quantization_config,
    )
    from compressed_tensors.quantization.quant_scheme import W4A16

    with load_quantizable_moe():
        model = AutoModelForCausalLM.from_pretrained(
            "inference-optimization/GLM-5.2-0.8B-A0.8B"
        )
    qscheme = QuantizationScheme(targets=["Linear"], **W4A16)
    qconfig = QuantizationConfig(config_groups={"": qscheme})
    apply_quantization_config(model, qconfig)

    save_qconfig = QuantizationConfig.from_pretrained(model)
    assert set(save_qconfig.ignore) == {
        "model.layers.2.mlp.gate",
        "model.layers.3.mlp.gate",
        "model.layers.4.mlp.gate",
        "model.layers.5.mlp.gate",
    }
