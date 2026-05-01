from functools import partial

import pytest
import torch
import transformers

_TRANSFORMERS_MAJOR = int(transformers.__version__.split(".")[0])

try:
    from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import (
        GlmMoeDsaConfig,
    )
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaMoE
except ImportError:
    if _TRANSFORMERS_MAJOR < 5:
        pytest.skip("glm_moe_dsa requires transformers >= 5.x", allow_module_level=True)
    raise

from llmcompressor.modeling.glm_moe_dsa import CalibrationGlmMoeDsaMoE  # noqa: E402
from llmcompressor.utils.helpers import calibration_forward_context  # noqa: E402
from tests.testing_utils import requires_gpu, requires_transformers_v5  # noqa: E402

pytestmark = requires_transformers_v5


def _tiny_config():
    """Small config for fast unit tests (8 experts instead of 256)."""
    return GlmMoeDsaConfig(
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_shared_experts=1,
        n_group=1,
        topk_group=1,
        num_hidden_layers=4,
    )


@requires_gpu
def test_calib_glm_moe_dsa_all_experts_triggered():
    config = _tiny_config()
    with torch.device("cuda"):
        original = GlmMoeDsaMoE(config)
        for param in original.parameters():
            param.data.normal_(mean=0.0, std=0.02)

    module = CalibrationGlmMoeDsaMoE(original, config, calibrate_all_experts=True)

    num_experts = len(module.experts)
    expert_triggered = [False for _ in range(num_experts)]

    def hook_fn(i, module, input, output):
        expert_triggered[i] = True

    for i, expert in enumerate(module.experts):
        expert.register_forward_hook(partial(hook_fn, i))

    hidden_dim = config.hidden_size
    batch, seq_len = 4, 32
    sample = torch.randn(batch, seq_len, hidden_dim, device="cuda")

    with calibration_forward_context(module):
        with torch.no_grad():
            _ = module(sample)

    assert all(expert_triggered), f"Not all experts were triggered: {expert_triggered}"


@requires_gpu
def test_calib_glm_moe_dsa_output_matches():
    config = _tiny_config()
    with torch.device("cuda"):
        original = GlmMoeDsaMoE(config)
        for param in original.parameters():
            param.data.normal_(mean=0.0, std=0.02)

    hidden_dim = config.hidden_size
    batch, seq_len = 4, 32
    sample = torch.randn(batch, seq_len, hidden_dim, device="cuda")

    with calibration_forward_context(original):
        true_out = original(sample)

    module = CalibrationGlmMoeDsaMoE(original, config, calibrate_all_experts=True)
    with calibration_forward_context(module):
        out = module(sample)
        assert torch.nn.functional.mse_loss(true_out, out) < 0.1

    module = CalibrationGlmMoeDsaMoE(original, config, calibrate_all_experts=False)
    with calibration_forward_context(module):
        out = module(sample)
        assert torch.nn.functional.mse_loss(true_out, out) < 0.1


@requires_gpu
def test_calib_glm_moe_dsa_experts_are_linear():
    """Verify that after unpacking, experts contain nn.Linear modules
    visible to named_modules(), which is the whole point of this fix."""
    config = _tiny_config()
    with torch.device("cuda"):
        original = GlmMoeDsaMoE(config)

    module = CalibrationGlmMoeDsaMoE(original, config, calibrate_all_experts=True)

    linear_names = [
        name for name, mod in module.named_modules() if isinstance(mod, torch.nn.Linear)
    ]
    # Each expert should have gate_proj, up_proj, down_proj = 3 Linear per expert
    # Plus shared_experts has 3 Linear
    expected_expert_linears = config.num_local_experts * 3
    expected_shared_linears = 3
    assert len(linear_names) == expected_expert_linears + expected_shared_linears, (
        f"Expected {expected_expert_linears + expected_shared_linears} Linear modules, "
        f"found {len(linear_names)}: {linear_names}"
    )
