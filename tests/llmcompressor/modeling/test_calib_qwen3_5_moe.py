import contextlib
from functools import partial

import pytest
import torch
import transformers

_TRANSFORMERS_MAJOR = int(transformers.__version__.split(".")[0])

try:
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
        Qwen3_5MoeTextConfig,
    )
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeSparseMoeBlock,
    )
except ImportError:
    if _TRANSFORMERS_MAJOR < 5:
        pytest.skip("qwen3_5_moe requires transformers >= 5.x", allow_module_level=True)
    raise

from llmcompressor.modeling.moe_context import moe_calibration_context  # noqa: E402
from llmcompressor.modeling.qwen3_5_moe import (  # noqa: E402
    CalibrationQwen3_5MoeSparseMoeBlock,
)
from llmcompressor.utils.dev import skip_weights_download  # noqa: E402
from llmcompressor.utils.helpers import calibration_forward_context  # noqa: E402
from tests.testing_utils import (  # noqa: E402
    requires_cadence,
    requires_gpu,
    requires_transformers_v5,
)

pytestmark = requires_transformers_v5


def _tiny_config():
    """Small config for fast unit tests (8 experts instead of the default)."""
    return Qwen3_5MoeTextConfig(
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_experts=8,
        num_experts_per_tok=2,
        num_hidden_layers=2,
    )


@requires_cadence("weekly")
@pytest.mark.parametrize("model_stub", ["Qwen/Qwen3.5-35B-A3B"])
def test_calib_replace_qwen3_5_moe_all_experts(model_stub):
    from transformers import AutoModelForCausalLM

    with skip_weights_download():
        model = AutoModelForCausalLM.from_pretrained(model_stub)

    with contextlib.ExitStack() as stack:
        stack.enter_context(calibration_forward_context(model))
        stack.enter_context(moe_calibration_context(model, calibrate_all_experts=True))

        # Find a CalibrationQwen3_5MoeSparseMoeBlock layer
        moe_layer = None
        for _, module in model.named_modules():
            if isinstance(module, CalibrationQwen3_5MoeSparseMoeBlock):
                moe_layer = module
                break

        assert moe_layer is not None

        num_experts = len(moe_layer.experts)
        expert_triggered = [False for _ in range(num_experts)]

        def hook_fn(i, module, input, output):
            expert_triggered[i] = True

        for i, expert in enumerate(moe_layer.experts):
            expert.register_forward_hook(partial(hook_fn, i))

        hidden_dim = model.config.hidden_size
        model_dtype = next(model.parameters()).dtype
        model_device = next(model.parameters()).device
        batch, seq_len = 4, 32
        sample = torch.randn(
            batch, seq_len, hidden_dim, dtype=model_dtype, device=model_device
        )

        with torch.no_grad():
            _ = moe_layer(sample)

        assert all(
            expert_triggered
        ), f"Not all experts were triggered: {expert_triggered}"


@requires_gpu
def test_calib_qwen3_5_moe_all_experts_triggered():
    config = _tiny_config()
    with torch.device("cuda"):
        original = Qwen3_5MoeSparseMoeBlock(config)
        for param in original.parameters():
            param.data.normal_(mean=0.0, std=0.02)

    module = CalibrationQwen3_5MoeSparseMoeBlock(
        original, config, calibrate_all_experts=True
    )

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
def test_calib_qwen3_5_moe_output_matches():
    config = _tiny_config()
    with torch.device("cuda"):
        original = Qwen3_5MoeSparseMoeBlock(config)
        for param in original.parameters():
            param.data.normal_(mean=0.0, std=0.02)

    hidden_dim = config.hidden_size
    batch, seq_len = 4, 32
    sample = torch.randn(batch, seq_len, hidden_dim, device="cuda")

    with calibration_forward_context(original):
        true_out = original(sample)

    module = CalibrationQwen3_5MoeSparseMoeBlock(
        original, config, calibrate_all_experts=True
    )
    with calibration_forward_context(module):
        out = module(sample)
        assert torch.nn.functional.mse_loss(true_out, out) < 1e-10

    module = CalibrationQwen3_5MoeSparseMoeBlock(
        original, config, calibrate_all_experts=False
    )
    with calibration_forward_context(module):
        out = module(sample)
        assert torch.nn.functional.mse_loss(true_out, out) < 1e-10


@requires_gpu
def test_calib_qwen3_5_moe_experts_are_linear():
    """Verify that after unpacking, experts contain nn.Linear modules
    visible to named_modules(), which is the whole point of this fix."""
    config = _tiny_config()
    with torch.device("cuda"):
        original = Qwen3_5MoeSparseMoeBlock(config)

    module = CalibrationQwen3_5MoeSparseMoeBlock(
        original, config, calibrate_all_experts=True
    )

    linear_names = [
        name for name, mod in module.named_modules() if isinstance(mod, torch.nn.Linear)
    ]
    # Each expert has gate_proj, up_proj, down_proj = 3 Linear per expert
    expected_expert_linears = config.num_experts * 3
    assert len(linear_names) >= expected_expert_linears, (
        f"Expected at least {expected_expert_linears} Linear modules, "
        f"found {len(linear_names)}: {linear_names}"
    )
