import contextlib
import types
from functools import partial

import pytest
import torch
import transformers

_TRANSFORMERS_MAJOR = int(transformers.__version__.split(".")[0])

try:
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextExperts
except ImportError:
    if _TRANSFORMERS_MAJOR < 5:
        pytest.skip("gemma4 requires transformers >= 5.x", allow_module_level=True)
    raise

from llmcompressor.modeling.gemma4 import SequentialGemma4TextExperts  # noqa: E402
from llmcompressor.modeling.moe_context import moe_calibration_context  # noqa: E402
from llmcompressor.utils.dev import skip_weights_download  # noqa: E402
from llmcompressor.utils.helpers import calibration_forward_context  # noqa: E402
from tests.testing_utils import (  # noqa: E402
    requires_cadence,
    requires_gpu,
    requires_transformers_v5,
)

pytestmark = requires_transformers_v5


def _tiny_config():
    """Small text config for fast unit tests (8 experts instead of the default)."""
    return Gemma4TextConfig(
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_experts=8,
        num_experts_per_tok=2,
        num_hidden_layers=2,
    )


@requires_cadence("weekly")
@pytest.mark.parametrize("model_stub", ["google/gemma-4-26B-A4B-it"])
def test_calib_replace_gemma4_all_experts(model_stub):
    from transformers import AutoModelForCausalLM

    with skip_weights_download():
        model = AutoModelForCausalLM.from_pretrained(model_stub)

    with contextlib.ExitStack() as stack:
        stack.enter_context(calibration_forward_context(model))
        stack.enter_context(moe_calibration_context(model, calibrate_all_experts=True))

        # Find a SequentialGemma4TextExperts layer
        moe_layer = None
        for _, module in model.named_modules():
            if isinstance(module, SequentialGemma4TextExperts):
                moe_layer = module
                break

        assert moe_layer is not None

        num_experts = moe_layer.num_experts
        expert_triggered = [False for _ in range(num_experts)]

        def hook_fn(i, module, input, output):
            expert_triggered[i] = True

        # Experts are stored as numbered children (self.0, self.1, ...)
        for i in range(num_experts):
            expert = getattr(moe_layer, str(i))
            expert.register_forward_hook(partial(hook_fn, i))

        text_config = model.config.text_config
        hidden_dim = text_config.hidden_size
        num_experts_per_tok = text_config.top_k_experts
        num_tokens = 4 * 32
        model_dtype = next(model.parameters()).dtype
        model_device = next(model.parameters()).device

        sample = torch.randn(
            num_tokens, hidden_dim, dtype=model_dtype, device=model_device
        )
        top_k_index = torch.randint(
            0, num_experts, (num_tokens, num_experts_per_tok), device=model_device
        )
        top_k_weights = torch.softmax(
            torch.randn(num_tokens, num_experts_per_tok, device=model_device), dim=-1
        )

        with torch.no_grad():
            _ = moe_layer(sample, top_k_index, top_k_weights)

        assert all(
            expert_triggered
        ), f"Not all experts were triggered: {expert_triggered}"


@requires_gpu
def test_calib_gemma4_module():
    text_config = _tiny_config()
    # SequentialGemma4TextExperts expects config.text_config (full model config)
    config = types.SimpleNamespace(text_config=text_config)

    with torch.device("cuda"):
        original = Gemma4TextExperts(text_config).eval()
        for param in original.parameters():
            param.data.normal_(mean=0.0, std=0.02)

    num_experts = text_config.num_experts
    num_experts_per_tok = text_config.num_experts_per_tok
    num_tokens = 128
    hidden_dim = text_config.hidden_size

    sample = torch.randn(num_tokens, hidden_dim, device="cuda")
    top_k_index = torch.randint(
        0, num_experts, (num_tokens, num_experts_per_tok), device="cuda"
    )
    top_k_weights = torch.softmax(
        torch.randn(num_tokens, num_experts_per_tok, device="cuda"), dim=-1
    )

    with calibration_forward_context(original):
        true_output = original(sample, top_k_index, top_k_weights)

    module = SequentialGemma4TextExperts(original, config, calibrate_all_experts=True)
    with calibration_forward_context(module):
        output = module(sample, top_k_index, top_k_weights)
        assert torch.nn.functional.mse_loss(true_output, output) < 1e-10

    module = SequentialGemma4TextExperts(original, config, calibrate_all_experts=False)
    with calibration_forward_context(module):
        output = module(sample, top_k_index, top_k_weights)
        assert torch.nn.functional.mse_loss(true_output, output) < 1e-10


@requires_gpu
def test_calib_gemma4_all_experts_triggered():
    text_config = _tiny_config()
    config = types.SimpleNamespace(text_config=text_config)

    with torch.device("cuda"):
        original = Gemma4TextExperts(text_config).eval()
        for param in original.parameters():
            param.data.normal_(mean=0.0, std=0.02)

    module = SequentialGemma4TextExperts(original, config, calibrate_all_experts=True)

    num_experts = text_config.num_experts
    expert_triggered = [False for _ in range(num_experts)]

    def hook_fn(i, module, input, output):
        expert_triggered[i] = True

    for i in range(num_experts):
        expert = getattr(module, str(i))
        expert.register_forward_hook(partial(hook_fn, i))

    num_tokens = 128
    hidden_dim = text_config.hidden_size
    num_experts_per_tok = text_config.num_experts_per_tok

    sample = torch.randn(num_tokens, hidden_dim, device="cuda")
    top_k_index = torch.randint(
        0, num_experts, (num_tokens, num_experts_per_tok), device="cuda"
    )
    top_k_weights = torch.softmax(
        torch.randn(num_tokens, num_experts_per_tok, device="cuda"), dim=-1
    )

    with calibration_forward_context(module):
        with torch.no_grad():
            _ = module(sample, top_k_index, top_k_weights)

    assert all(expert_triggered), f"Not all experts were triggered: {expert_triggered}"


@requires_gpu
def test_calib_gemma4_experts_are_linear():
    """Verify that after unpacking, experts contain nn.Linear modules."""
    text_config = _tiny_config()
    config = types.SimpleNamespace(text_config=text_config)

    with torch.device("cuda"):
        original = Gemma4TextExperts(text_config).eval()

    module = SequentialGemma4TextExperts(original, config, calibrate_all_experts=True)

    linear_names = [
        name for name, mod in module.named_modules() if isinstance(mod, torch.nn.Linear)
    ]
    # Each expert has gate_proj, up_proj, down_proj = 3 Linear per expert
    expected_linears = text_config.num_experts * 3
    assert len(linear_names) >= expected_linears, (
        f"Expected at least {expected_linears} Linear modules, "
        f"found {len(linear_names)}: {linear_names}"
    )
