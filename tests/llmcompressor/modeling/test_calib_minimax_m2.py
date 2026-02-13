import contextlib
import importlib
from functools import lru_cache, partial

import pytest
import torch
from transformers import AutoConfig

from llmcompressor.modeling.minimax_m2 import CalibrationMiniMaxM2SparseMoeBlock
from llmcompressor.modeling.moe_context import moe_calibration_context
from llmcompressor.utils.helpers import calibration_forward_context
from tests.testing_utils import requires_cadence


@lru_cache(maxsize=1)
def _load_minimax_remote_classes():
    """
    Load MiniMax M2 classes from the official HF repo via trust_remote_code.
    """
    config = AutoConfig.from_pretrained("MiniMaxAI/MiniMax-M2", trust_remote_code=True)
    modeling_module_name = config.__class__.__module__.replace(
        "configuration_minimax_m2", "modeling_minimax_m2"
    )
    modeling_module = importlib.import_module(modeling_module_name)
    return config.__class__, modeling_module.MiniMaxM2SparseMoeBlock, modeling_module.MiniMaxM2ForCausalLM


def _make_tiny_minimax_config(config_cls):
    return config_cls(
        vocab_size=256,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=64,
        num_experts_per_tok=2,
        num_local_experts=4,
        router_jitter_noise=0.0,
    )


def _assert_outputs_close(reference, candidate, atol=1e-5):
    if isinstance(reference, tuple):
        assert isinstance(candidate, tuple)
        assert len(reference) == len(candidate)
        for ref_tensor, cand_tensor in zip(reference, candidate):
            assert torch.allclose(ref_tensor, cand_tensor, atol=atol)
    else:
        assert torch.allclose(reference, candidate, atol=atol)


@requires_cadence("weekly")
def test_calib_replace_minimax_m2_all_experts():
    try:
        config_cls, _, model_cls = _load_minimax_remote_classes()
    except Exception as exc:
        pytest.skip(f"Unable to load MiniMax remote modeling: {exc}")

    model = model_cls(_make_tiny_minimax_config(config_cls)).eval()

    with contextlib.ExitStack() as stack:
        stack.enter_context(calibration_forward_context(model))
        stack.enter_context(moe_calibration_context(model, calibrate_all_experts=True))

        moe_layer = None
        for _, module in model.named_modules():
            if isinstance(module, CalibrationMiniMaxM2SparseMoeBlock):
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
        sample = torch.randn(2, 8, hidden_dim, dtype=torch.float32)

        with torch.no_grad():
            _ = moe_layer(sample)

        assert all(
            expert_triggered
        ), f"Not all experts were triggered: {expert_triggered}"


def test_calib_minimax_m2_module():
    try:
        config_cls, sparse_moe_block_cls, _ = _load_minimax_remote_classes()
    except Exception as exc:
        pytest.skip(f"Unable to load MiniMax remote modeling: {exc}")

    config = _make_tiny_minimax_config(config_cls)
    original = sparse_moe_block_cls(config).eval()

    sample = torch.randn(2, 4, config.hidden_size)

    with calibration_forward_context(original):
        true_output = original(sample)

    module = CalibrationMiniMaxM2SparseMoeBlock(
        original, config, calibrate_all_experts=True
    ).eval()
    with calibration_forward_context(module):
        output = module(sample)
    _assert_outputs_close(true_output, output)

    module = CalibrationMiniMaxM2SparseMoeBlock(
        original, config, calibrate_all_experts=False
    ).eval()
    with calibration_forward_context(module):
        output = module(sample)
    _assert_outputs_close(true_output, output)
