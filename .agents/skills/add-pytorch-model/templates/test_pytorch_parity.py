"""Contract and reference torch.nn.Module parity tests for <Arch>."""

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from compressed_tensors.modeling.kvcache import (
    QuantizedKVCache,
    initialize_hooked_kv_cache,
)
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    initialize_module_for_quantization,
    is_cached_attention_module,
)
from transformers.cache_utils import DynamicCache

from llmcompressor.modeling.<arch> import <Arch>Config, <Arch>ForCausalLM
from llmcompressor.modeling.moe.context import moe_calibration_context
from llmcompressor.modeling.moe.helpers import FusedExpertsProtocol
from llmcompressor.modeling.moe.linearize import get_linearized_moes, repack_moe
from llmcompressor.pipelines.sequential.helpers import trace_subgraphs
from llmcompressor.utils import load_context
from llmcompressor.utils.pytorch.module import get_no_split_params


def _get_device() -> str:
    if torch.accelerator.is_available():
        return torch.accelerator.current_accelerator().type
    return "cpu"


def _make_tiny_config() -> <Arch>Config:
    """Build a small config representing all layer/attention/MoE variants."""
    return <Arch>Config(
        vocab_size=1024,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        # MoE fields (remove if dense)
        moe_intermediate_size=128,
        num_experts=4,
        num_experts_per_tok=2,
        dtype=torch.float32,
    )


def _init_deterministic_model(config: <Arch>Config) -> <Arch>ForCausalLM:
    torch.manual_seed(0)
    model = <Arch>ForCausalLM(config)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "norm" in name.lower() and param.ndim == 1:
                param.fill_(1.0)
            elif "bias" in name.lower():
                param.zero_()
            else:
                torch.nn.init.normal_(param, mean=0.0, std=0.02)
    return model.eval()


def _build_reference_module(
    config: <Arch>Config, hf_model: <Arch>ForCausalLM
) -> torch.nn.Module:
    """Instantiate the original torch.nn.Module and copy weights from hf_model."""
    # Replace with actual import & weight conversion for the Source Architecture
    raise NotImplementedError(
        "Instantiate the reference torch.nn.Module and load hf_model weights"
    )


@torch.no_grad()
def test_<arch>_roundtrip_and_dynamic_cache(tmp_path: Path):
    """Verify save/load round-trip, DynamicCache parity, and QuantizedKVCache."""
    config = _make_tiny_config()
    _init_deterministic_model(config).save_pretrained(tmp_path)
    device = _get_device()

    with load_context(<Arch>ForCausalLM):
        model = <Arch>ForCausalLM.from_pretrained(
            tmp_path, dtype=torch.float32, device_map=device
        ).eval()

    attn_modules = [
        m for m in model.modules() if is_cached_attention_module(m)
    ]
    assert len(attn_modules) == model.config.num_hidden_layers

    input_ids = torch.randint(0, model.config.vocab_size, (1, 12), device=device)
    full_logits = model(input_ids=input_ids, use_cache=False).logits

    cache = DynamicCache(config=model.config)
    prefill_out = model(
        input_ids=input_ids[:, :-1], past_key_values=cache, use_cache=True
    )
    decode_out = model(
        input_ids=input_ids[:, -1:],
        past_key_values=prefill_out.past_key_values,
        use_cache=True,
    )

    assert torch.any(full_logits != 0)
    torch.testing.assert_close(
        decode_out.logits[:, -1, :],
        full_logits[:, -1, :],
        atol=1e-4,
        rtol=1e-4,
    )

    kv_scheme = QuantizationScheme(
        targets=["re:.*self_attn$"],
        input_activations=QuantizationArgs(
            num_bits=8,
            type="float",
            strategy="tensor",
            dynamic=False,
            symmetric=True,
        ),
    )
    for attn in attn_modules:
        initialize_hooked_kv_cache(model, attn)
        initialize_module_for_quantization(attn, scheme=kv_scheme)
        assert isinstance(attn.kv_cache, QuantizedKVCache)
        assert hasattr(attn, "k_scale")
        assert hasattr(attn, "v_scale")


@torch.no_grad()
def test_<arch>_moe_linearization(tmp_path: Path):
    """Verify @use_experts_implementation linearization, calibration, and repack."""
    config = _make_tiny_config()
    _init_deterministic_model(config).save_pretrained(tmp_path)
    device = _get_device()

    plain_model = <Arch>ForCausalLM.from_pretrained(
        tmp_path, dtype=torch.float32, device_map=device
    ).eval()
    fused_experts = [
        m for m in plain_model.modules() if isinstance(m, FusedExpertsProtocol)
    ]
    assert len(fused_experts) > 0, "Expected @use_experts_implementation modules"

    input_ids = torch.randint(0, plain_model.config.vocab_size, (1, 16), device=device)
    ref_logits = plain_model(input_ids=input_ids, use_cache=False).logits
    del plain_model

    with load_context(<Arch>ForCausalLM):
        lin_model = <Arch>ForCausalLM.from_pretrained(
            tmp_path, dtype=torch.float32, device_map=device
        ).eval()

    assert len(get_linearized_moes(lin_model)) > 0
    lin_logits = lin_model(input_ids=input_ids, use_cache=False).logits
    with moe_calibration_context():
        calib_logits = lin_model(input_ids=input_ids, use_cache=False).logits

    assert F.mse_loss(lin_logits, ref_logits) < 1e-6
    assert F.mse_loss(calib_logits, ref_logits) < 1e-6

    repack_moe(lin_model)
    assert len(get_linearized_moes(lin_model)) == 0


def test_<arch>_sequential_tracing(tmp_path: Path):
    """Verify model can be traced into sequential subgraphs on CPU."""
    config = _make_tiny_config()
    _init_deterministic_model(config).save_pretrained(tmp_path)
    model = <Arch>ForCausalLM.from_pretrained(
        tmp_path, dtype=torch.float32, device_map="cpu"
    ).eval()

    sample = {
        "input_ids": torch.randint(0, model.config.vocab_size, (1, 16)),
        "attention_mask": torch.ones((1, 16), dtype=torch.long),
    }
    targets = get_no_split_params(model)
    subgraphs = trace_subgraphs(
        model,
        sample,
        sequential_targets=targets,
        ignore=["lm_head"],
    )
    assert len(subgraphs) == model.config.num_hidden_layers + 1


@torch.no_grad()
def test_<arch>_pytorch_parity():
    """Verify Custom Modeling Package matches reference torch.nn.Module logits."""
    device = _get_device()
    config = _make_tiny_config()
    hf_model = _init_deterministic_model(config).to(device)
    ref_model = _build_reference_module(config, hf_model).to(device).eval()

    input_ids = torch.randint(0, config.vocab_size, (1, 12), device=device)

    # 1. Prefill parity
    hf_out = hf_model(input_ids=input_ids[:, :-1], use_cache=True)
    ref_prefill = ref_model(input_ids[:, :-1])
    ref_prefill_logits = (
        ref_prefill.logits if hasattr(ref_prefill, "logits") else ref_prefill
    )
    torch.testing.assert_close(
        hf_out.logits, ref_prefill_logits, atol=1e-4, rtol=1e-4
    )

    # 2. Next-token decode parity (exercises DynamicCache against reference)
    hf_decode = hf_model(
        input_ids=input_ids[:, -1:],
        past_key_values=hf_out.past_key_values,
        use_cache=True,
    )
    ref_full = ref_model(input_ids)
    ref_full_logits = (
        ref_full.logits if hasattr(ref_full, "logits") else ref_full
    )
    torch.testing.assert_close(
        hf_decode.logits[:, -1, :],
        ref_full_logits[:, -1, :],
        atol=1e-4,
        rtol=1e-4,
    )