"""Contract and vLLM parity tests for <Arch>."""

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
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache

from llmcompressor.modeling.<arch> import <Arch>Config, <Arch>ForCausalLM
from llmcompressor.modeling.moe.context import moe_calibration_context
from llmcompressor.modeling.moe.helpers import FusedExpertsProtocol
from llmcompressor.modeling.moe.linearize import get_linearized_moes, repack_moe
from llmcompressor.pipelines.sequential.helpers import trace_subgraphs
from llmcompressor.utils import load_context
from llmcompressor.utils.pytorch.module import get_no_split_params
from tests.testing_utils import requires_gpu

# Replace with a tokenizer ID compatible with this model's vocab_size / vLLM
TOKENIZER_ID = "<org/model-or-local-tokenizer-path>"


def _get_device() -> str:
    if torch.accelerator.is_available():
        return torch.accelerator.current_accelerator().type
    return "cpu"


def _make_tiny_config() -> <Arch>Config:
    """Build a small config representing all layer/attention/MoE variants."""
    return <Arch>Config(
        architectures=["<Arch>ForCausalLM"],
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
        dtype=torch.bfloat16,
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
    return model.to(dtype=torch.bfloat16).eval()


def _save_tiny_checkpoint(save_dir: Path) -> <Arch>Config:
    config = _make_tiny_config()
    model = _init_deterministic_model(config)
    model.save_pretrained(save_dir)
    if not TOKENIZER_ID.startswith("<"):
        AutoTokenizer.from_pretrained(TOKENIZER_ID).save_pretrained(save_dir)
    return config


@torch.no_grad()
def test_<arch>_roundtrip_and_dynamic_cache(tmp_path: Path):
    """Verify save/load round-trip, DynamicCache parity, and QuantizedKVCache."""
    _save_tiny_checkpoint(tmp_path)
    device = _get_device()

    with load_context(<Arch>ForCausalLM):
        model = <Arch>ForCausalLM.from_pretrained(
            tmp_path, dtype=torch.bfloat16, device_map=device
        ).eval()

    # 1. All attention layers must qualify for QuantizedKVCache hooking
    attn_modules = [
        m for m in model.modules() if is_cached_attention_module(m)
    ]
    assert len(attn_modules) == model.config.num_hidden_layers

    # 2. Uncached prefill must match cached prefill + decode
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
        atol=2e-2,
        rtol=2e-2,
    )

    # 3. QuantizedKVCache hooks and k_scale/v_scale attach cleanly
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
    _save_tiny_checkpoint(tmp_path)
    device = _get_device()

    plain_model = <Arch>ForCausalLM.from_pretrained(
        tmp_path, dtype=torch.bfloat16, device_map=device
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
            tmp_path, dtype=torch.bfloat16, device_map=device
        ).eval()

    assert len(get_linearized_moes(lin_model)) > 0
    lin_logits = lin_model(input_ids=input_ids, use_cache=False).logits
    with moe_calibration_context():
        calib_logits = lin_model(input_ids=input_ids, use_cache=False).logits

    assert F.mse_loss(lin_logits.float(), ref_logits.float()) < 1e-2
    assert F.mse_loss(calib_logits.float(), ref_logits.float()) < 1e-2

    repack_moe(lin_model)
    assert len(get_linearized_moes(lin_model)) == 0


def test_<arch>_sequential_tracing(tmp_path: Path):
    """Verify model can be traced into sequential subgraphs on meta/CPU."""
    _save_tiny_checkpoint(tmp_path)
    model = <Arch>ForCausalLM.from_pretrained(
        tmp_path, dtype=torch.bfloat16, device_map="cpu"
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


@requires_gpu(1)
@torch.no_grad()
def test_<arch>_vllm_parity(tmp_path: Path):
    """Verify Custom Modeling Package matches vLLM outputs on identical weights."""
    vllm = pytest.importorskip("vllm")

    if TOKENIZER_ID.startswith("<"):
        pytest.skip("Set TOKENIZER_ID to a valid tokenizer for vLLM parity")

    _save_tiny_checkpoint(tmp_path)
    prompts = [
        "The capital of France is",
        "Actions speak louder than",
    ]

    # 1. Reference outputs from vLLM (run first or free GPU memory between runs)
    llm = vllm.LLM(
        model=str(tmp_path),
        dtype="bfloat16",
        enforce_eager=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.4,
        max_model_len=128,
    )
    sampling_params = vllm.SamplingParams(
        temperature=0.0,
        max_tokens=8,
        prompt_logprobs=1,
        logprobs=1,
    )
    vllm_outputs = llm.generate(prompts, sampling_params)
    del llm
    torch.accelerator.empty_cache()

    # 2. Outputs from Custom Modeling Package
    tokenizer = AutoTokenizer.from_pretrained(tmp_path)
    device = _get_device()
    hf_model = <Arch>ForCausalLM.from_pretrained(
        tmp_path, dtype=torch.bfloat16, device_map=device
    ).eval()

    for prompt, vllm_out in zip(prompts, vllm_outputs):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Compare next-token logprobs from prefill
        hf_logits = hf_model(**inputs, use_cache=False).logits[0, -1, :].float()
        hf_logprobs = F.log_softmax(hf_logits, dim=-1)

        first_gen = vllm_out.outputs[0]
        vllm_first_token = first_gen.token_ids[0]
        vllm_first_lp = first_gen.logprobs[0][vllm_first_token].logprob

        assert hf_logprobs.argmax().item() == vllm_first_token
        assert abs(hf_logprobs[vllm_first_token].item() - vllm_first_lp) < 5e-2

        # Compare multi-step greedy decode (exercises DynamicCache)
        hf_gen = hf_model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        hf_new_tokens = hf_gen[0, inputs["input_ids"].shape[1] :].tolist()
        assert hf_new_tokens == list(first_gen.token_ids)