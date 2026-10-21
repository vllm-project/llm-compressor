# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Offline official GLM fixture with real calibration pipelines and export."""

import json
import os
from pathlib import Path

import pytest
import torch
from compressed_tensors.quantization import QuantizationStatus
from compressed_tensors.quantization.lifecycle.forward import fake_quantize
from compressed_tensors.quantization.utils import compute_dynamic_scales_and_zp
from compressed_tensors.utils import patch_attr
from safetensors import safe_open
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from torch.utils.data import DataLoader
from transformers import (
    CompressedTensorsConfig,
    GlmMoeDsaConfig,
    GlmMoeDsaForCausalLM,
    PreTrainedTokenizerFast,
)
from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa

from llmcompressor import oneshot
from llmcompressor.core import State
from llmcompressor.modeling.moe.linear_experts import LinearExperts2D
from llmcompressor.modeling.moe.linearize import repack_moe
from llmcompressor.modifiers.transform import FlexSmoothModifier, QuaRotModifier
from llmcompressor.observers import MinMaxObserver
from llmcompressor.recipe import Recipe
from llmcompressor.utils.helpers import DisableQuantization
from tests.flex_smooth.oneshot_trace import OneshotTrace, inspect_fine_partitions


@pytest.mark.parametrize(
    "pipeline",
    ["basic", "sequential", "default", "split-rejected", "fine", "fine-routed"],
)
@pytest.mark.parametrize(
    "mixed,dtype",
    [(False, torch.float32), (True, torch.float32), (True, torch.bfloat16)],
)
@torch.no_grad()
def test_official_glm_calibration_export(tmp_path, monkeypatch, pipeline, mixed, dtype):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    config = GlmMoeDsaConfig(
        vocab_size=96,
        hidden_size=64,
        intermediate_size=96,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        q_lora_rank=32,
        kv_lora_rank=32,
        qk_nope_head_dim=32,
        qk_rope_head_dim=32,
        v_head_dim=32,
        first_k_dense_replace=1,
        n_routed_experts=2,
        n_shared_experts=1,
        num_experts_per_tok=1,
        n_group=1,
        topk_group=1,
        index_topk=2,
        index_head_dim=64,
        index_n_heads=2,
        indexer_types=["full", "shared"],
        max_position_embeddings=64,
        use_cache=False,
        attn_implementation="eager",
    )
    with torch.random.fork_rng():
        torch.manual_seed(42)
        model = GlmMoeDsaForCausalLM(config).to(dtype).eval()
    for module in model.modules():
        if type(module).__name__.endswith("RMSNorm"):
            module.weight.copy_(torch.linspace(0.6, 1.4, module.weight.numel()))
    if pipeline == "fine-routed":
        # Force an unhit expert, whose weight scales/export must still be complete.
        model.model.layers[1].mlp.gate.e_score_correction_bias.copy_(
            torch.tensor([1000.0, -1000.0])
        )
    config.dtype = dtype
    config.save_pretrained(tmp_path / "input-config")
    config._name_or_path = str(tmp_path / "input-config")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({"[UNK]": 0, "[PAD]": 1}, unk_token="[UNK]")
        ),
        unk_token="[UNK]",
        pad_token="[PAD]",
    )
    tokens = torch.tensor([[1, 7, 4, 5, 9], [9, 3, 5, 1, 7]])
    before = model(tokens).logits
    loader = DataLoader(
        [{"input_ids": row, "attention_mask": torch.ones_like(row)} for row in tokens],
        batch_size=1,
    )
    modifier = FlexSmoothModifier(max_tokens=32)
    recipe = [QuaRotModifier(block_size=32), modifier]
    if mixed:
        recipe = Recipe.create_instance(
            str(
                Path(__file__).resolve().parents[2]
                / "examples/glm52_precision/mixed_mxfp.yaml"
            )
        ).modifiers
        modifier = recipe[1]
    trace = OneshotTrace(monkeypatch) if mixed else None
    pipeline_args = {} if pipeline == "default" else {"pipeline": pipeline}
    targets = ["GlmMoeDsaDecoderLayer"]
    if pipeline == "split-rejected":
        pipeline_args = {"pipeline": "sequential", "sequential_targets_per_subgraph": 1}
        targets = ["GlmMoeDsaAttention", "ExpertMLP"]
    if pipeline.startswith("fine"):
        pipeline_args = {"pipeline": "sequential", "sequential_targets_per_subgraph": 1}
        targets = [r"re:.*\.input_layernorm$", "ExpertMLP", "GlmMoeDsaMLP"]
    partitions = (
        inspect_fine_partitions(monkeypatch) if pipeline.startswith("fine") else None
    )
    arguments = dict(
        model=model,
        processor=tokenizer,
        dataset=loader,
        recipe=recipe,
        moe_calibrate_all_experts=pipeline != "fine-routed",
        sequential_targets=targets,
        **pipeline_args,
    )
    if pipeline == "split-rejected":
        # Upstream GPTQ's attention/expert partition is not automatically valid
        # for norm-linear transforms: cached norm output crosses its boundary.
        with pytest.raises(ValueError, match="Sequential partition splits FlexSmooth"):
            oneshot(**arguments)
        assert model.config.flex_smooth_config["status"] == "failed"
        assert not modifier._hooks and not modifier._cache
        return
    expert_calls = {0: [], 1: []}
    handles = []
    if pipeline == "fine-routed":
        # Linearize before registering hooks; oneshot uses these same experts.
        from llmcompressor.modeling.moe.linearize import linearize_moe

        linearize_moe(model)
        for index in range(config.n_routed_experts):
            expert = model.model.layers[1].mlp.experts[index]
            handles.append(
                expert.register_forward_pre_hook(
                    lambda module, args, index=index: expert_calls[index].append(
                        args[0].shape[0]
                    )
                    if isinstance(args[0], torch.Tensor)
                    else None
                )
            )
    try:
        oneshot(**arguments)
    finally:
        for handle in handles:
            handle.remove()
    if pipeline == "fine-routed":
        assert expert_calls[0] and max(expert_calls[0]) > 0
        assert expert_calls[1] and set(expert_calls[1]) == {0}
        partitions["expert_input_token_counts"] = expert_calls
    ordering = trace.verify(pipeline) if trace else None
    if mixed:
        _check_mixed_quantization(model, recipe[-1], f"{pipeline}-{dtype}")
    with DisableQuantization(model):
        actual = model(tokens).logits
    float_error = (actual.float() - before.float()).norm() / before.float().norm()
    if dtype == torch.float32:
        torch.testing.assert_close(actual, before, atol=2e-6, rtol=2e-5)
    else:
        # BF16 storage introduces rounding at each transform, unlike the strict
        # FP32/FP64 algebra gates. This is a bounded regression diagnostic.
        assert float_error < 0.02
        assert (actual - before).abs().max() < 0.02
    assert len(modifier.diagnostics) == 6
    assert not modifier._hooks and not modifier._cache
    assert model.config.flex_smooth_config["status"] == "applied"
    if mixed:
        report = _check_compressed_roundtrip(model, tokens, tmp_path / "mixed-output")
        report.update(
            pipeline=pipeline,
            dtype=str(dtype),
            float_transform_relative_l2=float_error.item(),
            event_ordering=ordering,
            partitions=partitions,
        )
        if output := os.environ.get("FLEXSMOOTH_REPORT_DIR"):
            Path(output, f"compressed-{pipeline}-{dtype}.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )
        return
    repack_moe(model)
    model.save_pretrained(tmp_path / "output", save_compressed=False)
    restored = GlmMoeDsaForCausalLM.from_pretrained(
        tmp_path / "output", local_files_only=True, attn_implementation="eager"
    ).eval()
    torch.testing.assert_close(restored(tokens).logits, actual, atol=2e-6, rtol=2e-5)
    assert restored.config.flex_smooth_config == model.config.flex_smooth_config
    if output := os.environ.get("FLEXSMOOTH_REPORT_DIR"):
        report = {
            "fixture_seed": 42,
            "pipeline": pipeline,
            "real_weights": False,
            "logits_max_abs_error": (actual - before).abs().max().item(),
            "reload_logits_max_abs_error": (restored(tokens).logits - actual)
            .abs()
            .max()
            .item(),
            "results": model.config.flex_smooth_config["results"],
        }
        Path(output, f"official-glm-{pipeline}.json").write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
        )


class _LinearizedGlm(GlmMoeDsaForCausalLM):
    """Construct explicit experts before the real HF quantizer loads their weights.

    Avoid Transformers' global patch scan of unrelated optional vision modules.
    This fixture tests compressed IO, not the load_quantizable_moe context manager.
    """

    def __init__(self, config):
        experts = LinearExperts2D.get_linear_experts_cls(
            modeling_glm_moe_dsa.GlmMoeDsaExperts
        )
        with patch_attr(modeling_glm_moe_dsa, "GlmMoeDsaExperts", experts):
            super().__init__(config)


def _check_compressed_roundtrip(model, tokens, output):
    expected = model(tokens).logits
    assert torch.isfinite(expected).all()
    snapshots = {}
    for name, module in model.named_modules():
        if scheme := getattr(module, "quantization_scheme", None):
            snapshots[name] = {
                "weight": fake_quantize(
                    module.weight, module.weight_scale, None, scheme.weights
                ).clone(),
                "scale": module.weight_scale.clone(),
                "scheme": scheme.model_dump(exclude={"format"}),
                "bits": scheme.weights.num_bits,
            }
    model.save_pretrained(output, save_compressed=True)
    saved_config = json.loads((output / "config.json").read_text())
    assert saved_config["quantization_config"]["quantization_status"] == "compressed"
    with safe_open(
        output / "model.safetensors", framework="pt", device="cpu"
    ) as checkpoint:
        for name, snapshot in snapshots.items():
            weight_key = "weight_packed" if snapshot["bits"] == 4 else "weight"
            packed = checkpoint.get_tensor(f"{name}.{weight_key}")
            assert packed.dtype == (
                torch.uint8 if snapshot["bits"] == 4 else torch.float8_e4m3fn
            )
            expected_shape = list(snapshot["weight"].shape)
            if snapshot["bits"] == 4:
                expected_shape[-1] //= 2
            assert list(packed.shape) == expected_shape
            encoded_scale = checkpoint.get_tensor(f"{name}.weight_scale")
            assert encoded_scale.dtype == torch.uint8
            torch.testing.assert_close(
                encoded_scale,
                (snapshot["scale"].float().log2() + 127).to(torch.uint8),
                atol=0,
                rtol=0,
            )
    restored, loading = _LinearizedGlm.from_pretrained(
        output,
        local_files_only=True,
        attn_implementation="eager",
        dtype=model.dtype,
        quantization_config=CompressedTensorsConfig(dequantize=True),
        output_loading_info=True,
    )
    assert not any(loading.values()), loading
    raw_weight_dtypes = sorted(
        {str(restored.get_submodule(name).weight.dtype) for name in snapshots}
    )
    # CT MX decompressors currently materialize BF16 weights even for dtype=FP32.
    # Normalize explicitly before executing the FP32 diagnostic fixture.
    restored.to(dtype=model.dtype).eval()
    for name, snapshot in snapshots.items():
        module = restored.get_submodule(name)
        torch.testing.assert_close(module.weight, snapshot["weight"], atol=0, rtol=0)
        torch.testing.assert_close(
            module.weight_scale, snapshot["scale"], atol=0, rtol=0
        )
        assert (
            module.quantization_scheme.model_dump(exclude={"format"})
            == snapshot["scheme"]
        )
        assert module.quantization_scheme.format == (
            "mxfp4-pack-quantized" if snapshot["bits"] == 4 else "mxfp8-quantized"
        )
        assert module.quantization_scheme.input_activations.dynamic
    actual = restored(tokens).logits
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert restored.config.flex_smooth_config == model.config.flex_smooth_config
    assert restored.config.quarot_config == model.config.quarot_config
    for modifier in (QuaRotModifier(block_size=32), FlexSmoothModifier()):
        with pytest.raises(ValueError, match="already has"):
            modifier.initialize(State(model=restored))
    return {
        "real_weights": False,
        "loader": "scoped explicit GLM experts + HF quantizer",
        "raw_decompressed_weight_dtypes": raw_weight_dtypes,
        "explicit_dtype_normalization": str(model.dtype),
        "quantized_modules": len(snapshots),
        "weights_and_scales_exact": True,
        "reload_logits_max_abs_error": (actual - expected).abs().max().item(),
        "packed_bytes": sum(p.stat().st_size for p in output.glob("*.safetensors")),
    }


def _check_mixed_quantization(model, modifier, pipeline):
    checked = {}
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        leaf = name.rsplit(".", 1)[-1]
        expected = None
        if ".self_attn." in name and leaf in {
            "q_a_proj",
            "q_b_proj",
            "kv_a_proj_with_mqa",
            "o_proj",
            "wq_b",
        }:
            expected = 8
        if ".mlp." in name and leaf in {"gate_proj", "up_proj", "down_proj"}:
            expected = 4 if ".experts." in name else 8
        scheme = getattr(module, "quantization_scheme", None)
        if expected is None:
            assert scheme is None, name
            continue
        assert scheme is not None, name
        assert scheme.weights.num_bits == expected
        assert scheme.input_activations.num_bits == expected
        assert scheme.weights.group_size == scheme.input_activations.group_size == 32
        assert not scheme.weights.dynamic and scheme.input_activations.dynamic
        assert module.quantization_status == QuantizationStatus.FROZEN
        assert not hasattr(module, "weight_observer")
        assert not hasattr(module, "input_scale")
        # Re-observe the final transformed weights: stale pre-transform scales fail.
        observer = MinMaxObserver("weight", scheme.weights.model_copy(deep=True))
        expected_scale = observer(module.weight).get_qparams()["scale"]
        torch.testing.assert_close(module.weight_scale, expected_scale, atol=0, rtol=0)
        # Check actual wrapped execution against an explicit dynamic input Q/DQ path.
        x = torch.linspace(-2, 2, module.in_features, dtype=module.weight.dtype).repeat(
            2, 1
        )
        x[1] *= 8
        scale, zp = compute_dynamic_scales_and_zp(x, scheme.input_activations, module)
        assert not torch.equal(scale[0], scale[1])
        xq = fake_quantize(x, scale, zp, scheme.input_activations)
        wq = fake_quantize(module.weight, module.weight_scale, None, scheme.weights)
        torch.testing.assert_close(
            module(x), torch.nn.functional.linear(xq, wq, module.bias)
        )
        checked[name] = expected
    assert set(checked.values()) == {4, 8}
    assert any("shared_experts" in name for name in checked)
    assert any("indexer.wq_b" in name for name in checked)
    assert not modifier._calibration_hooks
    if output := os.environ.get("FLEXSMOOTH_REPORT_DIR"):
        Path(output, f"mixed-mxfp-{pipeline}.json").write_text(
            json.dumps(
                {
                    "real_weights": False,
                    "pipeline": pipeline,
                    "targets": checked,
                    "post_transform_weight_scales_exact": True,
                    "dynamic_input_execution_checked": True,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
