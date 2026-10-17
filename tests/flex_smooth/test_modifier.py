# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from compressed_tensors.offload import OffloadCache, offload_module

from llmcompressor.core import EventType, State
from llmcompressor.core.lifecycle import CompressionLifecycle
from llmcompressor.modifiers.transform import FlexSmoothModifier, QuaRotModifier
from llmcompressor.modifiers.transform.flex_smooth.mappings import glm_mappings
from llmcompressor.recipe import Recipe
from tests.quarot.tiny_glm import TinyConfig, TinyGLM


@pytest.fixture
def model():
    with torch.random.fork_rng():
        torch.manual_seed(42)
        return TinyGLM().eval()


TOKENS = torch.tensor([[1, 3, 7, 9], [9, 1, 5, 2]])


@pytest.mark.parametrize("indexers", [("full", "shared"), ("full", "full")])
def test_mapping_matches_original_glm52_adapter(oracle, indexers):
    config = TinyConfig(indexer_types=indexers)
    config.num_key_value_heads = config.num_attention_heads
    model = TinyGLM(config)
    adapter = SimpleNamespace(
        config=config, _layer_has_indexer=lambda i: indexers[i] == "full"
    )
    original = [
        item
        for item in oracle.mapping(adapter)
        if item.subgraph_type in ("ov", "norm-linear")
    ]
    actual = glm_mappings(model, ("norm-linear", "ov"))
    assert {(item.kind, item.source, item.consumers) for item in actual} == {
        (item.subgraph_type, item.mapping.source, tuple(item.mapping.targets))
        for item in original
    }


@torch.no_grad()
def test_float64_algebra_and_untouched_key_rows(model):
    model.double()
    before = model(TOKENS)[0]
    original = deepcopy(model.state_dict())
    modifier = FlexSmoothModifier(alpha=0.4, beta=0.7)
    life = begin(model, modifier)
    model(TOKENS)
    life.event(EventType.CALIBRATION_END)
    life.finalize()
    torch.testing.assert_close(model(TOKENS)[0], before, atol=2e-12, rtol=2e-12)
    for i in range(2):
        name = f"model.layers.{i}.self_attn.kv_b_proj.weight"
        key = model.state_dict()[name].reshape(2, 64, -1)[:, :32]
        torch.testing.assert_close(
            key, original[name].reshape(2, 64, -1)[:, :32], atol=0, rtol=0
        )


def begin(model, modifier):
    lifecycle = CompressionLifecycle()
    lifecycle.initialize(model=model, recipe=[modifier])
    lifecycle.event(EventType.CALIBRATION_START)
    return lifecycle


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("indexers", [("full", "shared"), ("full", "full")])
@torch.no_grad()
def test_complete_modifier_against_original_source(oracle, dtype, indexers):
    with torch.random.fork_rng():
        torch.manual_seed(42)
        model = TinyGLM(TinyConfig(indexer_types=indexers)).eval().to(dtype)
    reference = deepcopy(model)
    modifier = FlexSmoothModifier()
    lifecycle = begin(model, modifier)
    expected_float = model(TOKENS)[0]
    plan = glm_mappings(reference, modifier.subgraphs)
    captured = {
        key: torch.cat(values).clone() for key, values in modifier._cache.items()
    }
    expected_scales = {}
    for item in plan:
        x = captured[item.source]
        consumers = [reference.get_submodule(name) for name in item.consumers]
        w = torch.cat([module.weight for module in consumers])
        alpha, beta, loss = (
            oracle.search.FlexSmoothAlphaBetaSearcher().search_alpha_beta(x, w)
        )
        calculator = oracle.scales.FlexSmoothScaleCalculator(alpha, beta, "max")
        w_max = oracle.scales.compute_multi_weight_scale(
            [module.weight for module in consumers], dtype
        )
        if item.kind == "norm-linear":
            scale = calculator.compute_smooth_scale(x.abs().amax(0), w_max)
            subgraph = oracle.subgraphs.NormLinearSubgraph(
                reference.get_submodule(item.source), consumers
            )
            oracle.fusion.NormLinearSubgraphFusion().apply_fusion(
                subgraph, {"scales": scale}
            )
        else:
            scale, v_scale = calculator.compute_ov_scales(
                x.abs().amax(0), w_max, item.heads, item.heads
            )
            kv = reference.get_submodule(item.source)
            view = kv.weight.reshape(item.heads, item.key_dim + item.value_dim, -1)
            virtual = torch.nn.Linear(
                kv.weight.shape[1], item.heads * item.value_dim, bias=False, dtype=dtype
            )
            virtual.weight.copy_(view[:, item.key_dim :, :].reshape_as(virtual.weight))
            subgraph = oracle.subgraphs.OVSubgraph(
                consumers[0], virtual, item.heads, item.heads
            )
            oracle.fusion.OVSubgraphFusion().apply_fusion(
                subgraph, {"o_scales": scale, "v_scales": v_scale}
            )
            view[:, item.key_dim :, :].copy_(
                virtual.weight.reshape(item.heads, item.value_dim, -1)
            )
        expected_scales[item.source] = (alpha, beta, loss, scale)
    lifecycle.event(EventType.SEQUENTIAL_EPOCH_END, modules=list(model.modules()))
    lifecycle.event(EventType.CALIBRATION_END)
    lifecycle.finalize()
    for key, (alpha, beta, loss, scale) in expected_scales.items():
        actual = modifier.diagnostics[key]
        assert (actual["alpha"], actual["beta"], actual["loss"]) == (alpha, beta, loss)
        torch.testing.assert_close(actual["scale"], scale, atol=0, rtol=0)
    max_error = 0.0
    for name, value in model.state_dict().items():
        expected = reference.state_dict()[name]
        torch.testing.assert_close(value, expected, atol=0, rtol=0, msg=name)
        max_error = max(
            max_error, (value.float() - expected.float()).abs().max().item()
        )
    torch.testing.assert_close(model(TOKENS)[0], reference(TOKENS)[0], atol=0, rtol=0)
    torch.testing.assert_close(
        model(TOKENS)[0],
        expected_float,
        atol=3e-6 if dtype == torch.float32 else 0.04,
        rtol=3e-5 if dtype == torch.float32 else 0.04,
    )
    assert not modifier._hooks and not modifier._cache
    if output := os.environ.get("FLEXSMOOTH_REPORT_DIR"):
        report = {
            "source_sha256": oracle.hashes,
            "parameter_max_abs_error": max_error,
            "dtype": str(dtype),
            "indexers": indexers,
            "diagnostics": {
                key: {**value, "scale": value["scale"].tolist()}
                for key, value in modifier.diagnostics.items()
            },
        }
        Path(
            output, f"flex-{str(dtype).split('.')[-1]}-{'-'.join(indexers)}.json"
        ).write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


@pytest.mark.parametrize(
    "subgraphs", [("ov",), ("norm-linear",), ("ov", "norm-linear")]
)
@torch.no_grad()
def test_recipe_fixed_scales_and_repeated_events(model, subgraphs):
    modifier = FlexSmoothModifier(
        alpha=0.4, beta=0.7, subgraphs=subgraphs, max_tokens=3
    )
    serialized = Recipe.create_instance([modifier]).yaml()
    life = CompressionLifecycle()
    life.initialize(model=model, recipe=serialized)
    modifier = life.recipe.modifiers[0]
    expected = model(TOKENS)[0]
    life.event(EventType.CALIBRATION_START)
    life.event(EventType.CALIBRATION_START)
    model(TOKENS)
    life.event(EventType.CALIBRATION_END)  # Basic/manual lifecycle fallback.
    first = deepcopy(model.state_dict())
    life.event(EventType.CALIBRATION_END)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, first[name], atol=0, rtol=0)
    torch.testing.assert_close(model(TOKENS)[0], expected, atol=3e-6, rtol=3e-5)
    assert all(
        row["tokens_seen"] == 8 and row["tokens_used"] == 3
        for row in modifier.diagnostics.values()
    )
    life.finalize()
    with pytest.raises(ValueError, match="already has FlexSmooth"):
        FlexSmoothModifier().initialize(State(model=model))


@pytest.mark.parametrize("device", ["cpu", "disk"])
@torch.no_grad()
def test_offload_writeback_and_cleanup(model, device, tmp_path):
    baseline = deepcopy(model)
    expected_modifier = FlexSmoothModifier(alpha=0.4, beta=0.7)
    life = begin(baseline, expected_modifier)
    baseline(TOKENS)
    life.event(EventType.CALIBRATION_END)
    life.finalize()
    for module in model.modules():
        if list(module.parameters(recurse=False)):
            offload_module(
                module,
                onload_device="cpu",
                offload_device=device,
                **({"offload_dir": tmp_path} if device == "disk" else {}),
            )
    modifier = FlexSmoothModifier(alpha=0.4, beta=0.7)
    life = begin(model, modifier)
    model(TOKENS)
    life.event(EventType.CALIBRATION_END)
    life.finalize()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, baseline.state_dict()[name], atol=0, rtol=0)
    assert isinstance(
        model.model.layers[0].self_attn.q_a_proj._parameters, OffloadCache
    )
    assert not modifier._hooks and not modifier._cache


def test_missing_inputs_and_failed_hook_clean_up(model):
    modifier = FlexSmoothModifier()
    life = begin(model, modifier)
    with pytest.raises(ValueError, match="No calibration inputs"):
        life.event(EventType.CALIBRATION_END)
    assert not modifier._hooks and not modifier._cache
    assert model.config.flex_smooth_config["status"] == "failed"


@pytest.mark.parametrize("policy", ["error", "identity"])
@torch.no_grad()
def test_all_zero_subgraph_policy(model, policy):
    modifier = FlexSmoothModifier(subgraphs=("ov",), on_degenerate=policy)
    for layer in model.model.layers:
        layer.self_attn.o_proj.weight.zero_()
    life = begin(model, modifier)
    expected = model(TOKENS)[0]
    if policy == "error":
        with pytest.raises(ValueError, match="no finite"):
            life.event(EventType.CALIBRATION_END)
        assert not modifier._hooks and not modifier._cache
    else:
        life.event(EventType.CALIBRATION_END)
        life.finalize()
        assert all(
            row["fallback"] == "no_finite_candidate"
            for row in modifier.diagnostics.values()
        )
        torch.testing.assert_close(model(TOKENS)[0], expected, atol=0, rtol=0)


@torch.no_grad()
def test_quarot_then_flexsmooth_with_sequential_layers(model):
    expected = model(TOKENS)[0]
    modifier = FlexSmoothModifier()
    life = CompressionLifecycle()
    life.initialize(model=model, recipe=[QuaRotModifier(block_size=32), modifier])
    life.event(EventType.CALIBRATION_START)
    model(TOKENS)
    for layer in model.model.layers:
        life.event(EventType.SEQUENTIAL_EPOCH_END, modules=list(layer.modules()))
    life.event(EventType.CALIBRATION_END)
    life.finalize()
    assert len(modifier.diagnostics) == 6
    torch.testing.assert_close(model(TOKENS)[0], expected, atol=4e-6, rtol=4e-5)


@pytest.mark.parametrize("broken", ["indexer", "alias", "geometry", "meta"])
def test_bad_topology_fails_before_mutation(model, broken):
    attention = model.model.layers[0].self_attn
    if broken == "indexer":
        del attention.indexer.wq_b
    elif broken == "alias":
        attention.q_a_layernorm.weight = attention.kv_a_layernorm.weight
    elif broken == "geometry":
        model.config.num_key_value_heads = 1
    else:
        attention.q_a_proj.to("meta")
    with pytest.raises(ValueError):
        FlexSmoothModifier().initialize(State(model=model))
    assert getattr(model.config, "flex_smooth_config", None) is None


@torch.no_grad()
def test_bias_fusion_preserves_norm_and_ov_outputs(model):
    for layer in model.model.layers:
        for module in (
            layer.input_layernorm,
            layer.self_attn.q_a_layernorm,
            layer.self_attn.kv_b_proj,
            layer.self_attn.o_proj,
        ):
            module.bias = torch.nn.Parameter(
                torch.linspace(-0.1, 0.1, module.weight.shape[0])
            )
    # The custom RMSNorm fixture ignores bias, so check affine LayerNorm explicitly.
    for layer in model.model.layers:
        for parent, name in (
            (layer, "input_layernorm"),
            (layer.self_attn, "q_a_layernorm"),
        ):
            old = getattr(parent, name)
            norm = torch.nn.LayerNorm(old.weight.numel())
            norm.weight.copy_(old.weight)
            norm.bias.copy_(old.bias)
            setattr(parent, name, norm)
    before = model(TOKENS)[0]
    life = begin(model, FlexSmoothModifier(alpha=0.4, beta=0.7))
    model(TOKENS)
    life.event(EventType.CALIBRATION_END)
    life.finalize()
    torch.testing.assert_close(model(TOKENS)[0], before, atol=3e-6, rtol=3e-5)


def test_nonfinite_hook_and_split_partition_release_hooks(model):
    modifier = FlexSmoothModifier()
    life = begin(model, modifier)
    with pytest.raises(ValueError, match="nonfinite calibration"):
        model.model.layers[0].self_attn.q_a_proj(torch.full((1, 2, 64), float("nan")))
    assert not modifier._hooks and not modifier._cache
    with pytest.raises(ValueError, match="Failed FlexSmooth"):
        life.event(EventType.CALIBRATION_END)


@torch.no_grad()
def test_split_partition_is_rejected(model):
    modifier = FlexSmoothModifier()
    life = begin(model, modifier)
    model(TOKENS)
    with pytest.raises(ValueError, match="partition splits"):
        life.event(
            EventType.SEQUENTIAL_EPOCH_END,
            modules=[model.model.layers[0].self_attn.o_proj],
        )
    assert not modifier._hooks and not modifier._cache


def test_model_swap_rejected(model):
    modifier = FlexSmoothModifier()
    life = begin(model, modifier)
    life.state.model = deepcopy(model)
    with pytest.raises(ValueError, match="model changed"):
        life.event(EventType.CALIBRATION_END)
    with pytest.raises(ValueError, match="before successful calibration"):
        modifier.finalize(State(model=model))
    assert not modifier._hooks and not modifier._cache


def test_nonfinite_inputs_rejected_even_after_cache_cap(model):
    modifier = FlexSmoothModifier(max_tokens=1)
    begin(model, modifier)
    module = model.model.layers[0].self_attn.q_a_proj
    module(torch.ones(1, 64))
    with pytest.raises(ValueError, match="nonfinite calibration"):
        module(torch.full((1, 64), float("nan")))
    assert not modifier._hooks and not modifier._cache


def test_pipeline_loss_mask_rejected_at_calibration_start(model):
    modifier = FlexSmoothModifier()
    life = CompressionLifecycle()
    life.initialize(model=model, recipe=[modifier])
    # BasicPipeline populates this after initialize, before calibration_start.
    life.state.loss_masks = []
    with pytest.raises(ValueError, match="loss masking"):
        life.event(EventType.CALIBRATION_START)
    assert not modifier._hooks and not modifier._cache
    assert getattr(model.config, "flex_smooth_config", None) is None
