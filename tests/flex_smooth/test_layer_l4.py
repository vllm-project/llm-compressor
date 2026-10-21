# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Checkpoint-free tests of the exact complete-layer server handoff."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from transformers import GlmMoeDsaConfig, GlmMoeDsaForCausalLM
from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa as glm

from llmcompressor.modeling.moe.linear_experts import LinearExperts2D
from tools.glm52_layer_l4 import (
    forward_kwargs,
    intermediate_algebra_status,
    layer_plan,
    load_layer,
    read_cache,
    run_validation,
    validate_values,
)
from tools.glm52_layer_l4_capture import capture_layer, capture_prefill


@pytest.fixture
def fixture(tmp_path):
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
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng():
        torch.manual_seed(42)
        model = GlmMoeDsaForCausalLM(config).eval()
        x = torch.randn(1, 5, config.hidden_size)
    with torch.no_grad():
        expert_type = LinearExperts2D.get_linear_experts_cls(glm.GlmMoeDsaExperts)
        model.model.layers[1].mlp.experts = expert_type.from_experts_module(
            model.model.layers[1].mlp.experts, config
        ).eval()
        for module in model.modules():
            if type(module).__name__.endswith("RMSNorm"):
                module.weight.copy_(torch.linspace(0.6, 1.4, module.weight.numel()))
    root = tmp_path / "checkpoint"
    root.mkdir()
    config.save_pretrained(root)
    save_file(model.state_dict(), str(root / "model.safetensors"))
    positions = torch.arange(5).unsqueeze(0)
    cos, sin = model.model.rotary_emb(x, positions)
    mask = torch.full((1, 1, 5, 5), torch.finfo(torch.float32).min).triu(1)
    values = dict(
        hidden_states=x, attention_mask=mask, position_ids=positions, cos=cos, sin=sin
    )
    return model, root, values


def reference_root():
    return Path(
        os.environ.get(
            "MODELSLIM_SOURCE",
            str(Path(__file__).resolve().parents[4] / "reference/msmodelslim"),
        )
    )


@pytest.mark.parametrize("layer", [0, 1])
@torch.no_grad()
def test_complete_layer_capture_replay_and_source_composition(fixture, tmp_path, layer):
    model, root, values = fixture
    if layer == 1:
        x, indices = model.model.layers[0](
            values["hidden_states"], **forward_kwargs(values)
        )
        values = {**values, "hidden_states": x, "prev_topk_indices": indices}
    cache = tmp_path / "full.safetensors"
    with capture_layer(
        model,
        model_dir=root,
        layer=layer,
        output=cache,
        input_kind="synthetic",
        input_description="five-token seeded CPU prefill",
    ):
        expected = model.model.layers[layer](
            values["hidden_states"], **forward_kwargs(values)
        )[0]
    cfg, plan = layer_plan(root, layer)
    actual, hashes = load_layer(root, cfg, plan)
    inputs, metadata = read_cache(cache, cfg, plan, hashes)
    torch.testing.assert_close(actual(inputs)[0], expected, rtol=0, atol=0)
    assert metadata["input_kind"] == "synthetic"
    result = run_validation(actual, inputs, reference_root(), max_tokens=4)
    if report_dir := os.environ.get("FLEXSMOOTH_REPORT_DIR"):
        Path(report_dir, f"complete-layer-{layer}-synthetic.json").write_text(
            json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
        )
    assert result["numerical_passed"], json.dumps(result["checks"], indent=2)
    assert result["indexer_selective"]
    assert all(item["tokens_used"] == 4 for item in result["calibration"].values())
    assert any(
        item["baseline_sha256"] != item["post_quarot_sha256"]
        for item in result["calibration"].values()
    )
    if layer == 1:
        assert result["routed_experts_exercised"]
        assert "mlp.gate.e_score_correction_bias" in hashes
        assert any("experts" in name for name in hashes)
    bad = dict(hashes)
    bad[next(iter(bad))] = "wrong"
    with pytest.raises(ValueError, match="provenance"):
        read_cache(cache, cfg, plan, bad)
    assert not model.model.layers[layer]._forward_hooks
    assert not model.model.layers[layer]._forward_pre_hooks



def _comparison(passed):
    return {
        "passed": passed,
        "max_abs": 0.0 if passed else 1.0,
        "relative_l2": 0.0 if passed else 1.0,
        "nonfinite_actual": 0,
        "nonfinite_reference": 0,
        "criterion": "test",
    }


def test_intermediate_sparse_routing_near_tie_is_diagnostic():
    checks = {
        "attention_output": _comparison(True),
        "attention_probabilities": _comparison(True),
        "router_logits": _comparison(True),
        "router_weights": _comparison(False),
        "router_indices": _comparison(False),
        "mlp_output": _comparison(False),
        "output": _comparison(False),
        "topk": _comparison(True),
    }

    status = intermediate_algebra_status(checks)

    assert status["passed"]
    assert not status["raw_passed"]
    assert status["strict_passed"]
    assert status["routing_sensitive"]
    assert status["nonblocking_failures"] == [
        "mlp_output",
        "output",
        "router_indices",
        "router_weights",
    ]


@pytest.mark.parametrize(
    "failure",
    [
        "router_logits",
        "attention_output",
        "no_route_flip",
    ],
)
def test_intermediate_sparse_routing_does_not_hide_real_failure(failure):
    checks = {
        "attention_output": _comparison(True),
        "router_logits": _comparison(True),
        "router_weights": _comparison(False),
        "router_indices": _comparison(False),
        "mlp_output": _comparison(False),
        "output": _comparison(False),
        "topk": _comparison(True),
    }

    if failure == "router_logits":
        checks["router_logits"] = _comparison(False)
    elif failure == "attention_output":
        checks["attention_output"] = _comparison(False)
    elif failure == "no_route_flip":
        checks["router_indices"] = _comparison(True)

    status = intermediate_algebra_status(checks)

    assert not status["passed"]



@pytest.mark.parametrize(
    "failure", ["too_long", "exception", "decode", "second_forward", "weights_changed"]
)
@torch.no_grad()
def test_capture_failure_cleans_hooks_and_writes_nothing(fixture, tmp_path, failure):
    model, root, values = fixture
    layer = model.model.layers[0]
    output = tmp_path / "failed.safetensors"
    with pytest.raises((ValueError, RuntimeError)):
        with capture_layer(
            model,
            model_dir=root,
            layer=0,
            output=output,
            input_kind="synthetic",
            input_description="failure test",
            max_tokens=4 if failure == "too_long" else 128,
        ):
            if failure == "exception":
                raise RuntimeError("inference failed")
            kwargs = forward_kwargs(values)
            if failure == "decode":
                kwargs["use_cache"] = True
            layer(values["hidden_states"], **kwargs)
            if failure == "second_forward":
                layer(values["hidden_states"], **kwargs)
            if failure == "weights_changed":
                layer.input_layernorm.weight.add_(1)
    assert not output.exists()
    assert not layer._forward_hooks and not layer._forward_pre_hooks


def test_reject_missing_shared_indices_and_unfamiliar_mask(fixture):
    model, root, values = fixture
    cfg, _ = layer_plan(root, 1)
    with pytest.raises(ValueError, match="prev_topk"):
        validate_values(values, cfg)
    cfg, _ = layer_plan(root, 0)
    with pytest.raises(ValueError, match="causal"):
        validate_values(
            {**values, "attention_mask": torch.zeros_like(values["attention_mask"])},
            cfg,
        )
    with pytest.raises(ValueError, match="exceed|needs"):
        layer_plan(root, 0, max_weight_gib=1e-9)


@torch.no_grad()
def test_cli_synthetic_cannot_claim_real_l4(fixture, tmp_path):
    model, root, values = fixture
    cache = tmp_path / "inputs.safetensors"
    with capture_layer(
        model,
        model_dir=root,
        layer=0,
        output=cache,
        input_kind="synthetic",
        input_description="CLI smoke",
    ):
        model.model.layers[0](values["hidden_states"], **forward_kwargs(values))
    command = [
        sys.executable,
        "tools/glm52_layer_l4.py",
        "--model-dir",
        str(root),
        "--layer",
        "0",
        "--activation-cache",
        str(cache),
        "--modelslim-source",
        str(reference_root()),
    ]
    for mode, expected in (
        (["--dry-run"], 0),
        ([], 3),
        (["--max-estimated-memory-gib", "0.00001"], 2),
    ):
        report = tmp_path / f"report-{expected}.json"
        result = subprocess.run(
            [*command, "--output", str(report), *mode], capture_output=True, text=True
        )
        assert result.returncode == expected, result.stdout + result.stderr
        content = json.loads(report.read_text(encoding="utf-8"))
        assert not content["full_layer_l4_passed"]
        if expected == 3:
            assert content["numerical_passed"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.no_grad()
def test_existing_model_prefill_stops_after_complete_layer(fixture, tmp_path, dtype):
    model, root, _ = fixture
    model.to(dtype)
    save_file(model.state_dict(), str(root / "model.safetensors"))
    tokens = torch.tensor([[1, 7, 4, 5, 9]])

    def unexpected_head(module, args):
        raise AssertionError("The helper must stop before LM head")

    handle = model.lm_head.register_forward_pre_hook(unexpected_head)
    try:
        outputs = capture_prefill(
            model,
            {"input_ids": tokens},
            model_dir=root,
            output_dir=tmp_path / "captured",
            layers=(0, 1),
            input_kind="synthetic",
            input_description="existing model prefill",
        )
    finally:
        handle.remove()
    for index in (0, 1):
        cfg, plan = layer_plan(root, index)
        replay, hashes = load_layer(root, cfg, plan)
        values, metadata = read_cache(outputs[index], cfg, plan, hashes)
        assert metadata["tokens"] == 5
        assert values["baseline_output"].shape == (1, 5, 64)
        assert not model.model.layers[index]._forward_hooks
        assert not model.model.layers[index]._forward_pre_hooks
        if dtype == torch.float32:
            torch.testing.assert_close(
                replay(values)[0], values["baseline_output"], rtol=0, atol=0
            )
        elif index == 1:
            assert run_validation(replay, values, reference_root())["numerical_passed"]


def test_live_config_mismatch_rejected(fixture, tmp_path):
    model, root, _ = fixture
    model.config.rms_norm_eps = 0.1
    with pytest.raises(ValueError, match="configs differ"):
        with capture_layer(
            model,
            model_dir=root,
            layer=0,
            output=tmp_path / "bad.safetensors",
            input_kind="synthetic",
            input_description="wrong eps",
        ):
            pass


def test_missing_expert_weight_is_not_silently_sampled(fixture):
    model, root, _ = fixture
    state = model.state_dict()
    del state["model.layers.1.mlp.experts.1.down_proj.weight"]
    save_file(state, str(root / "model.safetensors"))
    with pytest.raises(ValueError, match="Complete layer requires"):
        layer_plan(root, 1)
