# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import Counter
from copy import deepcopy

import pytest
import torch
from torch import nn

from llmcompressor.modifiers.transform.quarot.mappings import build_glm_plan

from .helpers import execute, fuse_plan, matrices_for
from .tiny_glm import TinyConfig, TinyGLM


@pytest.mark.parametrize(
    "indexers", [("full", "shared"), ("full", "full"), ("shared", "shared")]
)
def test_mapping_resolves_all_consumers_once(indexers):
    model = TinyGLM(TinyConfig(indexer_types=indexers))
    before = {name: value.clone() for name, value in model.state_dict().items()}
    plan = build_glm_plan(model)
    assert {space.name: space.size for space in plan.spaces} == {
        "rot": 64,
        "rot_b_proj": 32,
        "rot_uv": 32,
        "rot_kv_b_proj": 32,
    }
    assert len(plan.rotations) == len(set(plan.rotations))
    assert len(plan.fusions) == 9
    fused = Counter(name for fusion in plan.fusions for name in fusion.consumers)
    for name, module in model.named_modules():
        if (
            hasattr(module, "weight")
            and module.weight.ndim == 2
            and not isinstance(module, nn.Embedding)
        ):
            assert any(op.target == name for op in plan.rotations), name
            if not name.endswith(("o_proj", "down_proj")):
                assert fused[name] == 1, name
    for i, indexer in enumerate(indexers):
        targets = {
            op.target
            for op in plan.rotations
            if f"model.layers.{i}.self_attn.indexer." in op.target
        }
        assert len(targets) == (3 if indexer == "full" else 0)
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])


@pytest.mark.parametrize("broken", ["q_a_proj", "kv_b_proj", "o_proj", "indexer.wq_b"])
def test_missing_or_wrong_target_fails_before_mutation(broken):
    model = TinyGLM()
    model.model.layers[0].self_attn.set_submodule(broken, nn.Linear(7, 9, bias=False))
    before = {name: value.clone() for name, value in model.state_dict().items()}
    with pytest.raises(ValueError):
        build_glm_plan(model)
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])


def test_unsupported_mtp_and_batched_experts_fail_closed():
    model = TinyGLM()
    model.model.layers[-1].eh_proj = nn.Linear(128, 64)
    with pytest.raises(NotImplementedError, match="MTP"):
        build_glm_plan(model)
    del model.model.layers[-1].eh_proj
    model.model.layers[-1].mlp.experts = nn.Linear(64, 64)
    with pytest.raises(ValueError, match="explicit"):
        build_glm_plan(model)


def test_layernorm_is_not_rmsnorm():
    model = TinyGLM()
    model.model.layers[0].input_layernorm = nn.LayerNorm(64, bias=False)
    with pytest.raises(ValueError, match="RMSNorm"):
        build_glm_plan(model)


@pytest.mark.parametrize(
    "dtype,atol,rtol",
    [
        (torch.float64, 2e-12, 2e-12),
        (torch.float32, 4e-6, 4e-5),
        (torch.bfloat16, 0.08, 0.08),
    ],
)
@pytest.mark.parametrize("q_rank", [32, 64])
@torch.no_grad()
def test_complete_tiny_float_invariance(dtype, atol, rtol, q_rank):
    with torch.random.fork_rng():
        torch.manual_seed(42)
        model = TinyGLM(TinyConfig(q_lora_rank=q_rank)).to(dtype)
    rotated = deepcopy(model)
    plan = build_glm_plan(rotated)
    matrices = matrices_for(plan, dtype=torch.float64)
    fuse_plan(rotated, plan, torch.float64)
    execute(rotated, (plan.embedding, *plan.rotations), matrices, torch.float64)
    tokens = torch.tensor([[1, 2, 3, 4, 5], [5, 9, 3, 7, 1]])
    expected, before = model(tokens)
    actual, after = rotated(tokens)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    for left, right in zip(before, after, strict=True):
        for key in ("q", "k", "probabilities", "index_scores", "router_logits"):
            if key in left and left[key] is not None:
                torch.testing.assert_close(
                    right[key], left[key], atol=atol, rtol=rtol, msg=key
                )
        torch.testing.assert_close(
            right["hidden"],
            left["hidden"] @ matrices["rot"].to_dense().to(dtype),
            atol=atol,
            rtol=rtol,
        )
        torch.testing.assert_close(
            right["v"],
            left["v"] @ matrices["rot_uv"].to_dense().to(dtype),
            atol=atol,
            rtol=rtol,
        )


@torch.no_grad()
def test_indexer_omission_is_detectable():
    with torch.random.fork_rng():
        torch.manual_seed(42)
        model = TinyGLM().double()
    baseline = deepcopy(model)
    plan = build_glm_plan(model)
    matrices = matrices_for(plan, dtype=torch.float64)
    fuse_plan(model, plan, torch.float64)
    ops = [op for op in plan.rotations if not op.target.endswith("indexer.wq_b")]
    execute(model, (plan.embedding, *ops), matrices, torch.float64)
    tokens = torch.tensor([[1, 2, 3, 4, 5]])
    expected = baseline(tokens)[1][0]["index_scores"]
    actual = model(tokens)[1][0]["index_scores"]
    assert not torch.allclose(actual, expected, atol=1e-8, rtol=1e-8)
