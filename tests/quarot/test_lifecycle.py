# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Exercise public LLMC lifecycle, serialization and real offload caches on CPU."""

from copy import deepcopy

import pytest
import torch
from compressed_tensors.offload import OffloadCache, offload_module

from llmcompressor.core import EventType, State
from llmcompressor.core.lifecycle import CompressionLifecycle
from llmcompressor.modifiers.transform import QuaRotModifier
from llmcompressor.recipe import Recipe

from .tiny_glm import TinyGLM


@pytest.fixture
def model():
    with torch.random.fork_rng():
        torch.manual_seed(42)
        return TinyGLM().double().eval()


def snapshot(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def assert_identical(model, expected):
    actual = model.state_dict()
    assert actual.keys() == expected.keys()
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)


def start(model, **options):
    lifecycle = CompressionLifecycle()
    lifecycle.initialize(model=model, recipe=[QuaRotModifier(**options)])
    lifecycle.event(EventType.CALIBRATION_START)
    return lifecycle


@torch.no_grad()
def test_recipe_lifecycle_is_equivalent_and_applies_once(model):
    tokens = torch.tensor([[1, 5, 9, 3]])
    expected = model(tokens)[0]
    before = snapshot(model)
    recipe = Recipe.create_instance(
        [QuaRotModifier(block_size=32, precision="float64")]
    )
    serialized = recipe.yaml()
    assert "QuaRotModifier" in serialized
    assert "_matrices" not in serialized
    lifecycle = CompressionLifecycle()
    lifecycle.initialize(model=model, recipe=serialized)
    modifier = lifecycle.recipe.modifiers[0]
    assert_identical(model, before)
    lifecycle.event(EventType.CALIBRATION_START)
    torch.testing.assert_close(model(tokens)[0], expected, atol=2e-12, rtol=2e-12)
    assert model.config.quarot_config["status"] == "applied"
    assert modifier._matrices == {}
    transformed = snapshot(model)
    lifecycle.event(EventType.CALIBRATION_START)
    assert_identical(model, transformed)
    lifecycle.event(EventType.SEQUENTIAL_EPOCH_END, modules=list(model.modules()))
    lifecycle.event(EventType.CALIBRATION_END)
    lifecycle.finalize()
    assert modifier.finalized and modifier._plan is None
    with pytest.raises(ValueError, match="after finalizing"):
        lifecycle.event(EventType.CALIBRATION_START)
    with pytest.raises(ValueError, match="already has QuaRot"):
        QuaRotModifier().initialize(State(model=model))


@pytest.mark.parametrize(
    "invalid", ["block", "missing", "meta", "shared", "views", "quantized"]
)
def test_preflight_rejects_invalid_models_without_mutation(model, invalid):
    options = {}
    if invalid == "block":
        options["block_size"] = 128
    elif invalid == "missing":
        del model.model.layers[0].self_attn.indexer.wq_b
    elif invalid == "meta":
        model.lm_head.to("meta")
    elif invalid == "shared":
        mlp = model.model.layers[0].mlp
        mlp.up_proj.weight = mlp.gate_proj.weight
    elif invalid == "views":
        mlp = model.model.layers[0].mlp
        mlp.up_proj.weight = torch.nn.Parameter(mlp.gate_proj.weight.detach())
    else:
        model.lm_head.quantization_status = "frozen"
    before = snapshot(model)
    with pytest.raises(ValueError):
        QuaRotModifier(**options).initialize(State(model=model))
    assert getattr(model.config, "quarot_config", None) is None
    if invalid != "meta":
        assert_identical(model, before)


def test_model_replacement_after_initialize_is_rejected(model):
    lifecycle = CompressionLifecycle()
    lifecycle.initialize(model=model, recipe=[QuaRotModifier()])
    lifecycle.state.model = deepcopy(model)
    with pytest.raises(ValueError, match="model changed"):
        lifecycle.event(EventType.CALIBRATION_START)


@pytest.mark.parametrize("offload", [False, True])
@torch.no_grad()
def test_tied_embeddings_are_untied_before_fusion(model, offload):
    model.lm_head.weight = model.model.embed_tokens.weight
    model.config.tie_word_embeddings = True
    tokens = torch.tensor([[1, 2, 7, 5]])
    expected = model(tokens)[0]
    if offload:
        for module in (model.model.embed_tokens, model.lm_head):
            offload_module(module, onload_device="cpu", offload_device="cpu")
    lifecycle = start(model)
    assert model.lm_head.weight is not model.model.embed_tokens.weight
    assert not model.config.tie_word_embeddings
    torch.testing.assert_close(model(tokens)[0], expected, atol=2e-11, rtol=2e-11)
    lifecycle.finalize()


def test_shared_disk_cache_requires_prior_untying(model, tmp_path):
    embedding, head = model.model.embed_tokens, model.lm_head
    offload_module(
        embedding, onload_device="cpu", offload_device="disk", offload_dir=tmp_path
    )
    with OffloadCache.disable_onloading():
        head.weight = embedding.weight
    offload_module(
        head, onload_device="cpu", offload_device="disk", offload_dir=tmp_path
    )
    with pytest.raises(ValueError, match="before disk offloading"):
        QuaRotModifier().initialize(State(model=model))
    assert getattr(model.config, "quarot_config", None) is None


@pytest.mark.parametrize("device", ["cpu", "disk"])
@torch.no_grad()
def test_updates_persist_through_real_offload_cache(model, tmp_path, device):
    expected = deepcopy(model)
    start(expected).finalize()
    offloaded = []
    for module in model.modules():
        if list(module.parameters(recurse=False)):
            offload_module(
                module,
                onload_device="cpu",
                offload_device=device,
                **({"offload_dir": tmp_path} if device == "disk" else {}),
            )
            offloaded.append(module)
    lifecycle = start(model)
    for module in offloaded:
        assert isinstance(module._parameters, OffloadCache)
    assert_identical(model, snapshot(expected))
    tokens = torch.tensor([[1, 3, 5, 2]])
    torch.testing.assert_close(model(tokens)[0], expected(tokens)[0], atol=0, rtol=0)
    lifecycle.finalize()


def test_runtime_failure_marks_model_unsafe_to_retry(model, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("simulated offload write failure")

    monkeypatch.setattr(QuaRotModifier, "_rotate", fail)
    with pytest.raises(OSError, match="simulated"):
        start(model)
    assert model.config.quarot_config["status"] == "failed"
    with pytest.raises(ValueError, match="failed transform cannot be retried"):
        QuaRotModifier().initialize(State(model=model))


@pytest.mark.parametrize(
    "options", [{"block_size": 3}, {"precision": "float16"}, {"seed": -1}]
)
def test_invalid_recipe_options(options):
    with pytest.raises(ValueError):
        QuaRotModifier(**options)
