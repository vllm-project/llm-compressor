from concurrent.futures import ThreadPoolExecutor
from threading import get_ident
from types import SimpleNamespace

import pytest
import torch
from compressed_tensors.offload import OffloadCache, offload_module

import llmcompressor.pipelines.sequential.pipeline as pipeline


def test_submit_subgraph_staging_runs_for_disjoint_modules(monkeypatch):
    current = torch.nn.Linear(2, 2)
    next_module = torch.nn.Linear(2, 2)
    calls = []

    def fake_stage(modules, pin_memory=False):
        calls.append((modules, pin_memory))

    monkeypatch.setattr(pipeline, "subgraph_stage_modules", fake_stage)

    with ThreadPoolExecutor(max_workers=1) as executor:
        result = pipeline._submit_subgraph_staging(
            executor,
            {"next": next_module},
            {"current": current},
            pin_memory=True,
        )
        assert result is not None
        modules, future = result
        future.result()

    assert modules == {"next": next_module}
    assert calls == [({"next": next_module}, True)]


def test_submit_subgraph_staging_skips_shared_modules(monkeypatch):
    shared = torch.nn.Linear(2, 2)
    calls = []

    monkeypatch.setattr(
        pipeline,
        "subgraph_stage_modules",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    with ThreadPoolExecutor(max_workers=1) as executor:
        result = pipeline._submit_subgraph_staging(
            executor,
            {"next": shared},
            {"current": shared},
            pin_memory=False,
            jobs=[lambda: calls.append("prepare")],
        )

    assert result is None
    assert calls == []


def test_preparation_runs_before_staging_on_worker(monkeypatch):
    calls = []
    main_thread = get_ident()
    artifact = object()

    def prepare():
        assert get_ident() != main_thread
        assert not torch.is_grad_enabled()
        calls.append("prepare")
        return artifact

    monkeypatch.setattr(
        pipeline,
        "subgraph_stage_modules",
        lambda *args, **kwargs: calls.append("stage"),
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        _, future = pipeline._submit_subgraph_staging(
            executor, {}, {}, False, jobs=[None, prepare]
        )
        assert future.result() == [None, artifact]
    assert calls == ["prepare", "stage"]


def test_preparation_failure_propagates_before_staging(monkeypatch):
    calls = []

    def prepare():
        raise ValueError("preparation failed")

    monkeypatch.setattr(
        pipeline,
        "subgraph_stage_modules",
        lambda *args, **kwargs: calls.append("stage"),
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        _, future = pipeline._submit_subgraph_staging(
            executor, {}, {}, False, jobs=[prepare]
        )
        with pytest.raises(ValueError, match="preparation failed"):
            future.result()
    assert calls == []


@pytest.mark.parametrize("offload_device", ["cpu", "disk"])
def test_new_children_inherit_offload_policy_and_removed_children_are_excluded(
    offload_device, tmp_path
):
    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    module = model[0]
    kwargs = {"offload_dir": str(tmp_path)} if offload_device == "disk" else {}
    offload_module(module, "cpu", offload_device, **kwargs)
    subgraph = SimpleNamespace(
        submodule_dict=lambda model: dict(module.named_modules(prefix="0"))
    )
    previous = subgraph.submodule_dict(model)
    policy = pipeline.subgraph_onload_modules(previous)

    module.transform = torch.nn.Linear(4, 4)
    module.observer = torch.nn.Identity()
    current = pipeline._onload_new_children(model, subgraph, previous, policy)
    assert set(current) == {"0", "0.transform", "0.observer"}
    assert policy["0.transform"] == policy["0"]
    assert not isinstance(module.transform._parameters, OffloadCache)
    expected = module.transform.weight.detach().clone()

    del module.observer
    # Completion callbacks may also create children after the onload snapshot.
    module.transform.child = torch.nn.Identity()
    current = subgraph.submodule_dict(model)
    pipeline.subgraph_offload_modules(
        current, pipeline._scope_offload_kwargs(current, policy)
    )
    assert "0.observer" not in current
    for child in current.values():
        assert isinstance(child._parameters, OffloadCache)
        assert str(child._parameters.offload_device) == offload_device
        if offload_device == "disk":
            assert child._parameters.offload_dir == tmp_path
    torch.testing.assert_close(module.transform.weight, expected)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sequential_transforms_offload_all_new_children(device, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from compressed_tensors.transform.factory.base import TransformBase
    from transformers import LlamaConfig, LlamaForCausalLM

    from llmcompressor.args import DatasetArguments
    from llmcompressor.core import active_session, reset_session
    from llmcompressor.modifiers.transform import QuIPModifier

    model = LlamaForCausalLM(
        LlamaConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=32,
        )
    ).eval()
    batches = [{"input_ids": torch.randint(0, 32, (1, 4))}]
    with torch.no_grad():
        expected = model(**batches[0]).logits
    modifier = QuIPModifier(precision=torch.float64)
    args = DatasetArguments(sequential_targets=["LlamaDecoderLayer"])
    monkeypatch.setattr(pipeline, "get_main_device", lambda: torch.device(device))
    reset_session()
    try:
        active_session().initialize(model=model, recipe=[modifier], calib_data=batches)
        pipeline.SequentialPipeline()(model, batches, args)
        transforms = [m for m in model.modules() if isinstance(m, TransformBase)]
        assert transforms
        for transform in transforms:
            assert isinstance(transform._parameters, OffloadCache)
            assert transform._parameters.offload_device == torch.device("cpu")
        # Factory caches outlive individual subgraphs; they must never retain GPU
        # allocations after a child is onloaded and offloaded.
        for factory in modifier._transform_factories.values():
            assert all(
                weight.device.type == "cpu" for weight in factory.weights.values()
            )
        active_session().finalize()
        assert not modifier._transform_factories
        with torch.no_grad():
            actual = model(**batches[0]).logits.cpu()
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-4)
    finally:
        reset_session()
