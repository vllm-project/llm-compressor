from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

import llmcompressor.pipelines.sequential.offload as offload
from llmcompressor.pipelines.sequential.restructure import RestructureHandler


def test_submit_subgraph_staging_runs_for_disjoint_modules(monkeypatch):
    current = torch.nn.Linear(2, 2)
    next_module = torch.nn.Linear(2, 2)
    handler = RestructureHandler(torch.nn.Module())
    calls = []

    def fake_stage(modules, restructure_handler, pin_memory=False):
        calls.append((modules, restructure_handler, pin_memory))

    monkeypatch.setattr(offload, "subgraph_stage_modules", fake_stage)

    with ThreadPoolExecutor(max_workers=1) as executor:
        result = offload._submit_subgraph_staging(
            executor,
            {"next": next_module},
            {"current": current},
            handler,
            pin_memory=True,
        )
        assert result is not None
        modules, future = result
        future.result()

    assert modules == {"next": next_module}
    assert calls == [({"next": next_module}, handler, True)]


def test_submit_subgraph_staging_skips_shared_modules(monkeypatch):
    shared = torch.nn.Linear(2, 2)
    handler = RestructureHandler(shared)
    calls = []

    monkeypatch.setattr(
        offload,
        "subgraph_stage_modules",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    with ThreadPoolExecutor(max_workers=1) as executor:
        result = offload._submit_subgraph_staging(
            executor,
            {"next": shared},
            {"current": shared},
            handler,
            pin_memory=False,
        )

    assert result is None
    assert calls == []


@pytest.mark.parametrize("requires_restructure", [False, True])
def test_submit_subgraph_staging_respects_source_tree_ownership(
    monkeypatch, requires_restructure
):
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    handler = RestructureHandler(model)
    handler.convert_model_to_meta()
    handler.apply_restructuring()
    if requires_restructure:
        # Model a conversion that consumes the entire retained source tree.
        handler.mapping[model]["restructures"].append(handler.restructures[0])
        handler.mapping[model]["source"]._modules = {
            "0": handler.mapping[model[0]]["source"]
        }
    calls = []
    monkeypatch.setattr(
        offload,
        "subgraph_stage_modules",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = offload._submit_subgraph_staging(
            executor, {"0": model[0]}, {"": model}, handler, pin_memory=False
        )
        if result is not None:
            result[1].result()

    if requires_restructure:
        assert result is None
        assert calls == []
    else:
        assert result is not None
        assert calls == [(({"0": model[0]}, handler), {"pin_memory": False})]
