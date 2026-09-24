from concurrent.futures import ThreadPoolExecutor

import torch

from llmcompressor.args import DatasetArguments
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
        )

    assert result is None
    assert calls == []


def test_module_prefetch_is_disabled_by_default():
    assert DatasetArguments().sequential_module_prefetch is False
