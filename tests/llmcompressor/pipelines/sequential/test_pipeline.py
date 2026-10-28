from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import compressed_tensors.offload.module as offload_module
import torch

import llmcompressor.pipelines.sequential.pipeline as pipeline


def test_submit_subgraph_staging_runs_for_disjoint_modules(monkeypatch):
    current = torch.nn.Linear(2, 2)
    next_module = torch.nn.Linear(2, 2)
    calls = []

    def fake_stage(modules, pin_memory=False, **kwargs):
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


def test_subgraph_stage_modules_flushes_distributed_writes_first(monkeypatch):
    calls = []
    group = object()

    class FakeWriter:
        def flush_distributed(self, passed_group):
            assert passed_group is group
            calls.append("flush")

    monkeypatch.setattr(offload_module, "OffloadCache", dict)
    monkeypatch.setattr(offload_module, "disk_load_context", nullcontext)
    monkeypatch.setattr(
        offload_module,
        "stage_module_offload",
        lambda module, pin_memory=False: calls.append("stage"),
    )

    pipeline.subgraph_stage_modules(
        {"next": torch.nn.Linear(2, 2)},
        pin_memory=False,
        async_writer=FakeWriter(),
        distributed_group=group,
    )

    assert calls == ["flush", "stage"]
