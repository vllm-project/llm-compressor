from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from types import SimpleNamespace

import torch

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


def test_layerwise_pipeline_decompresses_and_compresses_current_subgraph(
    monkeypatch,
):
    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    module = model[0]
    events = []

    class FakeActivations:
        def iter_prefetch(self, input_names):
            yield {}

        def update(self, batch_idx, outputs):
            pass

        def delete(self, batch_idx, consumed_names):
            pass

    class FakeSubgraph:
        input_names = []
        consumed_names = []

        def submodule_dict(self, model):
            return {"0": model[0]}

        def submodules(self, model):
            return [model[0]]

        def forward(self, model, **inputs):
            events.append("forward")
            return {}

    class FakeModifier:
        def start_layerwise_calibration(self, model, modules):
            events.append(("start_layerwise_calibration", modules))

    subgraph = FakeSubgraph()
    modifier = FakeModifier()
    session = SimpleNamespace(
        lifecycle=SimpleNamespace(recipe=SimpleNamespace(modifiers=[modifier])),
        state=SimpleNamespace(),
    )
    dataset_args = SimpleNamespace(
        layerwise_decompression=True,
        layerwise_compression=True,
        moe_lazy_linearization_and_repack=True,
        repack_moe_layers=True,
        propagate_error=False,
        sequential_offload_device="cpu",
        sequential_targets=None,
        sequential_targets_per_subgraph=1,
        tracing_ignore=[],
        use_loss_mask=False,
        stage_weights_in_pinned_memory=False,
        log_sequential_error=False,
    )

    monkeypatch.setattr(pipeline, "active_session", lambda: session)
    monkeypatch.setattr(pipeline, "get_main_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(pipeline, "set_onload_device", lambda *args: None)
    monkeypatch.setattr(pipeline, "find_modules_outside_subgraphs", lambda *args: [])
    monkeypatch.setattr(pipeline, "infer_sequential_targets", lambda *args: [])
    monkeypatch.setattr(pipeline, "trace_subgraphs", lambda *args: [subgraph])
    monkeypatch.setattr(
        pipeline.IntermediatesCache,
        "from_dataloader",
        lambda *args: FakeActivations(),
    )
    monkeypatch.setattr(
        pipeline,
        "calibration_forward_context",
        lambda model: nullcontext(),
    )
    monkeypatch.setattr(
        pipeline, "DisableQuantization", lambda model: nullcontext()
    )
    monkeypatch.setattr(
        pipeline, "subgraph_onload_modules", lambda modules: {}
    )
    monkeypatch.setattr(
        pipeline, "subgraph_offload_modules", lambda modules, kwargs: None
    )
    monkeypatch.setattr(
        pipeline, "subgraph_stage_modules", lambda modules, pin_memory: None
    )
    monkeypatch.setattr(pipeline, "linearize_moe", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "repack_moe", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        pipeline.LifecycleCallbacks,
        "calibration_start",
        lambda: events.append("calibration_start"),
    )
    monkeypatch.setattr(
        pipeline.LifecycleCallbacks,
        "sequential_epoch_end",
        lambda modules: events.append(("sequential_epoch_end", modules)),
    )
    monkeypatch.setattr(
        pipeline.LifecycleCallbacks,
        "calibration_end",
        lambda: events.append("calibration_end"),
    )
    monkeypatch.setattr(
        pipeline, "is_module_quantized", lambda candidate: candidate is module
    )
    monkeypatch.setattr(
        pipeline,
        "decompress_module",
        lambda candidate, leave_decompressed=True: events.append(
            ("decompress_module", candidate, leave_decompressed)
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "compress_module",
        lambda candidate: events.append(("compress_module", candidate)),
    )

    pipeline.SequentialPipeline.__call__(model, [{}], dataset_args)

    assert events == [
        "calibration_start",
        ("decompress_module", module, False),
        ("start_layerwise_calibration", [module]),
        "forward",
        ("sequential_epoch_end", [module]),
        ("compress_module", module),
        "calibration_end",
    ]
