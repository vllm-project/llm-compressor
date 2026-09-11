import contextlib
from types import SimpleNamespace

import torch

import llmcompressor.pipelines.sequential.pipeline as pipeline_module
from llmcompressor.pipelines.sequential.pipeline import SequentialPipeline


class _FakeActivations:
    def iter(self, input_names):
        yield {}

    def update(self, batch_idx, outputs):
        pass

    def delete(self, batch_idx, consumed_names):
        pass


class _FakeSubgraph:
    input_names = []
    consumed_names = []

    def __init__(self, module, events):
        self.module = module
        self.events = events

    def submodules(self, model):
        return [self.module]

    def forward(self, model, **inputs):
        self.events.append("forward")
        return {}


class _FakeModifier:
    def __init__(self, events):
        self.events = events

    def start_layerwise_calibration(self, model, modules):
        self.events.append(("start_layerwise_calibration", modules))


def test_sequential_pipeline_decompresses_and_recompresses_current_subgraph(
    monkeypatch,
):
    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    module = model[0]
    events = []

    modifier = _FakeModifier(events)
    session = SimpleNamespace(
        lifecycle=SimpleNamespace(recipe=SimpleNamespace(modifiers=[modifier])),
        state=SimpleNamespace(),
    )
    dataset_args = SimpleNamespace(
        layerwise_decompression=True,
        layerwise_compression=True,
        propagate_error=False,
        sequential_offload_device="cpu",
        sequential_targets=None,
        sequential_targets_per_subgraph=1,
        tracing_ignore=[],
        use_loss_mask=False,
    )

    monkeypatch.setattr(pipeline_module, "active_session", lambda: session)
    monkeypatch.setattr(pipeline_module, "get_main_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(pipeline_module, "set_onload_device", lambda *args: None)
    monkeypatch.setattr(pipeline_module, "disable_offloading", contextlib.nullcontext)
    monkeypatch.setattr(
        pipeline_module,
        "calibration_forward_context",
        lambda model: contextlib.nullcontext(),
    )
    monkeypatch.setattr(
        pipeline_module, "DisableQuantization", lambda model: contextlib.nullcontext()
    )
    monkeypatch.setattr(pipeline_module, "infer_sequential_targets", lambda *args: [])
    monkeypatch.setattr(
        pipeline_module,
        "trace_subgraphs",
        lambda *args: [_FakeSubgraph(module, events)],
    )
    monkeypatch.setattr(
        pipeline_module.IntermediatesCache,
        "from_dataloader",
        lambda *args: _FakeActivations(),
    )
    monkeypatch.setattr(
        pipeline_module.LifecycleCallbacks,
        "calibration_start",
        lambda: events.append("calibration_start"),
    )
    monkeypatch.setattr(
        pipeline_module.LifecycleCallbacks,
        "sequential_epoch_end",
        lambda modules: events.append(("sequential_epoch_end", modules)),
    )
    monkeypatch.setattr(
        pipeline_module.LifecycleCallbacks,
        "calibration_end",
        lambda: events.append("calibration_end"),
    )
    monkeypatch.setattr(
        pipeline_module,
        "is_module_quantized",
        lambda candidate: candidate is module,
    )
    monkeypatch.setattr(
        pipeline_module,
        "decompress_module",
        lambda candidate, leave_decompressed=True: events.append(
            ("decompress_module", candidate, leave_decompressed)
        ),
    )
    monkeypatch.setattr(
        pipeline_module,
        "compress_module",
        lambda candidate: events.append(("compress_module", candidate)),
    )

    SequentialPipeline.__call__(model, [{}], dataset_args)

    assert events == [
        "calibration_start",
        ("decompress_module", module, False),
        ("start_layerwise_calibration", [module]),
        "forward",
        ("sequential_epoch_end", [module]),
        ("compress_module", module),
        "calibration_end",
    ]
