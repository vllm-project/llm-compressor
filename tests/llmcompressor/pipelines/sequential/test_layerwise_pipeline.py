import contextlib
from types import SimpleNamespace

import torch
from compressed_tensors.offload import set_onload_device

import llmcompressor.pipelines.sequential.helpers as helpers_module
import llmcompressor.pipelines.sequential.offload as offload_module
import llmcompressor.pipelines.sequential.pipeline as pipeline_module
from llmcompressor.pipelines.sequential.pipeline import SequentialPipeline
from llmcompressor.pipelines.sequential.restructure import RestructureHandler


class _FakeActivations:
    def iter_prefetch(self, input_names):
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

    def submodule_dict(self, model):
        return {"0": self.module}

    def forward(self, model, **inputs):
        self.events.append("forward")
        return {}


def test_sequential_pipeline_restructures_and_recompresses_current_subgraph(
    monkeypatch,
):
    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    set_onload_device(model, torch.device("cpu"))
    handler = RestructureHandler(model)
    handler.convert_model_to_meta()
    handler.apply_restructuring()
    module = model[0]
    events = []

    session = SimpleNamespace(
        lifecycle=SimpleNamespace(recipe=SimpleNamespace(modifiers=[])),
        state=SimpleNamespace(),
    )
    dataset_args = SimpleNamespace(
        layerwise_compression=True,
        propagate_error=False,
        log_sequential_error=False,
        sequential_offload_device="cpu",
        sequential_targets=None,
        sequential_targets_per_subgraph=1,
        tracing_ignore=[],
        use_loss_mask=False,
    )

    monkeypatch.setattr(pipeline_module, "active_session", lambda: session)
    monkeypatch.setattr(pipeline_module, "get_main_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(
        pipeline_module,
        "calibration_forward_context",
        lambda model: contextlib.nullcontext(),
    )
    monkeypatch.setattr(
        pipeline_module, "DisableQuantization", lambda model: contextlib.nullcontext()
    )
    monkeypatch.setattr(pipeline_module, "infer_sequential_targets", lambda *args: [])
    monkeypatch.setattr(pipeline_module, "trace_graph", lambda *args: (None, set()))
    monkeypatch.setattr(
        pipeline_module,
        "partition_subgraphs",
        lambda *args: [_FakeSubgraph(module, events)],
    )
    monkeypatch.setattr(pipeline_module, "find_modules_outside_graph", lambda *args: {})
    monkeypatch.setattr(
        pipeline_module,
        "subgraph_stage_modules",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        pipeline_module,
        "subgraph_onload_modules",
        lambda modules, handler, device: events.append("onload")
        or offload_module.subgraph_onload_modules(modules, handler, device),
    )
    monkeypatch.setattr(
        pipeline_module,
        "subgraph_offload_modules",
        lambda modules, offload_kwargs: events.append("offload"),
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
        helpers_module,
        "is_module_quantized",
        lambda candidate: candidate is module,
    )
    transfer_state = offload_module.transfer_module_state

    def materialize(target, source, device, **kwargs):
        events.append(("restructure", [target]))
        transfer_state(target, source, device, **kwargs)

    monkeypatch.setattr(offload_module, "transfer_module_state", materialize)
    monkeypatch.setattr(
        helpers_module,
        "compress_module",
        lambda candidate: events.append(("compress_module", candidate)),
    )
    monkeypatch.setattr(helpers_module, "is_distributed", lambda: False)

    SequentialPipeline.__call__(model, [{}], dataset_args, handler)

    assert events == [
        "calibration_start",
        "onload",
        ("restructure", [module]),
        "forward",
        ("sequential_epoch_end", {"0": module}),
        ("compress_module", module),
        "offload",
        "calibration_end",
    ]
