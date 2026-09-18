from unittest.mock import Mock

import torch

import llmcompressor.pipelines.sequential.offloading as offloading
from llmcompressor.pipelines.sequential.pipeline import _wait_for_overlapping_offloads


class _FakeOffloadCache(dict):
    pass


class _IndividualExpert(torch.nn.Module):
    def __init__(self, offset: float):
        super().__init__()
        self.offset = offset

    def forward(self, inputs):
        return inputs + self.offset


class _IndividualExpertModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = torch.nn.ModuleList(
            [_IndividualExpert(1.0), _IndividualExpert(2.0)]
        )


def test_onload_and_offload_only_transfer_offloaded_modules(monkeypatch):
    root = torch.nn.Module()
    root.child = torch.nn.Linear(4, 4)
    root._parameters = _FakeOffloadCache()

    calls = []

    def fake_get_cache_init_kwargs(module):
        calls.append(("init", module))
        return {"onload_device": "cpu", "offload_device": "cpu"}

    def fake_remove_module_offload(module, onload_tensors=False):
        calls.append(("onload", module))

    def fake_offload_module(module, **kwargs):
        calls.append(("offload", module))

    monkeypatch.setattr(offloading, "OffloadCache", _FakeOffloadCache)
    monkeypatch.setattr(offloading, "get_cache_init_kwargs", fake_get_cache_init_kwargs)
    monkeypatch.setattr(offloading, "remove_module_offload", fake_remove_module_offload)
    monkeypatch.setattr(offloading, "offload_module", fake_offload_module)

    modules = {"root": root, "root.child": root.child}
    device_map, kwargs = offloading.onload(modules)
    offloading.offload(modules, device_map, kwargs)

    assert calls == [
        ("init", root),
        ("onload", root),
        ("offload", root),
    ]


def test_individual_expert_sequential_target_can_be_onloaded(monkeypatch):
    model = _IndividualExpertModel()
    target = model.experts[1]
    target._parameters = _FakeOffloadCache()
    subgraph_modules = {"experts.1": target}

    calls = []

    def fake_get_cache_init_kwargs(module):
        calls.append(("init", module))
        return {"onload_device": "cpu", "offload_device": "cpu"}

    def fake_remove_module_offload(module, onload_tensors=False):
        calls.append(("onload", module))

    def fake_offload_module(module, **kwargs):
        calls.append(("offload", module))

    monkeypatch.setattr(offloading, "get_cache_init_kwargs", fake_get_cache_init_kwargs)
    monkeypatch.setattr(offloading, "OffloadCache", _FakeOffloadCache)
    monkeypatch.setattr(offloading, "remove_module_offload", fake_remove_module_offload)
    monkeypatch.setattr(offloading, "offload_module", fake_offload_module)
    device_map, kwargs = offloading.onload(subgraph_modules)
    assert target(torch.tensor([3.0])) == torch.tensor([5.0])
    offloading.offload(subgraph_modules, device_map, kwargs)

    assert calls == [
        ("init", target),
        ("onload", target),
        ("offload", target),
    ]


def test_pipeline_waits_only_for_overlapping_offloads():
    first_module = torch.nn.Linear(4, 4)
    second_module = torch.nn.Linear(4, 4)
    overlapping_future = Mock()
    overlapping_future.done.return_value = False
    independent_future = Mock()
    independent_future.done.return_value = False

    pending = [
        ({first_module}, overlapping_future),
        ({second_module}, independent_future),
    ]

    remaining = _wait_for_overlapping_offloads(pending, {first_module})

    overlapping_future.result.assert_called_once_with()
    independent_future.result.assert_not_called()
    assert remaining == [({second_module}, independent_future)]
