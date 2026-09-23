import torch

import llmcompressor.pipelines.sequential.offloading as offloading


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
    offload_kwargs = offloading.onload_modules(modules)
    offloading.offload_modules(modules, offload_kwargs)

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
    offload_kwargs = offloading.onload_modules(subgraph_modules)
    assert target(torch.tensor([3.0])) == torch.tensor([5.0])
    offloading.offload_modules(subgraph_modules, offload_kwargs)

    assert calls == [
        ("init", target),
        ("onload", target),
        ("offload", target),
    ]


class _StagingCache(_FakeOffloadCache):
    def __init__(self, tensor):
        super().__init__()
        self.offloaded_values = {"weight": tensor}
        self.onload_device = "cpu"
        self.is_staged = False
        self.stage_calls = []

    def stage(self, tensor, pin_memory=False):
        self.stage_calls.append((tensor.clone(), pin_memory))
        self.is_staged = True
        return tensor + (1 if pin_memory else 0)


def test_stage_modules_are_consumed_by_onload(monkeypatch):
    root = torch.nn.Module()
    root._parameters = _StagingCache(torch.tensor([2.0]))
    root._buffers = _StagingCache(torch.tensor([3.0]))
    stage_calls = []

    def fake_get_cache_init_kwargs(module):
        return {"onload_device": "cpu", "offload_device": "cpu"}

    def fake_remove_module_offload(module, onload_tensors=False):
        assert onload_tensors is True
        module._parameters = {
            name: tensor + 10
            for name, tensor in module._parameters.offloaded_values.items()
        }
        module._buffers = {
            name: tensor + 20
            for name, tensor in module._buffers.offloaded_values.items()
        }

    def fake_stage_module_offload(module, pin_memory=False):
        stage_calls.append(pin_memory)
        module._parameters.offloaded_values = {
            name: module._parameters.stage(tensor, pin_memory=pin_memory)
            for name, tensor in module._parameters.offloaded_values.items()
        }
        module._buffers.offloaded_values = {
            name: module._buffers.stage(tensor, pin_memory=pin_memory)
            for name, tensor in module._buffers.offloaded_values.items()
        }
        module._parameters.is_staged = True
        module._buffers.is_staged = True

    monkeypatch.setattr(offloading, "OffloadCache", _FakeOffloadCache)
    monkeypatch.setattr(offloading, "get_cache_init_kwargs", fake_get_cache_init_kwargs)
    monkeypatch.setattr(offloading, "remove_module_offload", fake_remove_module_offload)
    monkeypatch.setattr(offloading, "stage_module_offload", fake_stage_module_offload)

    modules = {"root": root}
    result = offloading.stage_modules(modules, pin_memory=True)
    assert result is None
    offloading.onload_modules(modules)

    assert root._parameters["weight"].item() == 12.0
    assert root._buffers["weight"].item() == 23.0
    assert stage_calls == [True]
    assert len(root._parameters.stage_calls) == 1
    assert len(root._buffers.stage_calls) == 1
    assert torch.equal(root._parameters.stage_calls[0][0], torch.tensor([2.0]))
    assert root._parameters.stage_calls[0][1] is True
    assert torch.equal(root._buffers.stage_calls[0][0], torch.tensor([3.0]))
    assert root._buffers.stage_calls[0][1] is True
