import torch

import llmcompressor.pipelines.sequential.offloading as offloading


class _FakeLinearExperts2D(torch.nn.Module):
    pass


class _FakeOffloadCache(dict):
    pass


class _NestedModule(_FakeLinearExperts2D):
    def __init__(self):
        super().__init__()
        self.child = torch.nn.Linear(4, 4)


def test_disable_offloading_controlled_uses_top_level_wrappers_only(monkeypatch):
    root = _NestedModule()

    calls = []

    def fake_get_cache_init_kwargs(module):
        calls.append(("init", module))
        return {"onload_device": "cpu", "offload_device": "cpu"}

    def fake_remove_module_offload(module, onload_tensors=False):
        calls.append(("onload", module))

    def fake_offload_module(module, **kwargs):
        calls.append(("offload", module))

    class _FakeAccelerator:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def empty_cache():
            raise AssertionError("empty_cache should not be called when unavailable")

    monkeypatch.setattr(offloading, "LinearExperts2D", _FakeLinearExperts2D)
    monkeypatch.setattr(offloading, "OffloadCache", _FakeOffloadCache)
    monkeypatch.setattr(offloading, "get_cache_init_kwargs", fake_get_cache_init_kwargs)
    monkeypatch.setattr(offloading, "remove_module_offload", fake_remove_module_offload)
    monkeypatch.setattr(offloading, "offload_module", fake_offload_module)
    monkeypatch.setattr(
        offloading.torch, "accelerator", _FakeAccelerator(), raising=False
    )
    monkeypatch.setattr(offloading, "_log_cuda_memory", lambda *_args, **_kwargs: None)

    root._parameters = _FakeOffloadCache()

    with offloading.disable_offloading_controlled(root, [root, root.child]):
        pass

    assert calls[0] == ("init", root)
    assert calls[1:4] == [
        ("onload", root),
        ("onload", root.child),
        ("offload", root),
    ]
    assert not any(call == ("init", root.child) for call in calls)
    assert not hasattr(root, "_onload_wrapper")
