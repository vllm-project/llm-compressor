from llmcompressor.core import EventType, LifecycleCallbacks


def test_sequential_epoch_end_passes_modules(monkeypatch):
    modules = [object()]
    seen = {}

    def fake_event(cls, event_type, **kwargs):
        seen["event_type"] = event_type
        seen["kwargs"] = kwargs

    monkeypatch.setattr(LifecycleCallbacks, "event", classmethod(fake_event))

    LifecycleCallbacks.sequential_epoch_end(modules, extra_kwarg=True)

    assert seen["event_type"] == EventType.SEQUENTIAL_EPOCH_END
    assert seen["kwargs"] == {"modules": modules, "extra_kwarg": True}


def test_sequential_epoch_start_passes_modules(monkeypatch):
    modules = [object()]
    seen = {}

    def fake_event(cls, event_type, **kwargs):
        seen["event_type"] = event_type
        seen["kwargs"] = kwargs

    monkeypatch.setattr(LifecycleCallbacks, "event", classmethod(fake_event))

    LifecycleCallbacks.sequential_epoch_start(modules, extra_kwarg=True)

    assert seen["event_type"] == EventType.SEQUENTIAL_EPOCH_START
    assert seen["kwargs"] == {"modules": modules, "extra_kwarg": True}


def test_calibration_start_passes_optional_module_scope(monkeypatch):
    modules = [object()]
    seen = {}

    def fake_event(cls, event_type, **kwargs):
        seen["event_type"] = event_type
        seen["kwargs"] = kwargs

    monkeypatch.setattr(LifecycleCallbacks, "event", classmethod(fake_event))

    LifecycleCallbacks.calibration_start(modules=modules)

    assert seen["event_type"] == EventType.CALIBRATION_START
    assert seen["kwargs"] == {"modules": modules}


def test_modifiers_share_updated_module_and_offload_bookkeeping():
    import torch

    from llmcompressor.core import create_session
    from llmcompressor.modifiers import Modifier
    from llmcompressor.modifiers.quantization import QuantizationModifier

    class ReplaceChildren(Modifier):
        def on_initialize(self, state, **kwargs):
            return True

        def on_sequential_epoch_start(self, state, event, modules, **kwargs):
            state.model[0] = torch.nn.Sequential(torch.nn.Linear(8, 8))
            kwargs["onloaded_modules"]["0"] = state.model[0]
            kwargs["onloaded_modules"]["0.0"] = state.model[0][0]
            kwargs["offload_kwargs"]["0.0"] = kwargs["offload_kwargs"]["0"].copy()

    class CheckScope(Modifier):
        def on_initialize(self, state, **kwargs):
            return True

        def on_sequential_epoch_start(self, state, event, modules, **kwargs):
            layer = state.model[0][0]
            assert layer in modules
            assert layer.weight_observer in modules
            assert (
                kwargs["onloaded_modules"]["0.0.weight_observer"]
                is layer.weight_observer
            )

        def on_sequential_epoch_end(self, state, event, modules, **kwargs):
            assert "0.0.weight_observer" not in kwargs["onloaded_modules"]
            assert "0.0.weight_observer" not in kwargs["offload_kwargs"]

    model = torch.nn.Sequential(torch.nn.Linear(8, 8))
    original = model[0]
    onloaded_modules = {"0": original}
    offload_kwargs = {
        "0": {
            "onload_device": "cpu",
            "offload_device": "disk",
            "offload_dir": "/tmp/example",
        }
    }
    with create_session() as session:
        session.initialize(
            model=model,
            recipe=[
                ReplaceChildren(),
                QuantizationModifier(
                    config_groups={
                        "weights": {
                            "targets": ["Linear"],
                            "weights": {"num_bits": 8, "strategy": "channel"},
                        }
                    }
                ),
                CheckScope(),
            ],
        )
        # Exercise per-subgraph observer cleanup as well as creation.
        session.state.layerwise_decompression = True
        LifecycleCallbacks.calibration_start(modules=[])
        LifecycleCallbacks.sequential_epoch_start(
            [original], onloaded_modules=onloaded_modules, offload_kwargs=offload_kwargs
        )
        assert original not in onloaded_modules.values()
        assert offload_kwargs["0.0"]["offload_device"] == "disk"
        LifecycleCallbacks.sequential_epoch_end(
            [original], onloaded_modules=onloaded_modules, offload_kwargs=offload_kwargs
        )
        session.finalize()
