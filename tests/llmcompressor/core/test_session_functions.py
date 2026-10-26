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


def test_sequential_epoch_start_supports_whole_model_scope(monkeypatch):
    seen = {}

    def fake_event(cls, event_type, **kwargs):
        seen["event_type"] = event_type
        seen["kwargs"] = kwargs

    monkeypatch.setattr(LifecycleCallbacks, "event", classmethod(fake_event))

    LifecycleCallbacks.sequential_epoch_start()

    assert seen["event_type"] == EventType.SEQUENTIAL_EPOCH_START
    assert seen["kwargs"] == {"modules": None}


def test_modifiers_keep_original_scope_and_defer_observer_cleanup():
    import torch

    from llmcompressor.core import create_session
    from llmcompressor.modifiers import Modifier
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.modifiers.transform import QuIPModifier

    class CheckScope(Modifier):
        def on_initialize(self, state, **kwargs):
            return True

        def _prepare_sequential_epoch(self, state, event, modules, **kwargs):
            layer = state.model[0]
            assert modules == scope
            assert hasattr(layer, "v_input")
            assert layer.v_input not in modules
            assert not hasattr(layer, "weight_observer")

        def on_sequential_epoch_start(self, state, event, modules, **kwargs):
            layer = state.model[0]
            assert modules == scope
            assert hasattr(layer, "weight_observer")

        def _cleanup_sequential_epoch(self, state, event, modules, **kwargs):
            assert modules == scope
            assert not hasattr(state.model[0], "weight_observer")

        def on_sequential_epoch_end(self, state, event, modules, **kwargs):
            # A later modifier such as QAD must still be able to re-observe weights.
            assert modules == scope
            assert hasattr(state.model[0], "weight_observer")

    model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8))
    scope = [model[0]]
    with create_session() as session:
        session.initialize(
            model=model,
            recipe=[
                QuIPModifier(transform_type="hadamard"),
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
        LifecycleCallbacks.sequential_epoch_start(scope)
        assert not hasattr(model[1], "quantization_scheme")
        assert not hasattr(model[1], "v_input")
        LifecycleCallbacks.sequential_epoch_end(scope)
        session.finalize()
