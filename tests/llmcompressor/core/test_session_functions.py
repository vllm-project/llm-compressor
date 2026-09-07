from llmcompressor.core import EventType, LifecycleCallbacks


def test_sequential_epoch_start_passes_context(monkeypatch):
    seen = {}

    def fake_event(cls, event_type, **kwargs):
        seen.update(event_type=event_type, **kwargs)

    monkeypatch.setattr(LifecycleCallbacks, "event", classmethod(fake_event))
    modules, subgraph, cache = [object()], object(), object()
    LifecycleCallbacks.sequential_epoch_start(
        modules, subgraph=subgraph, activations=cache, subgraph_index=2
    )
    assert seen == dict(
        event_type=EventType.SEQUENTIAL_EPOCH_START,
        modules=modules,
        subgraph=subgraph,
        activations=cache,
        subgraph_index=2,
    )


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
