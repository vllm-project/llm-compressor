from contextlib import nullcontext

import pytest
import torch

from llmcompressor.pipelines.sequential.transformers_helpers import HFTracer


class DummyAttributeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.register_buffer("offset", torch.ones(1))

    def forward(self, hidden_states):
        hidden_states = hidden_states * self.weight + self.weight
        return hidden_states + self.offset + self.offset


class CountingDummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.child = DummyAttributeModule()
        self.named_parameters_calls = 0
        self.named_buffers_calls = 0

    def forward(self, hidden_states):
        return self.child(hidden_states)

    def named_parameters(self, *args, **kwargs):
        self.named_parameters_calls += 1
        return super().named_parameters(*args, **kwargs)

    def named_buffers(self, *args, **kwargs):
        self.named_buffers_calls += 1
        return super().named_buffers(*args, **kwargs)


class DummySharedAttributeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        parameter = torch.nn.Parameter(torch.ones(1))
        self.first_parameter = parameter
        self.second_parameter = parameter

        buffer = torch.ones(1)
        self.register_buffer("first_buffer", buffer)
        self.register_buffer("second_buffer", buffer)

    def forward(self, hidden_states):
        return hidden_states + self.second_parameter + self.second_buffer


class CountingLeafModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        self.register_buffer("unused_offset", torch.ones(1))
        self.named_parameters_calls = 0
        self.named_buffers_calls = 0

    def forward(self, hidden_states):
        return self.linear(hidden_states)

    def named_parameters(self, *args, **kwargs):
        self.named_parameters_calls += 1
        return super().named_parameters(*args, **kwargs)

    def named_buffers(self, *args, **kwargs):
        self.named_buffers_calls += 1
        return super().named_buffers(*args, **kwargs)


class CountingSingleAttributeModel(torch.nn.Module):
    def __init__(self, access_parameter: bool):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.register_buffer("offset", torch.ones(1))
        self.access_parameter = access_parameter
        self.named_parameters_calls = 0
        self.named_buffers_calls = 0

    def forward(self, hidden_states):
        if self.access_parameter:
            return hidden_states + self.weight
        return hidden_states + self.offset

    def named_parameters(self, *args, **kwargs):
        self.named_parameters_calls += 1
        return super().named_parameters(*args, **kwargs)

    def named_buffers(self, *args, **kwargs):
        self.named_buffers_calls += 1
        return super().named_buffers(*args, **kwargs)


class FailingDummyModel(CountingDummyModel):
    def forward(self, hidden_states):
        self.child(hidden_states)
        raise RuntimeError("expected tracing failure")


def test_hf_tracer_caches_parameter_and_buffer_names():
    tracer = HFTracer()

    # Reuse the tracer to ensure that the name caches are rebuilt for every model.
    for _ in range(2):
        model = CountingDummyModel()
        graph = tracer.trace(
            model,
            dummy_inputs={"hidden_states": torch.zeros(1)},
        )

        get_attr_targets = [
            node.target for node in graph.nodes if node.op == "get_attr"
        ]
        assert get_attr_targets == ["child.weight", "child.offset"]
        assert model.named_parameters_calls == 1
        assert model.named_buffers_calls == 1
        assert tracer._parameter_to_name is None
        assert tracer._buffer_to_name is None


def test_hf_tracer_preserves_shared_attribute_names():
    model = DummySharedAttributeModel()
    graph = HFTracer().trace(
        model,
        dummy_inputs={"hidden_states": torch.zeros(1)},
    )

    get_attr_targets = [node.target for node in graph.nodes if node.op == "get_attr"]
    assert get_attr_targets == ["first_parameter", "first_buffer"]


def test_hf_tracer_builds_attribute_maps_lazily():
    model = CountingLeafModel()
    tracer = HFTracer()

    graph = tracer.trace(
        model,
        dummy_inputs={"hidden_states": torch.zeros(1, 1)},
    )

    assert any(node.op == "call_module" for node in graph.nodes)
    assert model.named_parameters_calls == 0
    assert model.named_buffers_calls == 0
    assert tracer._parameter_to_name is None
    assert tracer._buffer_to_name is None


@pytest.mark.parametrize(
    "access_parameter,expected_parameter_calls,expected_buffer_calls",
    [(True, 1, 0), (False, 0, 1)],
)
def test_hf_tracer_builds_attribute_maps_independently(
    access_parameter,
    expected_parameter_calls,
    expected_buffer_calls,
):
    model = CountingSingleAttributeModel(access_parameter)

    HFTracer().trace(
        model,
        dummy_inputs={"hidden_states": torch.zeros(1)},
    )

    assert model.named_parameters_calls == expected_parameter_calls
    assert model.named_buffers_calls == expected_buffer_calls


def test_hf_tracer_clears_attribute_maps_after_failure(monkeypatch):
    tracer = HFTracer()
    # Avoid installing process-wide patches in a deliberately failing trace.
    tracer.orig_fns = set()
    monkeypatch.setattr(tracer, "patch_for_tracing", lambda _: nullcontext())
    failing_model = FailingDummyModel()

    with pytest.raises(RuntimeError, match="expected tracing failure"):
        tracer.trace(
            failing_model,
            dummy_inputs={"hidden_states": torch.zeros(1)},
        )

    assert failing_model.named_parameters_calls == 1
    assert failing_model.named_buffers_calls == 1
    assert tracer._parameter_to_name is None
    assert tracer._buffer_to_name is None

    recovery_model = CountingDummyModel()
    graph = tracer.trace(
        recovery_model,
        dummy_inputs={"hidden_states": torch.zeros(1)},
    )

    get_attr_targets = [node.target for node in graph.nodes if node.op == "get_attr"]
    assert get_attr_targets == ["child.weight", "child.offset"]
    assert recovery_model.named_parameters_calls == 1
    assert recovery_model.named_buffers_calls == 1
    assert tracer._parameter_to_name is None
    assert tracer._buffer_to_name is None
