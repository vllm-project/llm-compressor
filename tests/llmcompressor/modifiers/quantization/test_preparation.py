from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import pytest
import torch
from compressed_tensors.offload import OffloadCache, offload_module

from llmcompressor.core import CompressionSession, EventType, State
from llmcompressor.modifiers.quantization import QuantizationModifier


@pytest.mark.parametrize("scheme", ["W8A8", "W8A16", "NVFP4"])
def test_prepared_quantization_matches_synchronous_calibration(scheme):
    torch.manual_seed(42)
    original = torch.nn.Sequential(torch.nn.Linear(128, 128))
    inputs = torch.randn(2, 128)
    results = []
    for prepare in (False, True):
        model = deepcopy(original)
        modifier = QuantizationModifier(scheme=scheme)
        session = CompressionSession()
        session.initialize(model=model, recipe=modifier)
        session.lifecycle.event(EventType.CALIBRATION_START, modules=[])
        prepared = None
        if prepare:
            job = modifier.create_calibration_preparation(session.state, [model[0]])
            with ThreadPoolExecutor(max_workers=1) as executor:
                prepared = executor.submit(job).result()
            assert not hasattr(model[0], "quantization_scheme")
            assert not model[0]._modules
            assert not model[0]._forward_pre_hooks
            assert all(
                param.device.type == "cpu"
                for param in prepared.parameters[model[0]].values()
            )

        session.lifecycle.event(
            EventType.SEQUENTIAL_EPOCH_START,
            modules=[model[0]],
            modifier_preparations=[prepared],
        )
        if prepare:
            assert model[0].weight_observer is prepared.observers[model[0]]["weight"]
        with torch.no_grad():
            model(inputs)
        session.lifecycle.event(EventType.SEQUENTIAL_EPOCH_END, modules=[model[0]])
        session.lifecycle.event(EventType.CALIBRATION_END)
        results.append(model.state_dict())

    assert results[0].keys() == results[1].keys()
    for key in results[0]:
        torch.testing.assert_close(results[0][key], results[1][key])


def test_preparation_does_not_onload_weights_or_change_offload_flags(monkeypatch):
    model = torch.nn.Sequential(torch.nn.Linear(32, 32))
    offload_module(model[0], "cuda", "cpu")
    modifier = QuantizationModifier(scheme="W8A8")
    state = State(model=model)
    modifier.on_initialize(state)

    def unexpected_onload(*args, **kwargs):
        pytest.fail("preparation onloaded a model weight")

    monkeypatch.setattr(type(model[0]._parameters), "onload", unexpected_onload)
    job = modifier.create_calibration_preparation(state, [model[0]])
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(job).result()
    assert result.parameters[model[0]]["weight_scale"].device.type == "cpu"
    assert not OffloadCache.onloading_disabled
    assert not hasattr(model[0], "weight_observer")


def test_preparation_defers_packed_weight_initialization():
    model = torch.nn.Sequential(torch.nn.Linear(32, 32))
    del model[0].weight
    model[0].register_parameter(
        "weight_packed", torch.nn.Parameter(torch.ones(32, 8), requires_grad=False)
    )
    modifier = QuantizationModifier(scheme="W8A8")
    state = State(model=model)
    modifier.on_initialize(state)
    result = modifier.create_calibration_preparation(state, [model[0]])()
    assert not result.parameters
    assert not result.observers
