from unittest.mock import patch

import pytest
import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStatus,
    initialize_module_for_quantization,
)
from torch import nn

from llmcompressor.core import (
    CompressionSession,
    Event,
    EventType,
    State,
)
from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.quantization.calibration import initialize_observer


class _FakeDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(64, 64)
        self.k_proj = nn.Linear(64, 64)
        self.v_proj = nn.Linear(64, 64)


def test_sequential_epoch_end_only_observes_passed_modules():
    """Verify only subgraph modules observed, not all model weights."""
    model = nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 10), nn.Linear(10, 10))
    args = QuantizationArgs(num_bits=8, type="int", symmetric=True, strategy="tensor")
    for layer in model:
        initialize_module_for_quantization(
            layer, QuantizationScheme(targets=[], weights=args)
        )
        initialize_observer(layer, "weight")
    modifier = QuantizationModifier(targets="Linear", scheme="FP8")
    state = State(model=model)
    modifier.on_initialize(state)
    modifier.on_sequential_epoch_initialize(state, None)
    modifier.on_calibration_start(state, None)

    # Wrap update_statistics_from_observed to track which observers are called
    with (
        patch.object(
            model[0].weight_observer,
            "update_statistics_from_observed",
            wraps=model[0].weight_observer.update_statistics_from_observed,
        ) as mock0,
        patch.object(
            model[1].weight_observer,
            "update_statistics_from_observed",
            wraps=model[1].weight_observer.update_statistics_from_observed,
        ) as mock1,
        patch.object(
            model[2].weight_observer,
            "update_statistics_from_observed",
            wraps=model[2].weight_observer.update_statistics_from_observed,
        ) as mock2,
    ):
        modifier.on_sequential_epoch_end(
            state,
            Event(type_=EventType.SEQUENTIAL_EPOCH_END),
            modules=list(model[0].modules()),
        )
        mock0.assert_called_once()  # Only first module should be observed
        mock1.assert_not_called()  # Second module should NOT be observed
        mock2.assert_not_called()  # Third module should NOT be observed


def test_sequential_activation_qparams_only_updated_once_per_module():
    """Verify activation qparams and sync only called once per module per chunk."""
    model = nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 10), nn.Linear(10, 10))
    act_args = QuantizationArgs(
        num_bits=8, type="int", symmetric=True, strategy="tensor"
    )
    for layer in model:
        initialize_module_for_quantization(
            layer, QuantizationScheme(targets=[], input_activations=act_args)
        )
        initialize_observer(layer, "input")
        layer.input_observer(layer.weight)  # Trigger observer to have statistics
    modifier = QuantizationModifier(targets="Linear", scheme="W8A8")
    state = State(model=model)
    modifier.on_initialize(state)
    modifier.on_sequential_epoch_initialize(state, None)
    modifier.on_calibration_start(state, None)

    # Patch get_qparams and sync_activation_stats to track calls
    with (
        patch.object(
            model[0].input_observer,
            "get_qparams",
            wraps=model[0].input_observer.get_qparams,
        ) as qp_mock0,
        patch.object(
            model[1].input_observer,
            "get_qparams",
            wraps=model[1].input_observer.get_qparams,
        ) as qp_mock1,
        patch.object(
            model[2].input_observer,
            "get_qparams",
            wraps=model[2].input_observer.get_qparams,
        ) as qp_mock2,
        patch.object(
            model[0].input_observer, "sync_activation_stats", return_value=[]
        ) as sync_mock0,
        patch.object(
            model[1].input_observer, "sync_activation_stats", return_value=[]
        ) as sync_mock1,
        patch.object(
            model[2].input_observer, "sync_activation_stats", return_value=[]
        ) as sync_mock2,
        patch(
            "llmcompressor.modifiers.quantization.quantization.mixin.is_distributed",
            return_value=True,
        ),
    ):
        # Process chunks sequentially
        modifier.on_sequential_epoch_end(
            state,
            Event(type_=EventType.SEQUENTIAL_EPOCH_END),
            modules=list(model[0].modules()),
        )
        modifier.on_sequential_epoch_end(
            state,
            Event(type_=EventType.SEQUENTIAL_EPOCH_END),
            modules=list(model[1].modules()),
        )
        modifier.on_sequential_epoch_end(
            state,
            Event(type_=EventType.SEQUENTIAL_EPOCH_END),
            modules=list(model[2].modules()),
        )

        # Each observer's methods should be called exactly once
        assert qp_mock0.call_count == 1
        assert qp_mock1.call_count == 1
        assert qp_mock2.call_count == 1
        assert sync_mock0.call_count == 1
        assert sync_mock1.call_count == 1
        assert sync_mock2.call_count == 1


def test_nested_parent_modules_produce_valid_global_scale():
    """When SEQUENTIAL_EPOCH_END receives parent modules (as the pipeline does),
    child Linear layers must get finite, positive global_scale values —
    not uninitialized torch.empty garbage."""
    model = nn.Sequential(_FakeDecoderLayer())
    tg = QuantizationArgs(
        num_bits=4, type="float", symmetric=True, group_size=16, strategy="tensor_group"
    )
    scheme = QuantizationScheme(
        targets=[],
        weights=tg,
        input_activations=QuantizationArgs(**{**tg.model_dump(), "dynamic": "local"}),
    )
    for child in (model[0].q_proj, model[0].k_proj, model[0].v_proj):
        initialize_module_for_quantization(child, scheme)

    modifier = QuantizationModifier(targets="Linear", scheme="NVFP4")
    state = State(model=model)
    modifier.on_initialize(state)
    modifier.on_sequential_epoch_initialize(state, None)
    modifier.on_calibration_start(state, None)

    for child in (model[0].q_proj, model[0].k_proj, model[0].v_proj):
        child.weight_observer(child.weight)
        child.input_observer(torch.randn(2, 64))

    modifier.on_sequential_epoch_end(
        state,
        Event(type_=EventType.SEQUENTIAL_EPOCH_END),
        modules=list(model.modules()),
    )

    for name, param in model.named_parameters():
        if "global_scale" in name:
            assert torch.isfinite(param).all() and param.item() > 0, (
                f"{name} not valid: {param.item()}"
            )


def test_session_initialization_defers_quantization_until_scope_is_onloaded():
    model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))
    modifier = QuantizationModifier(targets="Linear", scheme="W8A8")
    session = CompressionSession()

    session.initialize(recipe=modifier, model=model)

    assert getattr(model[0], "quantization_scheme", None) is None
    assert getattr(model[1], "quantization_scheme", None) is None

    session.lifecycle.event(EventType.SEQUENTIAL_EPOCH_INITIALIZE, modules=[model[0]])
    session.lifecycle.event(EventType.CALIBRATION_START, modules=[model[0]])

    assert model[0].quantization_scheme is not None
    assert hasattr(model[0], "weight_observer")
    assert getattr(model[1], "quantization_scheme", None) is None

    session.lifecycle.event(EventType.SEQUENTIAL_EPOCH_INITIALIZE, modules=[model[1]])
    assert model[1].quantization_scheme is not None
    assert not hasattr(model[1], "weight_observer")
    session.lifecycle.event(EventType.CALIBRATION_START, modules=[model[1]])
    assert hasattr(model[1], "weight_observer")

    session.lifecycle.event(EventType.CALIBRATION_END)


def test_overlapping_initialization_preserves_existing_quantization():
    model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))
    modifier = QuantizationModifier(targets="Linear", scheme="W8A8")
    modifier.initialize_quantization(model, modules=[model[0]])
    scale = model[0].weight_scale
    with torch.no_grad():
        scale.fill_(7)
    model[0].quantization_status = QuantizationStatus.FROZEN
    model[0].quantization_enabled = True

    modifier.initialize_quantization(model)

    assert model[0].weight_scale is scale
    assert torch.all(scale == 7)
    assert model[0].quantization_status == QuantizationStatus.FROZEN
    assert model[0].quantization_enabled
    assert model[1].quantization_status == QuantizationStatus.INITIALIZED
    assert not model[1].quantization_enabled


def test_basic_scope_initializes_before_calibration_start():
    model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))
    modifier = QuantizationModifier(targets="Linear", scheme="W8A8")
    session = CompressionSession()

    session.initialize(recipe=modifier, model=model)
    assert getattr(model[0], "quantization_scheme", None) is None
    assert getattr(model[1], "quantization_scheme", None) is None

    session.lifecycle.event(EventType.SEQUENTIAL_EPOCH_INITIALIZE)
    assert model[0].quantization_scheme is not None
    assert not hasattr(model[0], "weight_observer")
    session.lifecycle.event(EventType.CALIBRATION_START)

    assert model[0].quantization_scheme is not None
    assert model[1].quantization_scheme is not None
    assert hasattr(model[0], "weight_observer")
    assert hasattr(model[1], "weight_observer")

    session.lifecycle.event(EventType.CALIBRATION_END)


def test_session_always_defers_model_setup_for_mixed_recipes():
    class _NoOpModifier(Modifier):
        def on_initialize(self, state: State, **kwargs) -> bool:
            return True

    model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))
    quant_modifier = QuantizationModifier(targets="Linear", scheme="W8A8")
    session = CompressionSession()

    session.initialize(
        recipe=[quant_modifier, _NoOpModifier()],
        model=model,
    )

    assert not hasattr(model[0], "quantization_scheme")
    assert not hasattr(model[1], "quantization_scheme")

    session.lifecycle.event(EventType.SEQUENTIAL_EPOCH_INITIALIZE, modules=[model[0]])
    session.lifecycle.event(EventType.CALIBRATION_START, modules=[model[0]])
    assert model[0].quantization_scheme is not None
    assert not hasattr(model[1], "quantization_scheme")
    session.lifecycle.event(EventType.CALIBRATION_END)


def test_tensor_group_quantization_supports_lazy_initialization():
    model = nn.Sequential(_FakeDecoderLayer(), _FakeDecoderLayer())
    modifier = QuantizationModifier(targets="Linear", scheme="NVFP4")
    session = CompressionSession()
    session.initialize(model=model, recipe=modifier)

    session.lifecycle.event(EventType.SEQUENTIAL_EPOCH_INITIALIZE, modules=[])
    session.lifecycle.event(EventType.CALIBRATION_START, modules=[])
    session.lifecycle.event(
        EventType.SEQUENTIAL_EPOCH_INITIALIZE, modules=list(model[0].modules())
    )
    session.lifecycle.event(
        EventType.CALIBRATION_START, modules=list(model[0].modules())
    )
    assert hasattr(model[0].q_proj, "weight_observer")
    assert not hasattr(model[1].q_proj, "quantization_scheme")
    session.lifecycle.event(EventType.CALIBRATION_END)


def test_lazy_tensor_group_calibration_rejects_split_fused_layers():
    model = nn.Sequential(_FakeDecoderLayer())
    modifier = QuantizationModifier(targets="Linear", scheme="NVFP4")
    state = State(model=model)
    modifier.on_initialize(state)

    q_proj = model[0].q_proj
    with pytest.raises(ValueError, match="span multiple sequential calibration"):
        modifier.on_sequential_epoch_initialize(state, None, modules=[q_proj])


def test_layerwise_quantization_only_initializes_and_freezes_passed_modules():
    model = nn.Sequential(nn.Linear(256, 256), nn.Linear(256, 256))
    modifier = QuantizationModifier(targets="Linear", scheme="W8A16")
    state = State(model=model)
    state.layerwise_decompression = True

    modifier.on_initialize(state)

    assert not hasattr(model[0], "quantization_scheme")
    assert not hasattr(model[1], "quantization_scheme")

    modules = list(model[0].modules())
    with patch(
        "llmcompressor.modifiers.quantization.quantization.mixin.fuse_weight_observers"
    ) as fuse_weight_observers:
        modifier.on_sequential_epoch_initialize(
            state,
            Event(type_=EventType.SEQUENTIAL_EPOCH_INITIALIZE),
            modules=modules,
        )
        assert not hasattr(model[0], "weight_observer")
        modifier.on_calibration_start(state, None, modules=modules)

    fuse_weight_observers.assert_called_once_with(model, modules=modules)

    assert model[0].quantization_status == QuantizationStatus.CALIBRATION
    assert hasattr(model[0], "weight_observer")
    assert not hasattr(model[1], "quantization_scheme")

    modifier.on_sequential_epoch_end(
        state,
        Event(type_=EventType.SEQUENTIAL_EPOCH_END),
        modules=list(model[0].modules()),
    )
    assert hasattr(model[0], "weight_observer")
    modifier.on_calibration_end(state, None, modules=list(model[0].modules()))

    assert model[0].quantization_status == QuantizationStatus.FROZEN
    assert not hasattr(model[0], "weight_observer")
    assert not hasattr(model[1], "quantization_scheme")
