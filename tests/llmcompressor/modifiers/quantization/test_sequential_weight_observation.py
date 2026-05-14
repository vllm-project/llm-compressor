from unittest.mock import patch

from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    initialize_module_for_quantization,
)
from torch import nn

from llmcompressor.core import Event, EventType, State
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.quantization.calibration import initialize_observer


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
    modifier.on_start(state, None)

    # Patch update_statistics_from_observed to track which observers are called
    with patch.object(
        model[0].weight_observer, "update_statistics_from_observed"
    ) as mock0, patch.object(
        model[1].weight_observer, "update_statistics_from_observed"
    ) as mock1, patch.object(
        model[2].weight_observer, "update_statistics_from_observed"
    ) as mock2:
        modifier.on_event(
            state, Event(type_=EventType.SEQUENTIAL_EPOCH_END), modules=[model[0]]
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
    modifier.on_start(state, None)

    # Patch get_qparams and sync_activation_stats to track calls
    with patch.object(
        model[0].input_observer,
        "get_qparams",
        wraps=model[0].input_observer.get_qparams,
    ) as qp_mock0, patch.object(
        model[1].input_observer,
        "get_qparams",
        wraps=model[1].input_observer.get_qparams,
    ) as qp_mock1, patch.object(
        model[2].input_observer,
        "get_qparams",
        wraps=model[2].input_observer.get_qparams,
    ) as qp_mock2, patch.object(
        model[0].input_observer, "sync_activation_stats", return_value=[]
    ) as sync_mock0, patch.object(
        model[1].input_observer, "sync_activation_stats", return_value=[]
    ) as sync_mock1, patch.object(
        model[2].input_observer, "sync_activation_stats", return_value=[]
    ) as sync_mock2, patch(
        "llmcompressor.modifiers.quantization.quantization.mixin.is_distributed",
        return_value=True,
    ):
        # Process chunks sequentially
        modifier.on_event(
            state, Event(type_=EventType.SEQUENTIAL_EPOCH_END), modules=[model[0]]
        )
        modifier.on_event(
            state, Event(type_=EventType.SEQUENTIAL_EPOCH_END), modules=[model[1]]
        )
        modifier.on_event(
            state, Event(type_=EventType.SEQUENTIAL_EPOCH_END), modules=[model[2]]
        )

        # Each observer's methods should be called exactly once
        assert qp_mock0.call_count == 1
        assert qp_mock1.call_count == 1
        assert qp_mock2.call_count == 1
        assert sync_mock0.call_count == 1
        assert sync_mock1.call_count == 1
        assert sync_mock2.call_count == 1
