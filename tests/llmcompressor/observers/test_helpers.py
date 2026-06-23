import pytest
import torch
import torch.nn as nn
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStrategy,
    initialize_module_for_quantization,
)

from llmcompressor.modifiers.quantization.calibration import initialize_observer
from llmcompressor.observers.helpers import (
    flatten_for_calibration,
    fuse_weight_observers,
)


@pytest.mark.parametrize(
    "args",
    [
        QuantizationArgs(strategy="tensor"),
        QuantizationArgs(strategy="tensor_group", group_size=4),
    ],
)
def test_flatten_for_calibration_input(args):
    module = torch.nn.Linear(8, 10)
    scheme = QuantizationScheme(targets=[], input_activations=args)
    initialize_module_for_quantization(module, scheme)

    input = torch.empty((3, 5, 8))
    input_flattened = flatten_for_calibration(input, "input", scheme.input_activations)
    assert input_flattened.shape[1:-1] == module.input_scale.shape
    assert input_flattened.shape[1:-1] == module.input_zero_point.shape


@pytest.mark.parametrize(
    "args",
    [
        QuantizationArgs(strategy="tensor"),
        QuantizationArgs(strategy="channel"),
        QuantizationArgs(strategy="group", group_size=4),
        QuantizationArgs(strategy="tensor_group", group_size=4),
        QuantizationArgs(strategy="block", block_structure=[5, 4]),
        # When block structure does not evenly divide module.weight
        QuantizationArgs(strategy="block", block_structure=[3, 3]),
    ],
)
def test_flatten_for_calibration_weights(args):
    module = torch.nn.Linear(8, 10)
    scheme = QuantizationScheme(targets=[], weights=args)
    initialize_module_for_quantization(module, scheme)

    weight_flattened = flatten_for_calibration(
        module.weight,
        "weight",
        scheme.weights,
    )
    assert weight_flattened.shape[1:-1] == module.weight_scale.shape
    assert weight_flattened.shape[1:-1] == module.weight_zero_point.shape


class _SlidingAttention(nn.Module):
    """Mimics Gemma4 sliding attention where v_proj is None."""

    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size // num_heads, bias=False)
        self.v_proj = None


class _DecoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.self_attn = _SlidingAttention(hidden_size, num_heads)


class _ToyGemma4(nn.Module):
    def __init__(self, hidden_size=64, num_heads=4, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList(
            [_DecoderLayer(hidden_size, num_heads) for _ in range(num_layers)]
        )


def _attach_tensor_group_scheme(module):
    scheme = QuantizationScheme(
        targets=[],
        weights=QuantizationArgs(
            strategy=QuantizationStrategy.TENSOR_GROUP,
            group_size=16,
            num_bits=4,
        ),
    )
    initialize_module_for_quantization(module, scheme)
    initialize_observer(module, base_name="weight")


def test_fuse_weight_observers_with_none_v_proj():
    """
    fuse_weight_observers should skip None layers in fused groups.

    Gemma4 sliding attention layers have v_proj=None. Without the fix,
    getattr(None, 'weight_observer', None) returns None, which sets
    only_obs=False, and the assertion fires on the remaining q/k observers.
    """
    model = _ToyGemma4()

    for layer in model.layers:
        _attach_tensor_group_scheme(layer.self_attn.q_proj)
        _attach_tensor_group_scheme(layer.self_attn.k_proj)

    fuse_weight_observers(model)

    for layer in model.layers:
        q_obs = layer.self_attn.q_proj.weight_observer
        k_obs = layer.self_attn.k_proj.weight_observer
        assert q_obs._fusions, "q_proj observer should have fusions"
        assert k_obs._fusions, "k_proj observer should have fusions"
