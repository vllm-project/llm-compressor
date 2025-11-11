import pytest
import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    initialize_module_for_quantization,
)

from llmcompressor.modifiers.quantization.calibration import initialize_observer
from llmcompressor.observers import Observer


@pytest.mark.parametrize(
    "shape,group_size,actorder",
    [
        ((1, 1), None, False),
        ((1, 1), 1, False),
        ((1, 1), 1, True),
        ((64, 64), None, False),
        ((64, 64), 32, False),
        ((64, 64), 32, True),
        ((896, 4096), None, False),
        ((896, 4096), 7, False),
        ((896, 4096), 7, True),
        ((512, 64), None, False),
        ((512, 64), 128, False),
        ((512, 64), 128, True),
    ],
)
def test_observers_update(shape, group_size, actorder):
    module = torch.nn.Linear(*shape)
    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(group_size=group_size, actorder=actorder),
        input_activations=QuantizationArgs(),
        output_activations=QuantizationArgs(),
    )

    input = torch.empty(module.in_features, dtype=module.weight.dtype)
    output = torch.empty(module.out_features, dtype=module.weight.dtype)

    initialize_module_for_quantization(module, scheme)
    initialize_observer(module, "weight")
    initialize_observer(module, "input")
    initialize_observer(module, "output")

    for location, value in (
        ("weight", module.weight),
        ("input", input),
        ("output", output),
    ):
        observer = getattr(module, f"{location}_observer")
        updated_scale, updated_zero_point = observer(value)

        assert_alike(updated_scale, getattr(module, f"{location}_scale"))
        assert_alike(updated_zero_point, getattr(module, f"{location}_zero_point"))


def assert_alike(a, b):
    assert a.dtype == b.dtype
    assert a.shape == b.shape


@pytest.mark.parametrize("is_global", [False, True])
@pytest.mark.parametrize(
    "name,kwargs,observed,exp_min_vals,exp_max_vals",
    (
        (
            "memoryless_minmax",
            {},
            torch.tensor([[0.0, 0.0], [-3.0, 1.0], [-1.0, 3.0]]),
            torch.tensor([[0.0], [-3.0], [-1.0]]),
            torch.tensor([[0.0], [1.0], [3.0]]),
        ),
        (
            "static_minmax",
            {},
            torch.tensor([[0.0, 0.0], [-3.0, 1.0], [-1.0, 3.0]]),
            torch.tensor([[0.0], [-3.0], [-3.0]]),
            torch.tensor([[0.0], [1.0], [3.0]]),
        ),
        (
            "minmax",  # moving average
            {"averaging_constant": 0.1},
            torch.tensor([[0.0, 0.0], [-3.0, 1.0], [-1.0, 3.0]]),
            torch.tensor([[0.0], [-0.3], [-0.37]]),
            torch.tensor([[0.0], [0.1], [0.39]]),
        ),
        (
            "memoryless_mse",
            {},
            torch.tensor([[0.0, 0.0], [-3.0, 1.0], [-1.0, 3.0]]),
            torch.tensor([[0.0], [-3.0], [-1.0]]),
            torch.tensor([[0.0], [1.0], [3.0]]),
        ),
        (
            "mse",  # moving average
            {"averaging_constant": 0.1},
            torch.tensor([[0.0, 0.0], [-3.0, 1.0], [-1.0, 3.0]]),
            torch.tensor([[0.0], [-0.3], [-0.37]]),
            torch.tensor([[0.0], [0.1], [0.39]]),
        ),
    ),
)
def test_observer_min_max_vals(
    name, kwargs, observed, exp_min_vals, exp_max_vals, is_global
):
    observer = Observer.load_from_registry(
        name, base_name="input", args=QuantizationArgs(strategy="tensor"), **kwargs
    )

    min_vals, max_vals = [], []
    for _observed in observed:
        if not is_global:
            _, _, _min_vals, _max_vals = observer._forward_with_minmax(_observed)
        else:
            _, _min_vals, _max_vals = observer._get_global_scale_with_minmax(_observed)

        min_vals.append(_min_vals)
        max_vals.append(_max_vals)

    min_vals = torch.stack(min_vals)
    max_vals = torch.stack(max_vals)
    assert torch.allclose(min_vals, exp_min_vals)
    assert torch.allclose(max_vals, exp_max_vals)
