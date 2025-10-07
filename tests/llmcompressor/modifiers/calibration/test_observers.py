import pytest
import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    initialize_module_for_quantization,
)

from llmcompressor.modifiers.quantization.calibration import initialize_observer


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
