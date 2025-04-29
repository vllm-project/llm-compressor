import pytest
import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    initialize_module_for_quantization,
)

from llmcompressor.modifiers.quantization.calibration import initialize_observer


@pytest.mark.parametrize(
    "shape,group_size",
    [
        ((64, 64), None),
        ((64, 64), 128),
        ((1792, 4096), None),
        ((1792, 4096), 128),
        ((3420, 64), None),
        ((3420, 64), 128),
    ],
)
def test_observers_update(shape, group_size):
    module = torch.nn.Linear(*shape)
    args = QuantizationArgs(group_size=group_size)
    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=args,
        input_activations=args,
        output_activations=args,
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
        g_idx = getattr(module, "g_idx", None)
        updated_scale, updated_zero_point = observer(value, g_idx=g_idx)

        assert_alike(updated_scale, getattr(module, f"{location}_scale"))
        assert_alike(updated_zero_point, getattr(module, f"{location}_zero_point"))


def assert_alike(a, b):
    assert a.dtype == b.dtype
    assert a.shape == b.shape
