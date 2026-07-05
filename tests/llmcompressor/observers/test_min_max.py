import pytest
import torch
from compressed_tensors.quantization.quant_args import QuantizationArgs

from llmcompressor.observers import Observer


@pytest.mark.parametrize(
    "symmetric,expected_scale,expected_zero_point",
    [
        (True, 0.0078, 0),
        (False, 0.0039, -128),
    ],
)
def test_min_max_observer(symmetric, expected_scale, expected_zero_point):
    tensor = torch.tensor([1, 1, 1, 1, 1])
    num_bits = 8

    weights = QuantizationArgs(
        num_bits=num_bits, symmetric=symmetric, observer="minmax"
    )

    observer = weights.observer
    observer = Observer.load_from_registry(observer, base_name="weight", args=weights)
    qparams = observer(tensor).get_qparams()
    scale, zero_point = qparams["scale"], qparams["zero_point"]

    assert round(scale.item(), 4) == expected_scale
    assert round(zero_point.item(), 4) == expected_zero_point


def test_min_max_observer_symmetric_scale_range():
    tensor = torch.rand(4, 4)
    tensor *= 127

    num_bits = 8
    weights = QuantizationArgs(num_bits=num_bits, symmetric=True, observer="minmax")

    observer = weights.observer
    observer = Observer.load_from_registry(observer, base_name="weight", args=weights)
    qparams = observer(tensor).get_qparams()
    scale, zero_point = qparams["scale"], qparams["zero_point"]

    # if symmetric, max symmetric_range = abs(-128) / 255
    assert round(scale.item(), 4) <= 1.0039
    assert round(zero_point.item(), 4) == 0


def test_min_max_observer_value_update():
    inp = torch.tensor([1, 1, 1, 1, 1])
    inp_update_max = torch.tensor([127, 1, 1, 1, 1])
    inp_update_min = torch.tensor([-128, 1, 1, 1, 1])

    delta = 1e-6

    # update the min, max twice total
    tensors = [
        inp,
        inp,
        inp_update_max,  # update max
        inp,
        inp_update_min,  # update min
    ]

    tensor = inp
    num_bits = 8
    weights = QuantizationArgs(
        num_bits=num_bits, strategy="tensor", symmetric=True, observer="minmax"
    )
    observer = weights.observer
    observer = Observer.load_from_registry(observer, base_name="weight", args=weights)
    curr_max = 1
    curr_min = 1
    for i, tensor in enumerate(tensors):
        observer(tensor)
        min_vals = observer.min_vals
        max_vals = observer.max_vals
        curr_max = max(max_vals[0], curr_max)
        curr_min = min(min_vals[0], curr_min)

        if i < 2:
            assert curr_max == 1
            assert curr_min == 1
        elif i < 4:
            assert abs(curr_max - 2.2600) < delta
            assert curr_min == 1
        else:
            assert abs(curr_max - 2.2600) < delta
            assert abs(curr_min - (-0.2900)) < delta
