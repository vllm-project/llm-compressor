import pytest
import torch
from compressed_tensors.quantization.quant_args import QuantizationArgs

from llmcompressor.observers import Observer


def make_dummy_g_idx(columns: int, group_size: int) -> torch.Tensor:
    perm = torch.randperm(columns)
    return torch.tensor([index // group_size for index in range(columns)])[perm]


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
    scale, zero_point = observer(tensor)

    assert round(scale.item(), 4) == expected_scale
    assert round(zero_point.item(), 4) == expected_zero_point


def test_min_max_observer_symmetric_scale_range():
    tensor = torch.rand(4, 4)
    tensor *= 127

    num_bits = 8
    weights = QuantizationArgs(num_bits=num_bits, symmetric=True, observer="minmax")

    observer = weights.observer
    observer = Observer.load_from_registry(observer, base_name="weight", args=weights)
    scale, zero_point = observer(tensor)

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
        _, _, min_vals, max_vals = observer._forward_with_minmax(tensor)
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


def test_g_idx():
    group_size = 2
    input_shape = (128, 512)
    tensor = torch.rand(input_shape)
    weights = QuantizationArgs(num_bits=8, group_size=group_size, observer="minmax")

    module = torch.nn.Linear(512, 1)
    g_idx = make_dummy_g_idx(tensor.shape[1], group_size)
    module.weight_g_idx = g_idx

    observer = Observer.load_from_registry(
        weights.observer, base_name="weight", args=weights, module=module
    )
    scale_g_idx, zero_point_g_idx = observer(tensor)

    observer = Observer.load_from_registry(
        weights.observer, base_name="weight", args=weights, module=module
    )
    del module.weight_g_idx
    scale, zero_point = observer(tensor[:, torch.argsort(g_idx)])

    assert scale_g_idx == pytest.approx(scale)
    assert zero_point_g_idx == pytest.approx(zero_point)
