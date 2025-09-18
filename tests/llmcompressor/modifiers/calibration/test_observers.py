import pytest
import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    initialize_module_for_quantization,
)

from llmcompressor.modifiers.quantization.calibration import (
    calibrate_input_hook,
    initialize_observer,
    update_weight_global_scale,
    update_weight_zp_scale,
)


@pytest.mark.parametrize(
    "shape,group_size,actorder",
    [
        ((1, 1), None, False),
        ((1, 1), 128, False),
        ((1, 1), 128, True),
        ((64, 64), None, False),
        ((64, 64), 128, False),
        ((64, 64), 128, True),
        ((1792, 4096), None, False),
        ((1792, 4096), 128, False),
        ((1792, 4096), 128, True),
        ((3420, 64), None, False),
        ((3420, 64), 128, False),
        ((3420, 64), 128, True),
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
        g_idx = getattr(module, "g_idx", None)
        updated_scale, updated_zero_point = observer(value, g_idx=g_idx)

        assert_alike(updated_scale, getattr(module, f"{location}_scale"))
        assert_alike(updated_zero_point, getattr(module, f"{location}_zero_point"))


def assert_alike(a, b):
    assert a.dtype == b.dtype
    assert a.shape == b.shape


@pytest.mark.parametrize(
    "args,exp_min_val,exp_max_val,exp_quant,exp_loss",
    [
        (
            QuantizationArgs(
                num_bits=4,
                type="int",
                symmetric=True,
                strategy="tensor",  # equivalent to token
                observer="minmax",
            ),
            {"default": torch.tensor(0.0)},
            {"default": torch.tensor(23.0)},
            torch.tensor(
                [
                    [0.0000, 0.0000, 3.0625, 3.0625, 3.0625, 6.1250],
                    [6.1250, 6.1250, 9.1875, 9.1875, 9.1875, 12.2500],
                    [12.2500, 12.2500, 15.3125, 15.3125, 15.3125, 18.3750],
                    [18.3750, 18.3750, 21.5000, 21.5000, 21.5000, 21.5000],
                ],
                dtype=torch.bfloat16,
            ),
            0.85,
        ),
        (
            QuantizationArgs(
                num_bits=4,
                type="int",
                symmetric=True,
                strategy="channel",
                observer="minmax",
            ),
            {"default": torch.tensor([[0], [6], [12], [18]])},
            {"default": torch.tensor([[5], [11], [17], [23]])},
            torch.tensor(
                [
                    [0.0000, 1.3359, 2.0000, 2.6719, 4.0000, 4.6875],
                    [5.8750, 7.3438, 7.3438, 8.8125, 10.2500, 10.2500],
                    [11.3125, 13.6250, 13.6250, 15.8750, 15.8750, 15.8750],
                    [18.3750, 18.3750, 21.5000, 21.5000, 21.5000, 21.5000],
                ],
                dtype=torch.bfloat16,
            ),
            0.45,
        ),
        (
            QuantizationArgs(
                num_bits=4,
                type="int",
                symmetric=True,
                strategy="group",
                group_size=3,
                observer="minmax",
            ),
            {
                "default": torch.tensor([[0], [6], [12], [18]]),
                1: torch.tensor([[3], [9], [15], [21]]),
            },
            {
                "default": torch.tensor([[2], [8], [14], [20]]),
                1: torch.tensor([[5], [11], [17], [23]]),
            },
            torch.tensor(
                [
                    [0.0000, 1.0703, 1.8750, 2.6719, 4.0000, 4.6875],
                    [6.4375, 7.5000, 7.5000, 8.8125, 10.2500, 10.2500],
                    [11.1875, 13.0625, 13.0625, 15.8750, 15.8750, 15.8750],
                    [18.7500, 18.7500, 18.7500, 21.5000, 21.5000, 21.5000],
                ],
            ),
            0.45,
        ),
        (
            QuantizationArgs(
                num_bits=4,
                type="float",  # tensor group requires FP4
                symmetric=True,
                strategy="tensor_group",  # requires float4
                group_size=3,
                observer="minmax",
            ),
            {
                "default": torch.tensor([[0], [6], [12], [18]]),
                1: torch.tensor([[3], [9], [15], [21]]),
            },
            {
                "default": torch.tensor([[2], [8], [14], [20]]),
                1: torch.tensor([[5], [11], [17], [23]]),
            },
            torch.tensor(
                [
                    [0.0000, 1.0234, 2.0469, 3.2812, 3.2812, 4.9375],
                    [5.4688, 8.1875, 8.1875, 10.6875, 10.6875, 10.6875],
                    [9.8750, 14.7500, 14.7500, 16.3750, 16.3750, 16.3750],
                    [19.7500, 19.7500, 19.7500, 23.0000, 23.0000, 23.0000],
                ],
            ),
            1.1,
        ),
        (
            QuantizationArgs(
                num_bits=4,
                type="int",
                symmetric=True,
                strategy="block",
                block_structure=[2, 3],
                observer="minmax",
            ),
            {
                "block_0_0": torch.tensor([[0]]),
                "block_0_1": torch.tensor([[3]]),
                "block_1_0": torch.tensor([[12]]),
                "block_1_1": torch.tensor([[15]]),
            },
            {
                "block_0_0": torch.tensor([[8]]),
                "block_0_1": torch.tensor([[11]]),
                "block_1_0": torch.tensor([[20]]),
                "block_1_1": torch.tensor([[23]]),
            },
            torch.tensor(
                [
                    [0.0000, 1.0703, 2.1406, 2.9375, 4.4062, 4.4062],
                    [6.4375, 7.5000, 7.5000, 8.8125, 10.2500, 10.2500],
                    [10.6875, 13.3750, 13.3750, 15.3125, 15.3125, 18.3750],
                    [18.7500, 18.7500, 18.7500, 21.5000, 21.5000, 21.5000],
                ],
            ),
            0.5,
        ),
        (
            QuantizationArgs(
                num_bits=4,
                type="int",
                symmetric=True,
                strategy="token",  # equivalent to tensor
                observer="minmax",
            ),
            {"default": torch.tensor(0.0)},
            {"default": torch.tensor(23.0)},
            torch.tensor(
                [
                    [0.0000, 0.0000, 3.0625, 3.0625, 3.0625, 6.1250],
                    [6.1250, 6.1250, 9.1875, 9.1875, 9.1875, 12.2500],
                    [12.2500, 12.2500, 15.3125, 15.3125, 15.3125, 18.3750],
                    [18.3750, 18.3750, 21.5000, 21.5000, 21.5000, 21.5000],
                ],
                dtype=torch.bfloat16,
            ),
            0.85,
        ),
    ],
)
def test_static_weight_quantization(
    args, exp_min_val, exp_max_val, exp_quant, exp_loss
):
    """
    weight = tensor([[ 0,  1,  2,  3,  4,  5],
                     [ 6,  7,  8,  9, 10, 11],
                     [12, 13, 14, 15, 16, 17],
                     [18, 19, 20, 21, 22, 23]])
    """
    # set up weight
    input_size, output_size = 6, 4
    linear = torch.nn.Linear(input_size, output_size, bias=False)
    linear.weight.data = torch.arange(
        input_size * output_size, dtype=torch.bfloat16
    ).reshape(output_size, input_size)

    # initialize quantization parameters
    scheme = QuantizationScheme(targets=[], weights=args)
    initialize_module_for_quantization(linear, scheme)
    assert getattr(linear, "quantization_scheme") is scheme

    # calibrate quantization parameters
    initialize_observer(linear, "weight")
    update_weight_global_scale(linear)
    update_weight_zp_scale(linear)

    observer = getattr(linear, "weight_observer")
    assert (
        observer.min_val.keys()
        == observer.max_val.keys()
        == exp_min_val.keys()
        == exp_max_val.keys()
    )
    for key in observer.min_val.keys():
        assert torch.equal(observer.min_val[key], exp_min_val[key])
        assert torch.equal(observer.max_val[key], exp_max_val[key])

    # forward pass
    input = torch.eye(input_size, dtype=torch.bfloat16)
    output = linear(input)

    assert torch.allclose(output.T, exp_quant.to(output.dtype))
    assert torch.nn.functional.mse_loss(output.T, linear.weight) <= exp_loss


@pytest.mark.parametrize(
    "args,exp_min_val,exp_max_val,exp_quant,exp_loss",
    [
        (
            QuantizationArgs(
                num_bits=4,
                type="int",
                symmetric=True,
                strategy="tensor",  # equivalent to token
                observer="minmax",
            ),
            {"default": torch.tensor(0.0)},
            {"default": torch.tensor(5.0)},
            torch.tensor([[0.0000, 1.3359, 2.0000, 2.6719, 4.0000, 4.6875]]),
            0.06,
        ),
        (
            QuantizationArgs(
                num_bits=4,
                type="int",
                symmetric=True,
                strategy="token",  # equivalent to tensor
                observer="minmax",
            ),
            {"default": torch.tensor(0.0)},
            {"default": torch.tensor(5.0)},
            torch.tensor([[0.0000, 1.3359, 2.0000, 2.6719, 4.0000, 4.6875]]),
            0.06,
        ),
        # channel is not supported, but is in principle equivalent to token/tensor
        (
            QuantizationArgs(
                num_bits=4,
                type="int",
                symmetric=True,
                strategy="group",
                group_size=3,
                observer="minmax",
            ),
            {
                "default": torch.tensor([[0]]),
                1: torch.tensor([[3]]),
            },
            {
                "default": torch.tensor([[2]]),
                1: torch.tensor([[5]]),
            },
            torch.tensor([[0.0000, 1.0703, 1.8750, 2.6719, 4.0000, 4.6875]]),
            0.04,
        ),
        (
            QuantizationArgs(
                num_bits=4,
                type="float",  # tensor group requires FP4
                symmetric=True,
                strategy="tensor_group",
                group_size=3,
                observer="minmax",
            ),
            {
                "default": torch.tensor([[0]]),
                1: torch.tensor([[3]]),
            },
            {
                "default": torch.tensor([[2]]),
                1: torch.tensor([[5]]),
            },
            torch.tensor([[0.0000, 0.9766, 1.9531, 3.3125, 3.3125, 4.9688]]),
            0.1,
        ),
        # block is not supported, but is in principle similar to group
    ],
)
def test_static_activation_quantization(
    args, exp_min_val, exp_max_val, exp_quant, exp_loss
):
    """
    input = tensor([[ 0,  1,  2,  3,  4,  5]])
    """
    # set up activation (and identity weight)
    input_size = 6
    input = torch.arange(input_size, dtype=torch.bfloat16).unsqueeze(0)
    linear = torch.nn.Linear(input_size, input_size, bias=False)
    linear.weight.data = torch.eye(input_size, dtype=torch.bfloat16)

    # initialize quantization parameters
    scheme = QuantizationScheme(targets=[], input_activations=args)
    initialize_module_for_quantization(linear, scheme)
    assert getattr(linear, "quantization_scheme") is scheme

    # calibrate quantization parameters
    initialize_observer(linear, "input")
    linear.register_forward_pre_hook(calibrate_input_hook)

    # calibration forward pass
    output = linear(input)

    # check calibration
    observer = getattr(linear, "input_observer")
    assert (
        observer.min_val.keys()
        == observer.max_val.keys()
        == exp_min_val.keys()
        == exp_max_val.keys()
    )
    for key in observer.min_val.keys():
        assert torch.equal(observer.min_val[key], exp_min_val[key])
        assert torch.equal(observer.max_val[key], exp_max_val[key])

    # check forward pass
    assert torch.allclose(output, exp_quant.to(output.dtype))
    assert torch.nn.functional.mse_loss(output, input) <= exp_loss
