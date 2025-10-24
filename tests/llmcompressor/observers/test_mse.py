import pytest
import torch
from compressed_tensors.quantization import fake_quantize
from compressed_tensors.quantization.quant_args import QuantizationArgs

from llmcompressor.observers import MovingAverageMSEObserver, Observer


@pytest.mark.parametrize(
    "strategy,symmetric,exp_loss",
    [
        ("tensor", True, 4.8103e-06),
        ("tensor", False, 1.1258e-06),
        ("channel", True, 2.5675e-06),
        ("channel", False, 2.3696e-07),
        ("group", True, 3.1282e-06),
        ("group", False, 1.3794e-07),
        ("block", True, 2.8968e-06),
        ("block", False, 5.6068e-07),
    ],
)
def test_mse_observer(strategy, symmetric, exp_loss):
    tensor = torch.arange(24).reshape((6, 4)) / 24
    num_bits = 8
    weights = QuantizationArgs(
        num_bits=num_bits,
        strategy=strategy,
        symmetric=symmetric,
        group_size=(2 if strategy == "group" else None),
        block_structure=([3, 2] if strategy == "block" else None),
        observer="mse",
    )

    observer = weights.observer
    observer = Observer.load_from_registry(observer, base_name="weight", args=weights)
    assert isinstance(observer, MovingAverageMSEObserver)

    scale, zero_point = observer(tensor)
    q_tensor = fake_quantize(tensor, scale, zero_point, weights)
    mse_loss = torch.sum((tensor - q_tensor).abs_().pow_(2)) / tensor.numel()
    assert mse_loss == pytest.approx(exp_loss, abs=1e-10)


def test_mse_observer_symmetric_scale_range():
    tensor = torch.rand(4, 4)
    tensor *= 127

    num_bits = 8
    weights = QuantizationArgs(num_bits=num_bits, symmetric=True, observer="mse")

    observer = weights.observer
    observer = Observer.load_from_registry(observer, base_name="weight", args=weights)
    scale, zero_point = observer(tensor)

    # if symmetric, max symmetric_range = abs(-128) / 255
    assert round(scale.item(), 4) <= 1.0039
    assert round(zero_point.item(), 4) == 0


def test_mse_fp4():
    module = torch.nn.Linear(6, 4)
    module.weight.data = torch.arange(24, dtype=torch.bfloat16).reshape((4, 6)) / 24

    weights = QuantizationArgs(
        num_bits=4,
        type="float",  # must be fp4
        symmetric=True,
        strategy="tensor_group",
        group_size=3,
    )

    observer = Observer.load_from_registry(
        "mse", base_name="weight", args=weights, module=module
    )

    # must compute global scale first
    with pytest.raises(ValueError):
        scale, zero_point = observer(module.weight)

    # compute qparams
    global_scale = observer.get_global_scale(module.weight)
    module.weight_global_scale = global_scale
    scale, zero_point = observer(module.weight)

    # check mse loss
    qdq_tensor = fake_quantize(
        module.weight, scale, zero_point, weights, global_scale=global_scale
    )
    assert torch.nn.functional.mse_loss(qdq_tensor, module.weight) <= 0.0015  # 0.0013
