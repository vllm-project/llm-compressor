import pytest
import torch
from compressed_tensors.quantization import QuantizationStrategy, fake_quantize
from compressed_tensors.quantization.quant_args import QuantizationArgs
from compressed_tensors.utils.impl_backend import ImplBackend

from llmcompressor.observers import MovingAverageMSEObserver, Observer
from llmcompressor.observers.helpers import flatten_for_calibration
from llmcompressor.observers.mse import MemorylessMSEObserver, NVFP4ExpandedMSEObserver


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

    qparams = observer(tensor).get_qparams()
    scale, zero_point = qparams["scale"], qparams["zero_point"]
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
    qparams = observer(tensor).get_qparams()
    scale, zero_point = qparams["scale"], qparams["zero_point"]

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

    observer = Observer.load_from_registry("mse", base_name="weight", args=weights)

    qparams = observer(module.weight).get_qparams()
    scale, zero_point, global_scale = (
        qparams["scale"],
        qparams["zero_point"],
        qparams["global_scale"],
    )

    # check mse loss
    qdq_tensor = fake_quantize(
        module.weight, scale, zero_point, weights, global_scale=global_scale
    )
    assert torch.nn.functional.mse_loss(qdq_tensor, module.weight) <= 0.0015  # 0.0013


@pytest.mark.parametrize(
    "num_bits,quant_type,strategy,group_size,block_structure",
    [
        (4, "int", QuantizationStrategy.TENSOR, None, None),
        (4, "int", QuantizationStrategy.CHANNEL, None, None),
        (4, "int", QuantizationStrategy.GROUP, 512, None),
        (4, "int", QuantizationStrategy.TENSOR_GROUP, 512, None),
        (4, "float", QuantizationStrategy.GROUP, 512, None),
        (4, "float", QuantizationStrategy.TENSOR_GROUP, 512, None),
        (8, "float", QuantizationStrategy.BLOCK, None, [2, 512]),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Triton")
def test_mse_triton_matches_eager_when_tile_fits_group(
    num_bits, quant_type, strategy, group_size, block_structure
):
    """A full buffer preserves eager choices when each group spans a tile."""
    if quant_type == "float" and num_bits == 8:
        major, _ = torch.cuda.get_device_capability()
        if major < 9:
            pytest.skip("FP8 Triton QDQ requires SM90+")
    args = QuantizationArgs(
        num_bits=num_bits,
        type=quant_type,
        symmetric=True,
        strategy=strategy,
        group_size=group_size,
        block_structure=block_structure,
    )
    token_args = args.model_copy(update={"strategy": QuantizationStrategy.TOKEN})
    torch.manual_seed(0)
    observed = flatten_for_calibration(
        torch.randn(8, 1024, device="cuda"), "weight", args
    )
    search_args = (
        observed,
        args,
        token_args,
        0.5,
        5,
        100.0,
        2.4,
        1.0,
        1.0,
    )
    eager = ImplBackend.call("_grid_search_mse", *search_args)
    triton = ImplBackend.call("_grid_search_mse_triton", *search_args)
    assert torch.equal(eager[0], triton[0])
    assert torch.equal(eager[1], triton[1])


def test_mse_triton_error_buffer_defaults():
    args = QuantizationArgs(num_bits=8, symmetric=True, observer="mse")
    observer = MovingAverageMSEObserver(base_name="weight", args=args)
    assert observer.triton_error_buffer == 0.30

    fp4_args = QuantizationArgs(num_bits=4, type="float", symmetric=True)
    assert (
        MemorylessMSEObserver(base_name="weight", args=fp4_args).triton_error_buffer
        == 1.00
    )
    assert (
        MovingAverageMSEObserver(base_name="weight", args=fp4_args).triton_error_buffer
        == 1.00
    )

    nvfp4 = NVFP4ExpandedMSEObserver(base_name="weight", args=args)
    assert nvfp4.triton_error_buffer == 1.00


@pytest.mark.parametrize(
    "observer_cls", [MemorylessMSEObserver, MovingAverageMSEObserver]
)
def test_mse_observer_rejects_expand_below_one(observer_cls):
    # `expand` scales the initial search range (min/max * expand) and the grid
    # search can only shrink it further, so an expand < 1.0 starts below the
    # observed range and can never cover the data. Every MSE observer must
    # reject it rather than silently produce a degenerate range.
    args = QuantizationArgs(num_bits=8, symmetric=True, observer="mse")
    with pytest.raises(ValueError, match="expand value must be at least 1.0"):
        observer_cls(base_name="weight", args=args, expand=0.5)

    # expand >= 1.0 is accepted.
    observer_cls(base_name="weight", args=args, expand=1.0)
