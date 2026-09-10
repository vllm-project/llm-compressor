import pytest
import torch
from compressed_tensors.quantization import QuantizationArgs

from llmcompressor.observers import MixFP4Observer, Observer


def _mixfp4_args(**overrides):
    kwargs = dict(
        num_bits=4,
        type="float",
        strategy="tensor_group",
        symmetric=True,
        dynamic=False,
        group_size=16,
        observer="mixfp4",
        scale_dtype=torch.float8_e4m3fn,
        zp_dtype=torch.float8_e4m3fn,
    )
    kwargs.update(overrides)
    return QuantizationArgs(**kwargs)


def test_mixfp4_observer_registry_alias():
    args = _mixfp4_args()
    observer = Observer.load_from_registry("mixfp4", base_name="weight", args=args)
    legacy = Observer.load_from_registry(
        "mixed_fp4_int4", base_name="weight", args=args
    )

    assert isinstance(observer, MixFP4Observer)
    assert isinstance(legacy, MixFP4Observer)


def test_mixfp4_requires_global_scale():
    args = _mixfp4_args()
    module = torch.nn.Linear(16, 1, bias=False)
    observer = Observer.load_from_registry(
        "mixfp4", base_name="weight", args=args, module=module
    )

    with pytest.raises(ValueError, match="global scale"):
        observer(module.weight)


def test_mixfp4_packs_int4_flag_in_scale_sign_bit():
    int4_group = torch.tensor(
        [-7, -6, -5, -4, -3, -2, -1, 0, 0, 1, 2, 3, 4, 5, 6, 7],
        dtype=torch.float32,
    )
    fp4_group = torch.tensor(
        [-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0, 0.5, 1, 1.5, 2, 3, 4, 6],
        dtype=torch.float32,
    )
    weight = torch.stack([int4_group, fp4_group])

    args = _mixfp4_args()
    module = torch.nn.Linear(16, 2, bias=False)
    module.weight.data.copy_(weight)
    observer = Observer.load_from_registry(
        "mixfp4", base_name="weight", args=args, module=module
    )

    module.weight_global_scale = observer.get_global_scale(module.weight)
    scale, zero_point = observer(module.weight)
    raw_scale = scale.view(torch.uint8)

    assert scale.dtype == torch.float8_e4m3fn
    assert zero_point.dtype == torch.float8_e4m3fn
    assert scale.shape == (2, 1)
    assert (raw_scale[0, 0] & 0x80) != 0
    assert (raw_scale[1, 0] & 0x80) == 0


def test_mixfp4_zero_group_does_not_encode_negative_zero():
    args = _mixfp4_args()
    module = torch.nn.Linear(16, 1, bias=False)
    module.weight.data.zero_()
    observer = Observer.load_from_registry(
        "mixfp4", base_name="weight", args=args, module=module
    )

    module.weight_global_scale = observer.get_global_scale(module.weight)
    scale, _ = observer(module.weight)

    assert scale.view(torch.uint8).item() == 0


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"strategy": "group"}, "tensor_group"),
        ({"group_size": 8}, "group_size=16"),
        ({"num_bits": 8}, "num_bits=4"),
        ({"type": "int"}, "floating-point"),
        ({"symmetric": False}, "symmetric"),
        ({"scale_dtype": torch.bfloat16}, "float8_e4m3fn"),
    ],
)
def test_mixfp4_rejects_noncanonical_args(overrides, error):
    args = _mixfp4_args(**overrides)

    with pytest.raises(ValueError, match=error):
        Observer.load_from_registry("mixfp4", base_name="weight", args=args)


def test_mixfp4_rejects_activation_observer():
    args = _mixfp4_args()

    with pytest.raises(ValueError, match="weight"):
        Observer.load_from_registry("mixfp4", base_name="input", args=args)


def test_mixfp4_observer_scales_are_safe_for_fake_quantize():
    from compressed_tensors.quantization import fake_quantize

    int4_group = torch.tensor(
        [-7, -6, -5, -4, -3, -2, -1, 0, 0, 1, 2, 3, 4, 5, 6, 7],
        dtype=torch.float32,
    )
    fp4_group = torch.tensor(
        [-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0, 0.5, 1, 1.5, 2, 3, 4, 6],
        dtype=torch.float32,
    )
    weight = torch.stack([int4_group, fp4_group])

    args = _mixfp4_args()
    module = torch.nn.Linear(16, 2, bias=False)
    module.weight.data.copy_(weight)
    observer = Observer.load_from_registry(
        "mixfp4", base_name="weight", args=args, module=module
    )

    module.weight_global_scale = observer.get_global_scale(module.weight)
    scale, zero_point = observer(module.weight)
    quantized = fake_quantize(
        module.weight,
        scale,
        zero_point,
        args,
        global_scale=module.weight_global_scale,
    )

    assert torch.allclose(quantized, weight)
