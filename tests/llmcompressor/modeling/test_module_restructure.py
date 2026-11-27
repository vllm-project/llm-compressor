import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    initialize_module_for_quantization,
)

from llmcompressor.modeling.module_restructure import transfer_module_state


def test_transfer_moves_direct_storage_and_preserves_initialized_module():
    source = torch.nn.Linear(4, 4)
    source.register_buffer("position", torch.arange(4.0), persistent=False)
    source.constant = torch.arange(8.0)
    source_weight, source_bias, source_position = (
        source.weight,
        source.bias,
        source.position,
    )
    expected = source.weight.detach().clone()
    target = torch.nn.Linear(4, 4, device="meta")
    target.register_buffer("position", torch.empty(4, device="meta"), persistent=False)
    target.constant = torch.full((8,), 7.0)
    constant = target.constant
    weight = target.weight
    bias, position = target.bias, target.position
    weight.calibration_tag = object()
    calibration_tag = weight.calibration_tag
    weight.requires_grad_(False)
    initialize_module_for_quantization(
        target,
        QuantizationScheme(
            targets=["Linear"],
            weights=QuantizationArgs(num_bits=4, strategy="channel"),
            input_activations=QuantizationArgs(num_bits=8, strategy="tensor"),
        ),
    )
    qparams = {
        name: tensor
        for name, tensor in target.named_parameters()
        if name not in {"weight", "bias"}
    }
    assert all(tensor.is_meta for tensor in qparams.values())
    calls = []
    target.register_forward_pre_hook(lambda *_: calls.append("target"))

    transfer_module_state(target, source, torch.device("cpu"))

    assert target.weight is weight
    assert target.bias is bias
    assert target.position is position
    assert target.weight.calibration_tag is calibration_tag
    assert not source_weight.is_meta
    assert not source_bias.is_meta
    assert not source_position.is_meta
    assert target.weight.data_ptr() == source_weight.data_ptr()
    assert target.bias.data_ptr() == source_bias.data_ptr()
    assert not target.weight.requires_grad
    assert target.position.data_ptr() == source_position.data_ptr()
    assert target.constant is constant
    torch.testing.assert_close(target.constant, torch.full((8,), 7.0))
    for name, placeholder in qparams.items():
        tensor = target.get_parameter(name)
        assert tensor is placeholder
        assert isinstance(tensor, torch.nn.Parameter)
        assert tensor.device.type == "cpu"
        torch.testing.assert_close(
            tensor, torch.full_like(tensor, 1 if name.endswith("_scale") else 0)
        )
    assert not source._parameters and not source._buffers
    torch.testing.assert_close(target.weight, expected, rtol=0, atol=0)
    target(torch.ones(1, 4))
    assert calls == ["target"]


def test_transfer_leaves_child_modules_to_the_caller():
    source = torch.nn.Sequential(torch.nn.Linear(4, 4))
    target = torch.nn.Sequential(torch.nn.Linear(4, 4, device="meta"))
    expected = source[0].weight.detach().clone()

    transfer_module_state(target, source, torch.device("cpu"))
    assert target[0].weight.is_meta
    assert source[0].weight is not None

    transfer_module_state(target[0], source[0], torch.device("cpu"))
    torch.testing.assert_close(target[0].weight, expected)
    assert not source[0]._parameters


def test_transfer_preserves_materialized_state_alongside_meta_state():
    source = torch.nn.Linear(4, 4)
    target = torch.nn.Linear(4, 4, device="meta")
    target.weight = torch.nn.Parameter(torch.full((4, 4), 7.0))
    weight_pointer = target.weight.data_ptr()
    source_bias = source.bias

    transfer_module_state(target, source, torch.device("cpu"))

    assert target.weight.data_ptr() == weight_pointer
    torch.testing.assert_close(target.weight, torch.full((4, 4), 7.0))
    assert target.bias.data_ptr() == source_bias.data_ptr()
    assert not source._parameters
