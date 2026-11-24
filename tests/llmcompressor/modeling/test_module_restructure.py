import torch

from llmcompressor.modeling.module_restructure import (
    clear_module_state,
    transfer_module_state,
)
from llmcompressor.utils.pytorch.meta import copy_module_to_meta


def test_transfer_adopts_storage_and_preserves_initialized_targets():
    source = torch.nn.Linear(4, 4)
    source.register_buffer("position", torch.arange(4.0), persistent=False)
    source.position_alias = source.position
    source.constant = torch.arange(8.0)
    expected = source.weight.detach().clone()
    target = copy_module_to_meta(source)
    weight, position, constant = target.weight, target.position, target.constant
    marker = object()
    weight.marker = marker
    weight.requires_grad_(False)
    target.weight_scale = torch.nn.Parameter(torch.tensor([7.0]), requires_grad=False)
    target.input_scale = torch.nn.Parameter(
        torch.empty(1, device="meta"), requires_grad=False
    )
    target.register_buffer(
        "weight_zero_point", torch.empty(1, device="meta", dtype=torch.int32)
    )
    scale_pointer = target.weight_scale.data_ptr()
    calls = []
    target.register_forward_pre_hook(lambda *_: calls.append("target"))

    transfer_module_state(target, source, torch.device("cpu"))

    assert target.weight is weight
    assert target.weight.data_ptr() == source.weight.data_ptr()
    assert target.bias.data_ptr() == source.bias.data_ptr()
    assert target.weight.marker is marker
    assert not target.weight.requires_grad
    assert source.weight.requires_grad
    assert target.position is position
    assert target.position_alias is position
    assert position.data_ptr() == source.position.data_ptr()
    assert target.constant is constant
    assert constant.data_ptr() == source.constant.data_ptr()
    assert target.weight_scale.data_ptr() == scale_pointer
    torch.testing.assert_close(target.weight_scale, torch.tensor([7.0]))
    torch.testing.assert_close(target.input_scale, torch.ones(1))
    assert target.weight_zero_point.count_nonzero() == 0

    clear_module_state(source)
    torch.testing.assert_close(target.weight, expected, rtol=0, atol=0)
    target(torch.ones(1, 4))
    assert calls == ["target"]


def test_transfer_preserves_materialized_state_alongside_meta_state():
    source = torch.nn.Linear(4, 4)
    target = torch.nn.Linear(4, 4, device="meta")
    target.weight = torch.nn.Parameter(torch.full((4, 4), 7.0))
    weight_pointer = target.weight.data_ptr()

    transfer_module_state(target, source, torch.device("cpu"))

    assert target.weight.data_ptr() == weight_pointer
    torch.testing.assert_close(target.weight, torch.full((4, 4), 7.0))
    assert target.bias.data_ptr() == source.bias.data_ptr()


def test_transfer_converts_dtype_when_source_storage_cannot_be_shared():
    source = torch.nn.Linear(4, 4, bias=False)
    target = torch.nn.Linear(4, 4, bias=False, device="meta", dtype=torch.float16)
    weight = target.weight

    transfer_module_state(target, source, torch.device("cpu"))

    assert target.weight is weight
    assert target.weight.data_ptr() != source.weight.data_ptr()
    torch.testing.assert_close(target.weight, source.weight.half(), rtol=0, atol=0)


def test_transfer_adopts_inference_storage_outside_inference_mode():
    with torch.inference_mode():
        source = torch.nn.Linear(4, 4)
    target = torch.nn.Linear(4, 4, device="meta")
    weight = target.weight

    transfer_module_state(target, source, torch.device("cpu"))

    assert target.weight is weight
    assert target.weight.requires_grad
    assert not target.weight.is_inference()
    assert target.weight.data_ptr() == source.weight.data_ptr()
    torch.testing.assert_close(target.weight, source.weight, rtol=0, atol=0)
