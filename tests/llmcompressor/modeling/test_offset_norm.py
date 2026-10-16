from types import SimpleNamespace

import pytest
import torch
from compressed_tensors.offload import offload_module, update_offload_parameter
from transformers.models.gemma2.modeling_gemma2 import Gemma2RMSNorm

from llmcompressor.modeling.offset_norm import norm_calibration_context

OFFSETS = [0.001, 1e-9, -0.03, 0.024, 0.0, 1e-14, 0.125, -0.001]


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    view = torch.int32 if tensor.dtype == torch.float32 else torch.int16
    return tensor.detach().view(view)


def _model(weight: torch.Tensor) -> torch.nn.Module:
    model = torch.nn.Module()
    model.config = SimpleNamespace()
    model.norm = Gemma2RMSNorm(weight.numel()).to(dtype=weight.dtype)
    model.norm.weight.data.copy_(weight)
    return model


def _negative_zero(dtype: torch.dtype) -> torch.Tensor:
    zeros = torch.zeros(2, dtype=dtype)
    bits = _bits(zeros).clone()
    bits[1] = bits[1] | (1 << (bits.element_size() * 8 - 1))
    return bits.view(dtype)


@pytest.mark.unit
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_empty_context_preserves_offset_bytes(dtype):
    before = torch.tensor(OFFSETS, dtype=dtype)
    model = _model(before)
    with norm_calibration_context(model):
        assert model.norm.weight.dtype == dtype
    with norm_calibration_context(model):
        pass
    assert torch.equal(_bits(model.norm.weight), _bits(before))


@pytest.mark.unit
def test_empty_context_preserves_signed_zero():
    before = _negative_zero(torch.bfloat16)
    model = _model(before)
    with norm_calibration_context(model):
        pass
    assert torch.equal(_bits(model.norm.weight), _bits(before))


@pytest.mark.unit
def test_partial_smooth_keeps_untouched_bytes():
    before = torch.tensor(OFFSETS, dtype=torch.bfloat16)
    model = _model(before)
    with norm_calibration_context(model):
        smoothed = model.norm.weight.detach().clone()
        smoothed[6] = smoothed[6] * 2
        update_offload_parameter(model.norm, "weight", smoothed)

    after = model.norm.weight.detach()
    assert after[6].item() == 1.25
    assert torch.equal(_bits(after[:6]), _bits(before[:6]))
    assert torch.equal(_bits(after[7:]), _bits(before[7:]))


@pytest.mark.unit
@pytest.mark.parametrize("offload_device", ["cpu", "disk"])
def test_offload_restore_keeps_bytes_and_smoothed_entry(offload_device, tmp_path):
    before = torch.tensor(OFFSETS, dtype=torch.bfloat16)
    model = _model(before)
    kwargs = {"onload_device": "cpu", "offload_device": offload_device}
    if offload_device == "disk":
        kwargs["offload_dir"] = str(tmp_path)
    offload_module(model.norm, **kwargs)

    with norm_calibration_context(model):
        smoothed = model.norm.weight.detach().clone()
        smoothed[6] = smoothed[6] * 2
        update_offload_parameter(model.norm, "weight", smoothed)

    after = model.norm.weight.detach()
    assert after[6].item() == 1.25
    assert torch.equal(_bits(after[:6]), _bits(before[:6]))
    assert torch.equal(_bits(after[7:]), _bits(before[7:]))


@pytest.mark.unit
def test_exception_inside_context_restores_original_bytes():
    before = torch.tensor(OFFSETS, dtype=torch.bfloat16)
    model = _model(before)
    with pytest.raises(RuntimeError, match="boom"):
        with norm_calibration_context(model):
            raise RuntimeError("boom")
    assert isinstance(model.norm, Gemma2RMSNorm)
    assert torch.equal(_bits(model.norm.weight), _bits(before))
