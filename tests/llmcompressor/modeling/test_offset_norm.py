from types import SimpleNamespace

import pytest
import torch
from compressed_tensors.offload import offload_module, update_offload_parameter
from transformers.models.gemma2.modeling_gemma2 import Gemma2RMSNorm

from llmcompressor.modeling import offset_norm
from llmcompressor.modeling.offset_norm import (
    CalibrationOffsetNorm,
    norm_calibration_context,
)

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
    return torch.tensor([0.0, -0.0], dtype=dtype)


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
    files_before = set(tmp_path.iterdir()) if offload_device == "disk" else None

    with norm_calibration_context(model):
        if files_before is not None:
            assert set(tmp_path.iterdir()) > files_before
        smoothed = model.norm.weight.detach().clone()
        smoothed[6] = smoothed[6] * 2
        update_offload_parameter(model.norm, "weight", smoothed)

    if files_before is not None:
        assert set(tmp_path.iterdir()) == files_before
    after = model.norm.weight.detach()
    assert after[6].item() == 1.25
    assert torch.equal(_bits(after[:6]), _bits(before[:6]))
    assert torch.equal(_bits(after[7:]), _bits(before[7:]))


@pytest.mark.unit
def test_cpu_offload_accelerator_onload_persists_smooth():
    if not torch.accelerator.is_available():
        pytest.skip("no accelerator")
    onload = torch.accelerator.current_accelerator().type
    if onload == "cpu":
        pytest.skip("onload device is cpu")

    before = torch.tensor(OFFSETS, dtype=torch.bfloat16)
    model = _model(before)
    offload_module(model.norm, onload_device=onload, offload_device="cpu")

    with norm_calibration_context(model):
        smoothed = model.norm.weight.detach().clone()
        smoothed[6] = smoothed[6] * 2
        update_offload_parameter(model.norm, "weight", smoothed)

    after = model.norm.weight.detach().cpu()
    assert after[6].item() == 1.25
    assert torch.equal(_bits(after[:6]), _bits(before[:6]))
    assert torch.equal(_bits(after[7:]), _bits(before[7:]))


@pytest.mark.unit
def test_restore_failure_puts_every_norm_back(tmp_path, monkeypatch):
    before = torch.tensor(OFFSETS, dtype=torch.bfloat16)
    model = _two_disk_norms(before, str(tmp_path))
    files_before = set(tmp_path.iterdir())
    calls = {"n": 0}
    real_restore = CalibrationOffsetNorm.restore

    def fail_first(self, original):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("restore failed")
        return real_restore(self, original)

    monkeypatch.setattr(CalibrationOffsetNorm, "restore", fail_first)
    with pytest.raises(RuntimeError, match="restore failed"):
        with norm_calibration_context(model):
            pass

    assert type(model.a) is Gemma2RMSNorm
    assert type(model.b) is Gemma2RMSNorm
    assert torch.equal(_bits(model.a.weight.detach()), _bits(before))
    assert torch.equal(_bits(model.b.weight.detach()), _bits(before))
    assert set(tmp_path.iterdir()) == files_before


def _two_disk_norms(before: torch.Tensor, offload_dir: str) -> torch.nn.Module:
    model = torch.nn.Module()
    model.config = SimpleNamespace()
    model.a = Gemma2RMSNorm(before.numel()).to(dtype=before.dtype)
    model.b = Gemma2RMSNorm(before.numel()).to(dtype=before.dtype)
    model.a.weight.data.copy_(before)
    model.b.weight.data.copy_(before)
    for norm in (model.a, model.b):
        offload_module(
            norm,
            onload_device="cpu",
            offload_device="disk",
            offload_dir=offload_dir,
        )
    return model


@pytest.mark.unit
def test_release_failure_still_restores_later_norms(tmp_path, monkeypatch):
    before = torch.tensor(OFFSETS, dtype=torch.bfloat16)
    model = _two_disk_norms(before, str(tmp_path))
    files_before = set(tmp_path.iterdir())
    calls = {"n": 0}
    real_release = offset_norm._release_offload

    def fail_first(module):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("release failed")
        return real_release(module)

    monkeypatch.setattr(offset_norm, "_release_offload", fail_first)
    with pytest.raises(RuntimeError, match="release failed"):
        with norm_calibration_context(model):
            pass

    assert type(model.a) is Gemma2RMSNorm
    assert type(model.b) is Gemma2RMSNorm
    assert torch.equal(_bits(model.a.weight.detach()), _bits(before))
    assert torch.equal(_bits(model.b.weight.detach()), _bits(before))
    assert len(set(tmp_path.iterdir()) - files_before) == 1


@pytest.mark.unit
def test_offload_enter_failure_restores_mounted_norms(tmp_path, monkeypatch):
    before = torch.tensor(OFFSETS, dtype=torch.bfloat16)
    model = _two_disk_norms(before, str(tmp_path))
    files_before = set(tmp_path.iterdir())
    calls = {"n": 0}
    real_offload = offset_norm.offload_module

    def fail_second(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("offload failed")
        return real_offload(*args, **kwargs)

    monkeypatch.setattr(offset_norm, "offload_module", fail_second)
    with pytest.raises(RuntimeError, match="offload failed"):
        with norm_calibration_context(model):
            pass

    assert type(model.a) is Gemma2RMSNorm
    assert type(model.b) is Gemma2RMSNorm
    assert torch.equal(_bits(model.a.weight.detach()), _bits(before))
    assert torch.equal(_bits(model.b.weight.detach()), _bits(before))
    assert set(tmp_path.iterdir()) == files_before


@pytest.mark.unit
def test_failed_unmount_leaves_stand_in_weight(tmp_path, monkeypatch):
    before = torch.tensor(OFFSETS, dtype=torch.bfloat16)
    model = _two_disk_norms(before, str(tmp_path))
    files_before = set(tmp_path.iterdir())
    real_set = model.set_submodule

    def fail_put_back(name, module):
        if name == "a" and type(module) is Gemma2RMSNorm:
            raise RuntimeError("unmount failed")
        return real_set(name, module)

    monkeypatch.setattr(model, "set_submodule", fail_put_back)
    with pytest.raises(RuntimeError, match="unmount failed"):
        with norm_calibration_context(model):
            pass

    scale = (1.0 + before.float()).to(dtype=before.dtype)
    assert type(model.a) is CalibrationOffsetNorm
    assert torch.equal(_bits(model.a.weight.detach()), _bits(scale))
    assert type(model.b) is Gemma2RMSNorm
    assert torch.equal(_bits(model.b.weight.detach()), _bits(before))
    assert len(set(tmp_path.iterdir()) - files_before) == 1


@pytest.mark.unit
def test_exception_inside_context_restores_original_bytes():
    before = torch.tensor(OFFSETS, dtype=torch.bfloat16)
    model = _model(before)
    with pytest.raises(RuntimeError, match="boom"):
        with norm_calibration_context(model):
            raise RuntimeError("boom")
    assert isinstance(model.norm, Gemma2RMSNorm)
    assert torch.equal(_bits(model.norm.weight), _bits(before))
