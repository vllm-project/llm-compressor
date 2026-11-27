import pytest
import torch
from compressed_tensors.offload import offload_module, set_onload_device
from compressed_tensors.offload.cache import OffloadCache
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from transformers.models.gemma2.modeling_gemma2 import Gemma2RMSNorm

from llmcompressor.args.dataset_arguments import DatasetArguments
from llmcompressor.core import LifecycleCallbacks, create_session
from llmcompressor.modeling.offset_norm import NormCalibrationModule
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.pipelines.data_free import DataFreePipeline
from llmcompressor.pipelines.sequential.restructure import RestructureHandler
from llmcompressor.utils.dev import get_main_device


@pytest.mark.parametrize("offload_device", ["cpu", "disk"])
def test_datafree_materializes_before_calibration_start(
    monkeypatch, tmp_path, offload_device
):
    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    expected = model[0].weight.detach().clone()
    offload_module(
        model[0],
        onload_device="cpu",
        offload_device=offload_device,
        **({"offload_dir": tmp_path} if offload_device == "disk" else {}),
    )
    set_onload_device(model, get_main_device())
    handler = RestructureHandler(model)
    handler.convert_model_to_meta()
    handler.apply_restructuring()
    calibration_start = LifecycleCallbacks.calibration_start
    calls = []

    def check_real_weights():
        torch.testing.assert_close(model[0].weight.cpu(), expected, rtol=0, atol=0)
        calls.append("calibration_start")
        calibration_start()

    monkeypatch.setattr(LifecycleCallbacks, "calibration_start", check_real_weights)
    recipe = QuantizationModifier(
        config_groups={
            "test": QuantizationScheme(
                targets=["Linear"],
                weights=QuantizationArgs(num_bits=8, strategy="channel"),
            )
        }
    )
    with create_session() as session:
        session.initialize(model=model, recipe=recipe, start=-1)
        assert model[0].weight.is_meta
        DataFreePipeline()(model, None, DatasetArguments(), handler)
        session.finalize()

    assert calls == ["calibration_start"]
    assert not hasattr(model[0], "weight_observer")
    assert all(not tensor.is_meta for tensor in model.state_dict().values())


@pytest.mark.parametrize("offload_device", ["cpu", "disk"])
def test_datafree_restores_norm_after_calibration_before_offload(
    monkeypatch, tmp_path, offload_device
):
    model = torch.nn.Sequential(Gemma2RMSNorm(4, eps=1e-6))
    model[0].weight = torch.nn.Parameter(torch.arange(4.0))
    offload_module(
        model[0],
        onload_device="cpu",
        offload_device=offload_device,
        **({"offload_dir": tmp_path} if offload_device == "disk" else {}),
    )
    device = get_main_device()
    set_onload_device(model, device)
    handler = RestructureHandler(model)
    handler.convert_model_to_meta()
    handler.apply_restructuring()
    source = handler.mapping[model[0]]["source"]
    calibration_end = LifecycleCallbacks.calibration_end
    calls = []

    def update_calibrated_norm():
        calibration_end()
        assert isinstance(model[0], NormCalibrationModule)
        torch.testing.assert_close(model[0].weight.cpu(), torch.arange(4.0) + 1)
        model[0].weight = torch.nn.Parameter(model[0].weight + 2)
        calls.append("calibration_end")

    monkeypatch.setattr(LifecycleCallbacks, "calibration_end", update_calibrated_norm)
    with create_session() as session:
        session.initialize(model=model, start=-1)
        DataFreePipeline()(model, None, DatasetArguments(), handler)
        assert model[0] is source
        assert not handler.mapping
        session.finalize()

    assert calls == ["calibration_end"]
    assert isinstance(source, Gemma2RMSNorm)
    assert isinstance(source._parameters, OffloadCache)
    assert str(source._parameters.offload_device) == offload_device
    assert source._parameters.onload_device == device
    if offload_device == "disk":
        assert source._parameters.offload_dir == tmp_path
    torch.testing.assert_close(source.weight.cpu(), torch.arange(4.0) + 2)
