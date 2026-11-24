import pytest
import torch
from compressed_tensors.offload import offload_module, set_onload_device
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme

from llmcompressor.args.dataset_arguments import DatasetArguments
from llmcompressor.core import LifecycleCallbacks, create_session
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
        handler.finish()
        session.finalize()

    assert calls == ["calibration_start"]
    assert not hasattr(model[0], "weight_observer")
    assert all(not tensor.is_meta for tensor in model.state_dict().values())
