import torch
from compressed_tensors.offload import disable_onloading
from compressed_tensors.quantization import QuantizationMetadata

from tests.e2e.e2e_utils import run_oneshot_single
from tests.testing_utils import BaseTestConfig, requires_gpu


@requires_gpu(1)
def test_oneshot_integration():
    """
    Tests that moe_calibration_context is called within oneshot
    """
    config = BaseTestConfig(
        cadence="commit",
        model="nm-testing/tinysmokeqwen3moe",
        scheme="NVFP4",
        dataset_id="HuggingFaceH4/ultrachat_200k",
        dataset_split="train_sft",
        num_calibration_samples=1,
        max_seq_length=1,  # not enough tokens to send to all experts w/o context
    )

    model = run_oneshot_single(**config.model_dump())
    _test_qparams(model)


def _test_qparams(model: torch.nn.Module):
    all_qparam_names = QuantizationMetadata.all_qparam_names()

    with disable_onloading():
        all_qparams = [
            (module_name + "." + qparam_name, getattr(module, qparam_name))
            for module_name, module in model.named_modules()
            for qparam_name in all_qparam_names
            if hasattr(module, qparam_name)
        ]
    assert len(all_qparams) > 0, "Model does not have any qparams to test"

    for name, qparam in all_qparams:
        assert isinstance(qparam, torch.Tensor)
        assert qparam._version >= 1, f"{name} was never updated after initialization"
