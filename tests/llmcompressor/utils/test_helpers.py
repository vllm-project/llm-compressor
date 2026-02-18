import pytest
import torch
from compressed_tensors.offload import dispatch_model, offload_model
from transformers import (
    AutoModelForCausalLM,
    MllamaForConditionalGeneration,
)

from llmcompressor.utils import (
    DisableQuantization,
    calibration_forward_context,
    disable_cache,
    disable_lm_head,
)
from llmcompressor.utils.dev import skip_weights_download
from tests.testing_utils import requires_gpu, requires_hf_token


@pytest.mark.unit
def test_DisableQuantization():
    model = torch.nn.Linear(1, 1)
    with DisableQuantization(model):
        assert not model.quantization_enabled
    assert model.quantization_enabled


@pytest.mark.unit
def test_calibration_forward_context():
    with skip_weights_download():
        model = AutoModelForCausalLM.from_pretrained("nm-testing/tinysmokellama-3.2")
    model.config.use_cache = True
    model.train()

    with calibration_forward_context(model):
        assert not torch.is_grad_enabled()
        assert not model.config.use_cache
        assert not model.training
        assert model.lm_head.forward.__name__ == "dummy_forward"

    assert torch.is_grad_enabled()
    assert model.config.use_cache
    assert model.training
    assert model.lm_head.forward.__name__ == "forward"


@requires_gpu
@requires_hf_token
@pytest.mark.unit
@pytest.mark.parametrize(
    "model_cls,model_stub",
    [
        (MllamaForConditionalGeneration, "meta-llama/Llama-3.2-11B-Vision-Instruct"),
        (AutoModelForCausalLM, "nm-testing/tinysmokellama-3.2"),
    ],
)
def test_disable_cache(model_cls, model_stub):
    with skip_weights_download(model_cls):
        model = model_cls.from_pretrained(model_stub, device_map="cuda")
    inputs = {key: value.to(model.device) for key, value in model.dummy_inputs.items()}

    with disable_cache(model):
        output = model(**inputs)
        assert output.past_key_values is None

    output = model(**inputs)
    assert output.past_key_values is not None


@requires_gpu
@pytest.mark.parametrize("offload", ["sequential", "basic", "none"])
def test_disable_lm_head(offload):
    model = AutoModelForCausalLM.from_pretrained("nm-testing/tinysmokellama-3.2")
    if offload == "sequential":
        offload_model(model, "cuda")
    if offload == "basic":
        dispatch_model(model)
    if offload == "none":
        model = model.to("cuda")

    lm_input_device = None

    def hook(module, args):
        nonlocal lm_input_device
        lm_input_device = args[0].device

    model.lm_head.register_forward_pre_hook(hook)

    with disable_lm_head(model):
        input = {key: value.to("cuda") for key, value in model.dummy_inputs.items()}
        output = model(**input)
        assert output.logits.device == torch.device("meta")
