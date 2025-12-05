import pytest
import torch
from transformers import (
    AutoModelForCausalLM,
    MllamaForConditionalGeneration,
    PretrainedConfig,
    PreTrainedModel,
)

from llmcompressor.utils import (
    DisableQuantization,
    calibration_forward_context,
    disable_cache,
)
from llmcompressor.utils.dev import skip_weights_download
from tests.testing_utils import requires_gpu


@pytest.mark.unit
def test_DisableQuantization():
    model = torch.nn.Linear(1, 1)
    with DisableQuantization(model):
        assert not model.quantization_enabled
    assert model.quantization_enabled


@pytest.mark.unit
def test_calibration_forward_context():
    class DummyModel(PreTrainedModel):
        config_class = PretrainedConfig

    model = DummyModel(PretrainedConfig())
    model.config.use_cache = True
    model.train()

    with calibration_forward_context(model):
        assert not torch.is_grad_enabled()
        assert not model.config.use_cache
        assert not model.training
    assert torch.is_grad_enabled()
    assert model.config.use_cache
    assert model.training


@requires_gpu
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
