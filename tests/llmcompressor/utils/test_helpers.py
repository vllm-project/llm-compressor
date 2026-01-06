import pytest
import torch
from transformers import (
    AutoModelForCausalLM,
    MllamaForConditionalGeneration,
)

from llmcompressor.pipelines.sequential.helpers import dispatch_for_sequential
from llmcompressor.utils import (
    ALL_TOKEN,
    DisableQuantization,
    calibration_forward_context,
    convert_to_bool,
    disable_cache,
    disable_lm_head,
    flatten_iterable,
    interpolate,
    validate_str_iterable,
)
from llmcompressor.utils.dev import dispatch_for_generation, skip_weights_download
from tests.testing_utils import requires_gpu, requires_hf_token


@pytest.mark.unit
@pytest.mark.parametrize(
    "test_list,output",
    [
        ([], []),
        ([0, 1], [0, 1]),
        ([[0, 1], [2, 3]], [0, 1, 2, 3]),
        ([[0, 1], 2, 3], [0, 1, 2, 3]),
    ],
)
def test_flatten_iterable(test_list, output):
    flattened = flatten_iterable(test_list)
    assert flattened == output


@pytest.mark.unit
@pytest.mark.parametrize(
    "test_bool,output",
    [
        (True, True),
        ("t", True),
        ("T", True),
        ("true)", True),
        ("True", True),
        (1, True),
        ("1", True),
        (False, False),
        ("f", False),
        ("F", False),
        ("false", False),
        ("False", False),
        (0, False),
        ("0", False),
    ],
)
def test_convert_to_bool(test_bool, output):
    converted = convert_to_bool(test_bool)
    assert converted == output


@pytest.mark.unit
@pytest.mark.parametrize(
    "test_list,output",
    [
        (ALL_TOKEN, ALL_TOKEN),
        (ALL_TOKEN.lower(), ALL_TOKEN),
        ([], []),
        ([0, 1], [0, 1]),
        ([[0], [1]], [0, 1]),
    ],
)
def test_validate_str_iterable(test_list, output):
    validated = validate_str_iterable(test_list, "")
    assert validated == output


@pytest.mark.unit
def test_validate_str_iterable_negative():
    with pytest.raises(ValueError):
        validate_str_iterable("will fail", "")


@pytest.mark.unit
@pytest.mark.parametrize(
    "x_cur,x0,x1,y0,y1,inter_func,out",
    [
        (0.0, 0.0, 1.0, 0.0, 5.0, "linear", 0.0),
        (0.0, 0.0, 1.0, 0.0, 5.0, "cubic", 0.0),
        (0.0, 0.0, 1.0, 0.0, 5.0, "inverse_cubic", 0.0),
        (1.0, 0.0, 1.0, 0.0, 5.0, "linear", 5.0),
        (1.0, 0.0, 1.0, 0.0, 5.0, "cubic", 5.0),
        (1.0, 0.0, 1.0, 0.0, 5.0, "inverse_cubic", 5.0),
        (0.5, 0.0, 1.0, 0.0, 5.0, "linear", 2.5),
        (0.5, 0.0, 1.0, 0.0, 5.0, "cubic", 4.375),
        (0.5, 0.0, 1.0, 0.0, 5.0, "inverse_cubic", 1.031),
    ],
)
def test_interpolate(x_cur, x0, x1, y0, y1, inter_func, out):
    interpolated = interpolate(x_cur, x0, x1, y0, y1, inter_func)
    assert abs(out - interpolated) < 0.01


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
        dispatch_for_sequential(model)
    if offload == "basic":
        dispatch_for_generation(model)
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
