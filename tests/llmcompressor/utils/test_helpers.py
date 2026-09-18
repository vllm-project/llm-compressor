import builtins
import pickle
import sys

import pytest
import torch
from compressed_tensors.offload import dispatch_model, set_onload_device
from transformers import (
    AutoModelForCausalLM,
    MllamaForConditionalGeneration,
)

from llmcompressor.utils import (
    DisableQuantization,
    calibration_forward_context,
    disable_cache,
    disable_lm_head,
    import_from_path,
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
    model = AutoModelForCausalLM.from_pretrained("nm-testing/tinysmokeqwen3")
    if offload == "sequential":
        set_onload_device(model, "cuda")
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


PREPROCESS_SOURCE = """
def preprocess(example):
    return example
"""


@pytest.fixture
def preprocess_file(tmp_path):
    file_path = tmp_path / "custom_preprocess.py"
    file_path.write_text(PREPROCESS_SOURCE)
    return file_path


@pytest.mark.unit
def test_import_from_path_file_with_suffix(preprocess_file):
    # The form documented in the docstring and in
    # `TextGenerationDataset.preprocessing_func`: "/path/to/file.py:func_name".
    func = import_from_path(f"{preprocess_file}:preprocess")

    assert func.__name__ == "preprocess"
    assert func({"a": 1}) == {"a": 1}


@pytest.mark.unit
def test_import_from_path_file_without_suffix(preprocess_file):
    without_suffix = preprocess_file.with_suffix("")
    func = import_from_path(f"{without_suffix}:preprocess")

    assert func.__name__ == "preprocess"


@pytest.mark.unit
def test_import_from_path_dotted_module_containing_py_segment():
    # The module path must not be truncated at a package that happens to start with
    # "py", such as `llmcompressor.pytorch`.
    func = import_from_path("llmcompressor.pytorch.utils.helpers:get_quantized_layers")

    assert func.__name__ == "get_quantized_layers"


@pytest.mark.unit
def test_import_from_path_missing_module():
    with pytest.raises(ImportError, match="Cannot find module with path"):
        import_from_path("llmcompressor.does.not.exist:some_name")


@pytest.mark.unit
def test_import_from_path_missing_attribute(preprocess_file):
    with pytest.raises(AttributeError, match="Cannot find not_there in"):
        import_from_path(f"{preprocess_file}:not_there")


@pytest.mark.unit
def test_import_from_path_loads_the_file_once(preprocess_file, tmp_path):
    counter = tmp_path / "counting_module.py"
    counter.write_text(
        "import builtins\n"
        "builtins._llmcompressor_test_loads = "
        "getattr(builtins, '_llmcompressor_test_loads', 0) + 1\n"
        "def preprocess(example):\n    return example\n"
    )

    for _ in range(3):
        import_from_path(f"{counter}:preprocess")

    assert builtins._llmcompressor_test_loads == 1


@pytest.mark.unit
def test_import_from_path_result_is_picklable(preprocess_file):
    # `preprocessing_func` is handed to `dataset.map(num_proc=...)`, which pickles it.
    # That needs the loaded module registered in sys.modules.
    func = import_from_path(f"{preprocess_file}:preprocess")

    assert pickle.loads(pickle.dumps(func)) is func


@pytest.mark.unit
def test_import_from_path_keeps_same_named_files_apart(tmp_path):
    for name in ("a", "b"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "custom_preprocess.py").write_text(
            f"def preprocess(example):\n    return '{name}'\n"
        )

    first = import_from_path(f"{tmp_path / 'a' / 'custom_preprocess.py'}:preprocess")
    second = import_from_path(f"{tmp_path / 'b' / 'custom_preprocess.py'}:preprocess")

    assert first({}) == "a"
    assert second({}) == "b"
    assert first.__module__ != second.__module__


@pytest.mark.unit
def test_import_from_path_does_not_leave_a_broken_module_registered(tmp_path):
    broken = tmp_path / "broken_preprocess.py"
    broken.write_text("raise RuntimeError('boom')\n")
    before = set(sys.modules)

    with pytest.raises(RuntimeError, match="boom"):
        import_from_path(f"{broken}:preprocess")

    assert set(sys.modules) == before
