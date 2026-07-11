from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from llmcompressor.args import DatasetArguments, ModelArguments
from llmcompressor.entrypoints.utils import pre_process, raise_if_model_is_quantized


def test_raise_if_model_is_quantized():
    model = SimpleNamespace(
        config=SimpleNamespace(quantization_config={"quant_method": "fp8"})
    )

    with pytest.raises(ValueError, match="already has a `quantization_config`"):
        raise_if_model_is_quantized(model)


def test_raise_if_model_is_quantized_allows_full_precision_model():
    model = SimpleNamespace(config=SimpleNamespace())

    raise_if_model_is_quantized(model)


def test_pre_process_checks_for_quantized_model_before_processor_init(monkeypatch):
    model = SimpleNamespace(
        config=SimpleNamespace(quantization_config={"quant_method": "fp8"})
    )
    initialize_processor_from_path = Mock()
    monkeypatch.setattr(
        "llmcompressor.entrypoints.utils.initialize_processor_from_path",
        initialize_processor_from_path,
    )

    with pytest.raises(ValueError, match="full-precision checkpoint"):
        pre_process(
            ModelArguments(model=model),
            DatasetArguments(),
            output_dir=None,
        )

    initialize_processor_from_path.assert_not_called()
