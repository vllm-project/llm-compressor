import pytest
import torch
from compressed_tensors.compressors import compress_module, decompress_module
from compressed_tensors.quantization import QuantizationStatus
from transformers import AutoModelForCausalLM
from transformers.utils.quantization_config import CompressedTensorsConfig

COMPRESSED_MODEL = "nm-testing/TinyLlama-1.1B-Chat-v1.0-W4A16-G128-compressed"


def _first_quant_linear(model):
    for _, m in model.named_modules():
        if getattr(m, "quantization_scheme", None) is not None and isinstance(
            m, torch.nn.Linear
        ):
            return m
    raise AssertionError("no quantized linear found")


@pytest.mark.integration
def test_run_compressed_true_keeps_modules_compressed():
    model = AutoModelForCausalLM.from_pretrained(
        COMPRESSED_MODEL,
        quantization_config=CompressedTensorsConfig(run_compressed=True),
        dtype=torch.bfloat16,
        device_map="cpu",
    )
    mod = _first_quant_linear(model)
    assert getattr(mod, "quantization_status", None) == QuantizationStatus.COMPRESSED
    assert hasattr(mod, "weight_shape")
    assert not hasattr(mod, "weight")

    fmt = mod.quantization_scheme.format
    orig_shape = tuple(mod.weight_shape.tolist())

    decompress_module(mod)
    assert hasattr(mod, "weight")
    assert tuple(mod.weight.shape) == orig_shape

    compress_module(mod, format=fmt)
    assert getattr(mod, "quantization_status", None) == QuantizationStatus.COMPRESSED
    assert not hasattr(mod, "weight")


def test_ct_model_loads_compressed_and_stashes_formats():
    from llmcompressor.args import ModelArguments
    from llmcompressor.entrypoints.utils import initialize_model_from_path

    model = initialize_model_from_path(
        ModelArguments(model=COMPRESSED_MODEL, save_compressed=True)
    )
    mod = _first_quant_linear(model)
    assert not hasattr(mod, "weight")
    assert hasattr(mod, "_ct_input_format")
    assert model._recompress_on_calibration is True
    assert model._sequential_decompression_active is True


def test_force_full_decompression_loads_dense():
    from llmcompressor.args import ModelArguments
    from llmcompressor.entrypoints.utils import initialize_model_from_path

    model = initialize_model_from_path(
        ModelArguments(model=COMPRESSED_MODEL, force_full_decompression=True)
    )
    mod = _first_quant_linear(model)
    assert hasattr(mod, "weight")
    assert getattr(model, "_sequential_decompression_active", False) is False
