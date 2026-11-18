import pytest
import torch
from compressed_tensors.compressors import decompress_module
from compressed_tensors.quantization.utils import is_module_quantized
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.quantization_config import CompressedTensorsConfig

from tests.testing_utils import parse_params, requires_gpu, requires_version

CONFIG_DIR = "tests/llmcompressor/transformers/compression/decompression_configs"


@requires_gpu
@requires_version("transformers", ">=5.23")
@pytest.mark.parametrize("config", parse_params(CONFIG_DIR))
def test_hf_quantizer_decompress_match_manual_decompress(config):
    """
    Check that HFQuantizer decompression is working as expected.
    Manually decompress a compressed model and compare the generations
    """

    compressed_model_stub = config["compressed_model_stub"]

    sample_inputs = [
        "I love 4-bit quantization because",
        "What is the capital of France?",
        "def fibonacci(n):",
    ]
    tokenizer = AutoTokenizer.from_pretrained(compressed_model_stub)

    # Decompress using HFQuantizer from AutoModelForCausalLM. Note that
    # transformers>=5.23 loads a plain dense model with no quantization
    # metadata when `CompressedTensorsConfig(dequantize=True)` is used
    # (see huggingface/transformers#50238)
    hf_quantizer_model = AutoModelForCausalLM.from_pretrained(
        compressed_model_stub,
        device_map="auto",
        quantization_config=CompressedTensorsConfig(dequantize=True),
    )

    # Manually decompress from compressed model. Pass leave_decompressed=False
    # so that the quantization metadata is removed, matching the plain dense
    # model loaded above through the HFQuantizer
    manual_model = AutoModelForCausalLM.from_pretrained(
        compressed_model_stub,
        dtype=hf_quantizer_model.dtype,
        device_map=hf_quantizer_model.device,
    )
    for module in manual_model.modules():
        if is_module_quantized(module):
            decompress_module(module, leave_decompressed=False)

    # Check generations
    device = manual_model.device
    for input in sample_inputs:
        inputs = tokenizer(input, return_tensors="pt", padding=True).to(device)
        manual_output = manual_model.generate(**inputs, max_length=15)
        hf_quantizer_output = hf_quantizer_model.generate(**inputs, max_length=15)
        assert torch.equal(manual_output, hf_quantizer_output)
