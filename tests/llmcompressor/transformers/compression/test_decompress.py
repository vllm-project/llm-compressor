import pytest
import torch
from compressed_tensors.compressors import ModelCompressor
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.quantization_config import CompressedTensorsConfig

from tests.testing_utils import parse_params, requires_gpu

CONFIG_DIR = "tests/llmcompressor/transformers/compression/decompression_configs"


@requires_gpu
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

    # Decompress using HFQuantizer from AutoModelForCausalLM
    hf_quantizer_model = AutoModelForCausalLM.from_pretrained(
        compressed_model_stub,
        dtype="auto",
        device_map="auto",
        quantization_config=CompressedTensorsConfig(run_compressed=False),
    )

    # Manually decompress from compressed model
    manual_model = AutoModelForCausalLM.from_pretrained(
        compressed_model_stub,
        dtype=hf_quantizer_model.dtype,
        device_map=hf_quantizer_model.device,
    )
    ModelCompressor().decompress_model(manual_model)

    # Check generations
    device = manual_model.device
    for input in sample_inputs:
        inputs = tokenizer(input, return_tensors="pt", padding=True).to(device)
        manual_output = manual_model.generate(**inputs, max_length=15)
        hf_quantizer_output = hf_quantizer_model.generate(**inputs, max_length=15)
        assert torch.equal(manual_output, hf_quantizer_output)
