import gc

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.quantization_config import CompressedTensorsConfig

from tests.testing_utils import parse_params, requires_gpu

COMPRESSED_LINEAR_CONFIG_DIR = (
    "tests/llmcompressor/transformers/compression/run_compressed_configs"
)


@pytest.fixture(params=parse_params(COMPRESSED_LINEAR_CONFIG_DIR))
def decompressed_linear_uncompressed_linear_models(request):
    config = request.param
    # config: {compressed_model_stub, uncompressed_model_stub}

    quantization_config = CompressedTensorsConfig(run_compressed=False)

    # Decompressed using HFQuantizer
    # Linear foward
    decompressed_model = AutoModelForCausalLM.from_pretrained(
        config["compressed_model_stub"],
        dtype="auto",
        device_map="auto",
        quantization_config=quantization_config,
    )

    # Load model as is at the uncompressed state
    # Linear forward
    uncompressed_model = AutoModelForCausalLM.from_pretrained(
        config["uncompressed_model_stub"],
        dtype=decompressed_model.dtype,
        device_map=decompressed_model.device,
    )

    tokenizer = AutoTokenizer.from_pretrained(config["compressed_model_stub"])

    yield decompressed_model, uncompressed_model, tokenizer

    del decompressed_model
    del uncompressed_model
    del tokenizer

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


@requires_gpu
def test_decompressed_linear_uncompressed_linear(
    decompressed_linear_uncompressed_linear_models,
):
    """
    Uncompressed-Linear-forward decompressed-Linear-foward check

    Uncompressed:  Optimized model saved as run_compressed=False, no need to decompress
    Decompressed:  Optimized model saved as run_compressed=True, and decompressed using
        AutoModelForCausalLM decompression

    AutoModelForCausalLM decompression diagram flow https://tinyurl.com/2ynb6wbu
    """

    SAMPLE_INPUT = [
        "I love 4-bit quantization because",
        "What is the capital of France?",
        "def fibonacci(n):",
    ]

    decompressed_model, uncompressed_model, tokenizer = (
        decompressed_linear_uncompressed_linear_models
    )
    decompressed_device = decompressed_model.device
    uncompressed_device = uncompressed_model.device

    inputs = tokenizer(SAMPLE_INPUT, return_tensors="pt", padding=True).to(
        decompressed_device
    )
    decompressed_output = decompressed_model.generate(**inputs, max_length=50)
    inputs = inputs.to(uncompressed_device)
    uncompressed_output = uncompressed_model.generate(**inputs, max_length=50)

    for idx in range(len(SAMPLE_INPUT)):
        assert torch.equal(decompressed_output[idx], uncompressed_output[idx])
