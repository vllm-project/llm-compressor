import gc

import pytest
import torch
from compressed_tensors.linear.compressed_linear import CompressedLinear
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
        torch_dtype="auto",
        device_map="auto",
        quantization_config=quantization_config,
    )

    # Load model as is at the uncompressed state
    # Linear forward
    uncompressed_model = AutoModelForCausalLM.from_pretrained(
        config["uncompressed_model_stub"],
        torch_dtype=decompressed_model.dtype,
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


class Test_Compressed_CompressedLinear_Decompressed_Linear:
    @pytest.fixture(params=parse_params(COMPRESSED_LINEAR_CONFIG_DIR), scope="class")
    def compressed_compressed_linear_decompressed_linear_models(self, request):
        """
        Compressed-CompresesdLinear, Decompressed-Linear check

        Compressed:    Optimized model saved as run_compressed=True, no decompression
        Decompressed:  Optimized model saved as run_compressed=True, and decompressed
            using AutoModelForCausalLM decompression

        All compressed model should have CompressedLinear, which has its custom forward

        """
        config = request.param
        # config: {compressed_model_stub}

        # Should have CompressedLinear modules
        # Compressed Linear forward
        compressed_model = AutoModelForCausalLM.from_pretrained(
            config["compressed_model_stub"],
            torch_dtype="auto",
            device_map="auto",
        )

        # Should just be linear modules
        # Linear forward
        quantization_config = CompressedTensorsConfig(run_compressed=False)
        decompressed_model = AutoModelForCausalLM.from_pretrained(
            config["compressed_model_stub"],
            torch_dtype=compressed_model.dtype,
            device_map=compressed_model.device,
            quantization_config=quantization_config,
        )

        tokenizer = AutoTokenizer.from_pretrained(config["compressed_model_stub"])

        yield compressed_model, decompressed_model, tokenizer

        del decompressed_model
        del compressed_model
        del tokenizer

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def test_compressed_linear_modules_exist(
        self,
        compressed_compressed_linear_decompressed_linear_models,
    ):
        compressed_model, _, _ = compressed_compressed_linear_decompressed_linear_models
        compressed_linear_counts = 0
        for submodule in compressed_model.modules():
            if isinstance(submodule, CompressedLinear):
                compressed_linear_counts += 1

        # some linear models are not compressed - ex. lm_head
        assert compressed_linear_counts > 0

    def test_compressed_matches_decompressed__hf_quantizer(
        self,
        compressed_compressed_linear_decompressed_linear_models,
    ):
        SAMPLE_INPUT = [
            "I love 4-bit quantization because",
            "What is the capital of France?",
            "def fibonacci(n):",
        ]
        compressed_model, decompressed_model, tokenizer = (
            compressed_compressed_linear_decompressed_linear_models
        )

        decompressed_device = decompressed_model.device
        compressed_device = compressed_model.device

        inputs = tokenizer(SAMPLE_INPUT, return_tensors="pt", padding=True).to(
            decompressed_device
        )

        decompressed_model_out = decompressed_model.generate(**inputs, max_length=50)

        inputs = inputs.to(compressed_device)

        compressed_model_out = compressed_model.generate(**inputs, max_length=50)

        # Compare outputs for each input
        for idx in range(len(SAMPLE_INPUT)):
            torch.equal(compressed_model_out[idx], decompressed_model_out[idx])
