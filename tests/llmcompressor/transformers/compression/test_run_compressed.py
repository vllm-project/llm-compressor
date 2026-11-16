import gc

import pytest
import torch
from compressed_tensors.quantization import QuantizationMetadata, forward_quantize
from compressed_tensors.quantization.utils import is_module_quantized
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.quantization_config import CompressedTensorsConfig

from tests.testing_utils import parse_params, requires_gpu, requires_version

COMPRESSED_LINEAR_CONFIG_DIR = (
    "tests/llmcompressor/transformers/compression/run_compressed_configs"
)


@pytest.fixture(params=parse_params(COMPRESSED_LINEAR_CONFIG_DIR))
def decompressed_linear_uncompressed_linear_models(request):
    config = request.param
    # config: {compressed_model_stub, uncompressed_model_stub}

    quantization_config = CompressedTensorsConfig(dequantize=True)

    # Decompressed using HFQuantizer
    # Linear foward
    decompressed_model = AutoModelForCausalLM.from_pretrained(
        config["compressed_model_stub"],
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
    # The uncompressed checkpoint stores dense weights alongside its quantization
    # parameters, so as loaded it keeps simulating the quantized model at runtime
    # (weight QDQ and dynamic activation quantization). Materialize the weight QDQ
    # and strip the quantization metadata so that the model is fully dense,
    # matching the HFQuantizer load above. Note that transformers>=5.23 loads a
    # plain dense model (dropping activation quantization) when
    # `CompressedTensorsConfig(dequantize=True)` is used
    # (see huggingface/transformers#50238). The weights are materialized through
    # forward_quantize, the same QDQ the wrapped forward applies, rather than
    # through compress_module: its triton quantize kernel divides in float32,
    # which lands on a different grid than the checkpoint values on GPUs
    for module in uncompressed_model.modules():
        if is_module_quantized(module):
            scheme = module.quantization_scheme
            if scheme.weights is not None:
                module.weight.data = forward_quantize(
                    module, module.weight.data, "weight", scheme.weights
                )
            QuantizationMetadata.clear_quantization(module)
            if hasattr(module, "quantization_status"):
                delattr(module, "quantization_status")

    tokenizer = AutoTokenizer.from_pretrained(config["compressed_model_stub"])

    yield decompressed_model, uncompressed_model, tokenizer

    del decompressed_model
    del uncompressed_model
    del tokenizer

    gc.collect()
    torch.accelerator.empty_cache()
    torch.accelerator.synchronize()


@requires_gpu
@requires_version("transformers", ">=5.23")
def test_decompressed_linear_uncompressed_linear(
    decompressed_linear_uncompressed_linear_models,
):
    """
    Uncompressed-Linear-forward decompressed-Linear-foward check

    Uncompressed:  Optimized model saved as dequantize=True, no need to decompress
    Decompressed:  Optimized model saved as dequantize=False, and decompressed using
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
