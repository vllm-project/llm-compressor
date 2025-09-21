import copy

import pytest
import torch
from compressed_tensors import QUANTIZATION_CONFIG_NAME
from compressed_tensors.compressors import ModelCompressor
from compressed_tensors.quantization import QuantizationStatus
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.utils.quantization_config import CompressedTensorsConfig

from tests.testing_utils import parse_params, requires_gpu

CONFIG_DIR = "tests/llmcompressor/transformers/compression/decompression_configs"


@requires_gpu
@pytest.mark.parametrize("config", parse_params(CONFIG_DIR))
def test_hf_quantizer_decompress_match_manual_decompress(config):
    """
    Check that HFQuantizer decompression is working as expected.
    Manually decompress a compressed model and compare the generations

    Decompression:
    Given a skeleton model and path to the optimized model,
    write the optimized model's safetensors to the skeleton model and decompress
    Ex. write weight_scale to the skeleton model and then convert from fp4 to fp16

    """

    compressed_model_stub = config["compressed_model_stub"]
    skeleton_model_stub = config["skeleton_model_stub"]

    sample_inputs = [
        "I love 4-bit quantization because",
        "What is the capital of France?",
        "def fibonacci(n):",
    ]

    tokenizer = AutoTokenizer.from_pretrained(compressed_model_stub)

    # Decompress using HFQuantizer from AutoModelForCausalLM
    decompressed_model_hf_quantizer = AutoModelForCausalLM.from_pretrained(
        compressed_model_stub,
        torch_dtype="auto",
        device_map="auto",
        quantization_config=CompressedTensorsConfig(run_compressed=False),
    )

    # Manually decompress this model
    dense_model = AutoModelForCausalLM.from_pretrained(
        skeleton_model_stub,
        torch_dtype=decompressed_model_hf_quantizer.dtype,
        device_map=decompressed_model_hf_quantizer.device,
    )

    # decompression from HFQuantizer should populate weight_scale
    assert hasattr(
        decompressed_model_hf_quantizer.model.layers[0].self_attn.q_proj,
        "weight_scale",
    )

    # dense model should not have weight_scale populated
    assert not hasattr(dense_model.model.layers[0].self_attn.q_proj, "weight_scale")

    config = AutoConfig.from_pretrained(compressed_model_stub)

    compression_config = getattr(config, QUANTIZATION_CONFIG_NAME, None)
    compressor = ModelCompressor.from_compression_config(compression_config)
    compressor.quantization_config.quantization_status = QuantizationStatus.FROZEN

    # use the model_path to load the decompressed weights into dense_model
    orig_dense_model = copy.deepcopy(dense_model)

    # overwrite the weights of the dense model
    compressor.decompress(
        model_path=compressed_model_stub,
        model=dense_model,
    )

    # self.dense_model should be decompressed
    assert dense_model is not orig_dense_model

    decompressed_model_manual = dense_model

    assert hasattr(
        decompressed_model_manual.model.layers[0].self_attn.q_proj,
        "weight_scale",
    )

    device = decompressed_model_manual.device
    decompressed_model_manual = decompressed_model_manual.to(device)
    decompressed_model_hf_quantizer = decompressed_model_hf_quantizer.to(device)

    for input in sample_inputs:
        inputs = tokenizer(input, return_tensors="pt", padding=True).to(device)

        decompressed_model_manual_output = decompressed_model_manual.generate(
            **inputs, max_length=50
        )

        decompressed_model_hf_quantizer_out = decompressed_model_hf_quantizer.generate(
            **inputs, max_length=50
        )

        assert torch.equal(
            decompressed_model_hf_quantizer_out, decompressed_model_manual_output
        )
