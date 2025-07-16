import os
import shutil
import tempfile
import unittest

import torch
from compressed_tensors.linear.compressed_linear import CompressedLinear
from parameterized import parameterized_class
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.quantization_config import CompressedTensorsConfig

from tests.testing_utils import parse_params, requires_gpu

COMPRESSED_LINEAR_CONFIG_DIR = (
    "tests/llmcompressor/transformers/compression/run_compressed_configs"
)


@requires_gpu
@parameterized_class(parse_params(COMPRESSED_LINEAR_CONFIG_DIR))
class Test_Decompressed_Linear_Uncompressed_Linear(unittest.TestCase):
    """
    Uncompressed-Linear-forward decompressed-Linear-foward check

    Uncompressed:  Optimized model saved as run_compressed=False, no need to decompress
    Decompressed:  Optimized model saved as run_compressed=True, and decompressed using
        AutoModelForCausalLM decompression

    AutoModelForCausalLM decompression diagram flow https://tinyurl.com/2ynb6wbu
    """

    compressed_model_stub = None
    uncompressed_model_stub = None

    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp()

        quantization_config = CompressedTensorsConfig(run_compressed=False)

        # Decompressed using HFQuantizer
        # Linear foward
        cls.decompressed_model = AutoModelForCausalLM.from_pretrained(
            cls.compressed_model_stub,
            torch_dtype="auto",
            device_map="auto",
            quantization_config=quantization_config,
        )

        # Load model as is at the uncompressed state
        # Linear forward
        cls.uncompressed_model = AutoModelForCausalLM.from_pretrained(
            cls.uncompressed_model_stub,
            torch_dtype=cls.decompressed_model.dtype,
            device_map=cls.decompressed_model.device,
        )

        cls.tokenizer = AutoTokenizer.from_pretrained(cls.compressed_model_stub)

    def test_compressed_matches_decompressed(self):
        SAMPLE_INPUT = [
            "I love 4-bit quantization because",
            "What is the capital of France?",
            "def fibonacci(n):",
        ]

        decompressed_device = self.decompressed_model.device
        uncompressed_device = self.uncompressed_model.device

        inputs = self.tokenizer(SAMPLE_INPUT, return_tensors="pt", padding=True).to(
            decompressed_device
        )

        decompressed_output = self.decompressed_model.generate(**inputs, max_length=50)

        inputs = inputs.to(uncompressed_device)

        uncompressed_output = self.uncompressed_model.generate(**inputs, max_length=50)

        for idx in range(len(SAMPLE_INPUT)):
            assert torch.equal(decompressed_output[idx], uncompressed_output[idx])

    @classmethod
    def tearDownClass(cls):
        if os.path.isdir(cls.test_dir):
            shutil.rmtree(cls.test_dir)
        del cls.decompressed_model
        del cls.uncompressed_model
        torch.cuda.empty_cache()


@requires_gpu
@parameterized_class(parse_params(COMPRESSED_LINEAR_CONFIG_DIR))
class Test_Compressed_CompressedLinear_Decompressed_Linear(unittest.TestCase):
    """
    Compressed-CompresesdLinear, Decompressed-Linear check

    Compressed:    Optimized model saved as run_compressed=True, no decompression
    Decompressed:  Optimized model saved as run_compressed=True, and decompressed using
        AutoModelForCausalLM decompression

    All compressed model should have CompressedLinear, which has its custom forward call

    """

    compressed_model_stub = None

    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp()

        # Should have CompressedLinear modules
        # Compressed Linear forward
        cls.compressed_model = AutoModelForCausalLM.from_pretrained(
            cls.compressed_model_stub,
            torch_dtype="auto",
            device_map="auto",
        )

        # Should just be linear modules
        # Linear forward
        quantization_config = CompressedTensorsConfig(run_compressed=False)
        cls.decompressed_model = AutoModelForCausalLM.from_pretrained(
            cls.compressed_model_stub,
            torch_dtype=cls.compressed_model.dtype,
            device_map=cls.compressed_model.device,
            quantization_config=quantization_config,
        )

        cls.tokenizer = AutoTokenizer.from_pretrained(cls.compressed_model_stub)

    def test_compressed_linear_modules_exist(self):
        compressed_linear_counts = 0
        for submodule in self.compressed_model.modules():
            if isinstance(submodule, CompressedLinear):
                compressed_linear_counts += 1

        # some linear models are not compressed - ex. lm_head
        assert compressed_linear_counts > 0

    def test_compressed_matches_decompressed__hf_quantizer(self):
        SAMPLE_INPUT = [
            "I love 4-bit quantization because",
            "What is the capital of France?",
            "def fibonacci(n):",
        ]

        decompressed_device = self.decompressed_model.device
        compressed_device = self.compressed_model.device

        inputs = self.tokenizer(SAMPLE_INPUT, return_tensors="pt", padding=True).to(
            decompressed_device
        )

        decompressed_model_out = self.decompressed_model.generate(
            **inputs, max_length=50
        )

        inputs = inputs.to(compressed_device)

        compressed_model_out = self.compressed_model.generate(**inputs, max_length=50)

        # Compare outputs for each input
        for idx in range(len(SAMPLE_INPUT)):
            torch.equal(compressed_model_out[idx], decompressed_model_out[idx])

    @classmethod
    def tearDownClass(cls):
        if os.path.isdir(cls.test_dir):
            shutil.rmtree(cls.test_dir)
        del cls.decompressed_model
        del cls.compressed_model
        torch.cuda.empty_cache()
