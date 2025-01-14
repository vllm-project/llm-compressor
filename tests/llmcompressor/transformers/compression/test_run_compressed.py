import shutil
import tempfile
import unittest

import torch
from compressed_tensors import QUANTIZATION_CONFIG_NAME
from compressed_tensors.compressors import ModelCompressor
from compressed_tensors.quantization import QuantizationStatus
from parameterized import parameterized_class
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from tests.testing_utils import parse_params, requires_gpu

CONFIG_DIR = "tests/llmcompressor/transformers/compression/run_compressed_configs"


@requires_gpu
@parameterized_class(parse_params(CONFIG_DIR))
class TestQuantizationMatches(unittest.TestCase):
    model_stub = None
    empty_model = None

    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp()

        # TODO: Give option on HFQuantizer to run run_compressed True/False
        # currently hardcoded to True
        cls.compressed_model = AutoModelForCausalLM.from_pretrained(
            cls.model_stub,
            torch_dtype="auto",
            device_map="auto",
            # run_compressed=True, # TODO: Give option on HFQuantizer
        )
        # TODO: Use ModelCompressor until decompression is supported through
        # HFQuant/run_compressed can be turned off.
        cls.uncompressed_model = AutoModelForCausalLM.from_pretrained(
            cls.empty_model,
            torch_dtype=cls.compressed_model.dtype,
            device_map=cls.compressed_model.device,
        )
        breakpoint()

        cls.tokenizer = AutoTokenizer.from_pretrained(cls.compressed_model_stub)

    def test_compressed_matches_uncompressed(self):
        SAMPLE_INPUT = [
            "I love 4-bit quantization because",
            "What is the capital of France?",
            "def fibonacci(n):",
        ]

        inputs = self.tokenizer(SAMPLE_INPUT, return_tensors="pt", padding=True).to(
            self.compressed_model.device
        )
        compressed_output = self.tokenizer.batch_decode(
            self.compressed_model.generate(**inputs, max_length=50)
        )
        uncompressed_output = self.tokenizer.batch_decode(
            self.uncompressed_model.generate(**inputs, max_length=50)
        )

        for idx in range(len(SAMPLE_INPUT)):
            assert compressed_output[idx] == uncompressed_output[idx]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir)
        del cls.compressed_model
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
        for _, submodule in iter_named_leaf_modules(
            self.compressed_model,
        ):
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

        # overwrite weights in cpu to cuda
        self.decompressed_model = self.decompressed_model.to(decompressed_device)
        self.compressed_model = self.compressed_model.to(compressed_device)

        inputs = self.tokenizer(SAMPLE_INPUT, return_tensors="pt", padding=True).to(
            decompressed_device
        )

        decompressed_model_out = self.tokenizer.batch_decode(
            self.decompressed_model.generate(**inputs, max_length=50)
        )

        inputs = inputs.to(compressed_device)

        compressed_model_out = self.tokenizer.batch_decode(
            self.compressed_model.generate(**inputs, max_length=50)
        )

        # Compare outputs for each input
        for idx in range(len(SAMPLE_INPUT)):
            assert compressed_model_out[idx] == decompressed_model_out[idx]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir)
        del cls.decompressed_model
        del cls.compressed_model
        torch.cuda.empty_cache()
