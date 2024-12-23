import shutil
import tempfile
import unittest

import torch
from compressed_tensors.linear.compressed_linear import CompressedLinear
from compressed_tensors.quantization.utils import iter_named_leaf_modules
from parameterized import parameterized_class
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.quantization_config import CompressedTensorsConfig

from tests.testing_utils import parse_params, requires_gpu

COMPRESSED_LINEAR_CONFIG_DIR = (
    "tests/llmcompressor/transformers/compression/run_compressed_configs"
)


@requires_gpu
@parameterized_class(parse_params(COMPRESSED_LINEAR_CONFIG_DIR))
class TestRunCompressedDecompression(unittest.TestCase):
    """
    Test the run_compressed input arg to AutoModelForCausalLM, where HFQuantizer is
    responsible for decompressing if model is compressed.

    Diagram flow https://tinyurl.com/2ynb6wbu

        Given an optimized model that was saved (uncompressed),
        and saved as run_compressed (compressed), decompress the compressed model
        and check the outputs.

        All modules should be linear, runs default foward calls

    Test the run_compressed input arg to AutoModelForCausalLM, where HFQuantizer is
    responsible for decompressing if model is compressed.

    Diagram flow https://tinyurl.com/2ynb6wbu


    """

    compressed_model_stub = None
    uncompressed_model_stub = None

    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp()

        quantization_config = CompressedTensorsConfig(run_compressed=False)
        cls.decompressed_model = AutoModelForCausalLM.from_pretrained(
            cls.compressed_model_stub,
            torch_dtype="auto",
            device_map="auto",
            quantization_config=quantization_config,
        )

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

        inputs = self.tokenizer(SAMPLE_INPUT, return_tensors="pt", padding=True).to(
            self.decompressed_model.device
        )
        decompressed_output = self.tokenizer.batch_decode(
            self.decompressed_model.generate(**inputs, max_length=50)
        )
        uncompressed_output = self.tokenizer.batch_decode(
            self.uncompressed_model.generate(**inputs, max_length=50)
        )

        for idx in range(len(SAMPLE_INPUT)):
            assert decompressed_output[idx] == uncompressed_output[idx]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir)
        del cls.decompressed_model
        del cls.uncompressed_model
        torch.cuda.empty_cache()


@requires_gpu
@parameterized_class(parse_params(COMPRESSED_LINEAR_CONFIG_DIR))
class TestRunCompressedForward(unittest.TestCase):
    """
    Given an optimized model that was saved (uncompressed),
    and saved as run_compressed (compressed), do not decompressed the compressed model
    and check the outputs.

    All compressed model should have CompressedLinear, which has its custom forward call

    """

    compressed_model_stub = None

    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp()

        # Should have CompressedLinear modules
        cls.compressed_model = AutoModelForCausalLM.from_pretrained(
            cls.compressed_model_stub,
            torch_dtype="auto",
            device_map="auto",
        )

        # Should just be linear modules
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

    def test_compressed_matches_uncompressed(self):
        SAMPLE_INPUT = [
            "I love 4-bit quantization because",
            "What is the capital of France?",
            "def fibonacci(n):",
        ]

        inputs = self.tokenizer(SAMPLE_INPUT, return_tensors="pt", padding=True).to(
            self.decompressed_model.device
        )
        compressed_model_out = self.tokenizer.batch_decode(
            self.decompressed_model.generate(**inputs, max_length=50)
        )
        decompressed_model_out = self.tokenizer.batch_decode(
            self.decompressed_model.generate(**inputs, max_length=50)
        )

        for idx in range(len(SAMPLE_INPUT)):
            assert compressed_model_out[idx] == decompressed_model_out[idx]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir)
        del cls.decompressed_model
        del cls.compressed_model
        torch.cuda.empty_cache()
