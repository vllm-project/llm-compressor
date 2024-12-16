import shutil
import tempfile
import unittest

import torch
from parameterized import parameterized_class
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.quantization_config import CompressedTensorsConfig

from tests.testing_utils import parse_params, requires_gpu

CONFIG_DIR = "tests/llmcompressor/transformers/compression/run_compressed_configs"


@requires_gpu
@parameterized_class(parse_params(CONFIG_DIR))
class TestQuantizationMatches(unittest.TestCase):
    """
    Test the run_compressed input arg to AutoModelForCausalLM, where HFQuantizer is
    responsible for decompressing if model is compressed.

    Diagram flow https://tinyurl.com/2ynb6wbu

    """

    compressed_model_stub = None  # model was compressed on save
    uncompressed_model_stub = None  # model was not compressed on save

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

    def test_compressed_matches_uncompressed(self):
        SAMPLE_INPUT = [
            "I love 4-bit quantization because",
            "What is the capital of France?",
            "def fibonacci(n):",
        ]

        inputs = self.tokenizer(SAMPLE_INPUT, return_tensors="pt", padding=True).to(
            self.decompressed_model.device
        )
        compressed_output = self.tokenizer.batch_decode(
            self.decompressed_model.generate(**inputs, max_length=50)
        )
        uncompressed_output = self.tokenizer.batch_decode(
            self.uncompressed_model.generate(**inputs, max_length=50)
        )

        for idx in range(len(SAMPLE_INPUT)):
            assert compressed_output[idx] == uncompressed_output[idx]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir)
        del cls.decompressed_model
        del cls.base_model
        torch.cuda.empty_cache()
