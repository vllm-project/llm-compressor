import shutil
import tempfile
import unittest

import torch
from parameterized import parameterized_class
from transformers import AutoModelForCausalLM, AutoTokenizer

from tests.testing_utils import parse_params, requires_gpu, requires_torch

CONFIG_DIR = "tests/llmcompressor/transformers/compression/run_compressed_configs"


@requires_torch
@requires_gpu
@parameterized_class(parse_params(CONFIG_DIR))
class TestQuantizationMatches(unittest.TestCase):
    model_stub = None

    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp()

        cls.compressed_model = AutoModelForCausalLM.from_pretrained(
            cls.model_stub,
            torch_dtype="auto",
            device_map="auto",
            run_compressed=True,
        )
        cls.uncompressed_model = AutoModelForCausalLM.from_pretrained(
            cls.model_stub,
            torch_dtype="auto",
            device_map="auto",
        )
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_stub)
        cls.device = cls.compressed_model.device

    def test_compressed_matches_uncompressed(self):
        SAMPLE_INPUT = [
            "I love 4-bit quantization because",
            "What is the capital of Paris?",
            "def fibonacci(n):",
        ]

        inputs = self.tokenizer(SAMPLE_INPUT, return_tensors="pt", padding=True).to(
            self.device
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
