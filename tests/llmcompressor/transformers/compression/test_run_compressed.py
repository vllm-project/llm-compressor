import shutil
import tempfile
import unittest

import torch
from compressed_tensors import QUANTIZATION_CONFIG_NAME
from compressed_tensors.compressors import ModelCompressor
from compressed_tensors.quantization import QuantizationStatus
from parameterized import parameterized_class
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from llmcompressor.transformers import SparseAutoModelForCausalLM
from tests.testing_utils import parse_params, requires_gpu, requires_torch

CONFIG_DIR = "tests/llmcompressor/transformers/compression/run_compressed_configs"


@requires_torch
@requires_gpu
@parameterized_class(parse_params(CONFIG_DIR))
class TestQuantizationMatches(unittest.TestCase):
    model_stub = None
    empty_model = None

    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp()

        cls.compressed_model = SparseAutoModelForCausalLM.from_pretrained(
            cls.model_stub, torch_dtype="auto", device_map="auto", run_compressed=True
        )

        # TODO: Use ModelCompressor until decompression is supported through
        # HFQuant/run_compressed can be turned off.
        cls.uncompressed_model = AutoModelForCausalLM.from_pretrained(
            cls.empty_model,
            torch_dtype=cls.compressed_model.dtype,
            device_map=cls.compressed_model.device,
        )
        config = AutoConfig.from_pretrained(cls.model_stub)
        compression_config = getattr(config, QUANTIZATION_CONFIG_NAME, None)
        cls.compressor = ModelCompressor.from_compression_config(compression_config)
        cls.compressor.quantization_config.quantization_status = (
            QuantizationStatus.FROZEN
        )
        cls.compressor.decompress(
            model_path=cls.model_stub, model=cls.uncompressed_model
        )

        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_stub)

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
