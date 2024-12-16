import copy
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

CONFIG_DIR = "tests/llmcompressor/transformers/compression/decompression_configs"


@requires_gpu
@parameterized_class(parse_params(CONFIG_DIR))
class TestQuantizationMatches(unittest.TestCase):
    """
    Test the decompression, which copies the attrs of compressed_model_stub's
    safetensors to skeleton_model_stub and decompresses. Ex. fp4 -> fp16
    """

    compressed_model_stub = None
    skeleton_model_stub = None

    SAMPLE_INPUTS = [
        "I love 4-bit quantization because",
        "What is the capital of France?",
        "def fibonacci(n):",
    ]

    @classmethod
    def setUpClass(self):
        self.test_dir = tempfile.mkdtemp()
        self.tokenizer = AutoTokenizer.from_pretrained(self.compressed_model_stub)

        self.compressed_model = AutoModelForCausalLM.from_pretrained(
            self.compressed_model_stub,
            torch_dtype="auto",
            device_map="auto",
        )

        self.dense_model = AutoModelForCausalLM.from_pretrained(
            self.skeleton_model_stub,
            torch_dtype=self.compressed_model.dtype,
            device_map=self.compressed_model.device,
        )

        assert not hasattr(
            self.dense_model.model.layers[0].self_attn.q_proj, "weight_scale"
        )

        self.decompressed_model = None
        config = AutoConfig.from_pretrained(self.compressed_model_stub)

        compression_config = getattr(config, QUANTIZATION_CONFIG_NAME, None)
        self.compressor = ModelCompressor.from_compression_config(compression_config)
        self.compressor.quantization_config.quantization_status = (
            QuantizationStatus.FROZEN
        )

        # use the model_path to load the decompressed weights into dense_model
        dense_model = copy.deepcopy(self.dense_model)

        # overwrite the weights of the dense model
        self.compressor.decompress(
            model_path=self.compressed_model_stub,
            model=self.dense_model,
        )

        # self.dense_model should be decompressed
        assert dense_model is not self.dense_model

        self.decompressed_model = self.dense_model

        assert hasattr(
            self.decompressed_model.model.layers[0].self_attn.q_proj, "weight_scale"
        )

    def test_compressed_matches_uncompressed(self):
        for input in self.SAMPLE_INPUTS:
            inputs = self.tokenizer(input, return_tensors="pt", padding=True).to(
                self.compressed_model.device
            )
            compressed_output = self.tokenizer.batch_decode(
                self.compressed_model.generate(**inputs, max_length=50)
            )
            uncompressed_output = self.tokenizer.batch_decode(
                self.decompressed_model.generate(**inputs, max_length=50)
            )

            assert compressed_output == uncompressed_output

    @classmethod
    def tearDownClass(self):
        shutil.rmtree(self.test_dir)
        del self.compressed_model
        del self.dense_model
        del self.decompressed_model
        torch.cuda.empty_cache()
