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
from transformers.utils.quantization_config import CompressedTensorsConfig

from tests.testing_utils import parse_params, requires_gpu

CONFIG_DIR = "tests/llmcompressor/transformers/compression/decompression_configs"


@requires_gpu
@parameterized_class(parse_params(CONFIG_DIR))
class TestQuantizationMatches(unittest.TestCase):
    """
    Test decompression - given a skeleton model and path to the optimized model,
    write the optimized model's safetensors to the skeleton model and decompress
    Ex. write weight_scale to skeleton model and then fp4 -> fp16

    Check that HFQuantizer decompression and manual decompressed generates the
    same output

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
        self.decompressed_model = AutoModelForCausalLM.from_pretrained(
            self.compressed_model_stub,
            torch_dtype="auto",
            device_map="auto",
            quantization_config=CompressedTensorsConfig(run_compressed=False),
        )

        # manually decompress this model
        self.dense_model = AutoModelForCausalLM.from_pretrained(
            self.skeleton_model_stub,
            torch_dtype=self.compressed_model.dtype,
            device_map=self.compressed_model.device,
        )

        assert not hasattr(
            self.dense_model.model.layers[0].self_attn.q_proj, "weight_scale"
        )

        self.decompressed_model_manual = None
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

        self.decompressed_model_manual = self.dense_model

        assert hasattr(
            self.decompressed_model_manual.model.layers[0].self_attn.q_proj,
            "weight_scale",
        )

    def test_compressed_matches_uncompressed(self):
        decompressed_model_manual = self.decompressed_model_manual.device
        compressed_device = self.compressed_model.device
        decompressed_model_device = self.decompressed_model.device

        self.decompressed_model_manual = self.decompressed_model_manual.to(
            decompressed_model_manual
        )
        self.compressed_model = self.compressed_model.to(compressed_device)
        self.decompressed_model = self.decompressed_model.to(decompressed_model_device)

        for input in self.SAMPLE_INPUTS:
            inputs = self.tokenizer(input, return_tensors="pt", padding=True).to(
                self.compressed_model.device
            )

            compressed_output = self.tokenizer.batch_decode(
                self.compressed_model.generate(**inputs, max_length=50)
            )

            inputs = inputs.to(self.decompressed_model_manual.device)

            decompressed_model_manual_output = self.tokenizer.batch_decode(
                self.decompressed_model_manual.generate(**inputs, max_length=50)
            )

            decompressed_model_out = self.tokenizer.batch_decode(
                self.decompressed_model.generate(**inputs, max_length=50)
            )

            assert (
                compressed_output
                == decompressed_model_manual_output
                == decompressed_model_out
            )

    @classmethod
    def tearDownClass(self):
        shutil.rmtree(self.test_dir)
        del self.compressed_model
        del self.dense_model
        del self.decompressed_model
        del self.decompressed_model_manual
        torch.cuda.empty_cache()
