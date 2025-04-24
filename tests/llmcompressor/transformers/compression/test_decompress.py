import copy
import os
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
class TestDecompression(unittest.TestCase):
    """
    Check that HFQuantizer decompression is working as expected.
    Manually decompress a compressed model and compare the generations

    Decompression:
    Given a skeleton model and path to the optimized model,
    write the optimized model's safetensors to the skeleton model and decompress
    Ex. write weight_scale to the skeleton model and then convert from fp4 to fp16

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

        # Decompress using HFQuantizer from AutoModelForCausalLM
        self.decompressed_model_hf_quantizer = AutoModelForCausalLM.from_pretrained(
            self.compressed_model_stub,
            torch_dtype="auto",
            device_map="auto",
            quantization_config=CompressedTensorsConfig(run_compressed=False),
        )

        # Manually decompress this model
        self.dense_model = AutoModelForCausalLM.from_pretrained(
            self.skeleton_model_stub,
            torch_dtype=self.decompressed_model_hf_quantizer.dtype,
            device_map=self.decompressed_model_hf_quantizer.device,
        )

        # decompression from HFQuantizer should populate weight_scale
        assert hasattr(
            self.decompressed_model_hf_quantizer.model.layers[0].self_attn.q_proj,
            "weight_scale",
        )

        # dense model should not have weight_scale populated
        assert not hasattr(
            self.dense_model.model.layers[0].self_attn.q_proj, "weight_scale"
        )

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

    def test_hf_quantizer_decompress_match_manual_decompress(self):
        manual_device = self.decompressed_model_manual.device
        decompressed_model_hf_quantizer = self.decompressed_model_hf_quantizer.device

        self.decompressed_model_manual = self.decompressed_model_manual.to(
            manual_device
        )
        self.decompressed_model_hf_quantizer = self.decompressed_model_hf_quantizer.to(
            decompressed_model_hf_quantizer
        )

        for input in self.SAMPLE_INPUTS:
            inputs = self.tokenizer(input, return_tensors="pt", padding=True).to(
                self.decompressed_model_manual.device
            )
            inputs = inputs.to(self.decompressed_model_manual.device)

            decompressed_model_manual_output = self.decompressed_model_manual.generate(
                **inputs, max_length=50
            )

            decompressed_model_hf_quantizer_out = (
                self.decompressed_model_hf_quantizer.generate(**inputs, max_length=50)
            )

            assert torch.equal(
                decompressed_model_hf_quantizer_out, decompressed_model_manual_output
            )

    @classmethod
    def tearDownClass(self):
        if os.path.isdir(self.test_dir):
            shutil.rmtree(self.test_dir)
        del self.dense_model
        del self.decompressed_model_hf_quantizer
        del self.decompressed_model_manual
