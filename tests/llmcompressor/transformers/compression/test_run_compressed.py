import shutil
import tempfile
import unittest

import torch
from compressed_tensors.linear.compressed_linear import CompressedLinear
from compressed_tensors.quantization.utils import iter_named_leaf_modules
from parameterized import parameterized_class
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.quantization_config import CompressedTensorsConfig
from compressed_tensors.quantization.lifecycle.forward import dequantize

from tests.testing_utils import parse_params, requires_gpu

COMPRESSED_LINEAR_CONFIG_DIR = (
    "tests/llmcompressor/transformers/compression/run_compressed_configs"
)


@requires_gpu
@parameterized_class(parse_params(COMPRESSED_LINEAR_CONFIG_DIR))
class _Test_Decompressed_Linear_Uncompressed_Linear(unittest.TestCase):
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

    """
    def _test_compressed_matches_decompressed(self):
        SAMPLE_INPUT = [
            "I love 4-bit quantization because",
            "What is the capital of France?",
            "def fibonacci(n):",
        ]

        decompressed_device = self.decompressed_model.device
        uncompressed_device = self.uncompressed_model.device

        # overwrite weights in cpu to cuda
        self.decompressed_model = self.decompressed_model.to(decompressed_device)
        self.uncompressed_model = self.uncompressed_model.to(uncompressed_device)

        inputs = self.tokenizer(SAMPLE_INPUT, return_tensors="pt", padding=True).to(
            decompressed_device
        )

        decompressed_output = self.decompressed_model.generate(**inputs, max_length=50)

        inputs = inputs.to(uncompressed_device)

        uncompressed_output = self.uncompressed_model.generate(**inputs, max_length=50)

        for idx in range(len(SAMPLE_INPUT)):
            assert torch.equal(decompressed_output[idx], uncompressed_output[idx])
    """

    @classmethod
    def tearDownClass(cls):
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
        from llmcompressor.transformers.tracing.debug import get_model_class
        pretrained_model_class = get_model_class("TraceableQwen2VLForConditionalGeneration")

        cls.compressed_model = pretrained_model_class.from_pretrained(
            cls.compressed_model_stub,
            torch_dtype="auto",
            device_map="auto",
        )
        # Should just be linear modules
        # Linear forward
        quantization_config = CompressedTensorsConfig(run_compressed=False)
        cls.decompressed_model = pretrained_model_class.from_pretrained(
            cls.compressed_model_stub,
            torch_dtype=cls.compressed_model.dtype,
            device_map=cls.compressed_model.device,
            quantization_config=quantization_config,
        )

        def _run_dequant():
            attn_layers = ["q_proj", "v_proj", "o_proj", "k_proj"]
            mlp_layers = ["gate_proj", "up_proj", "down_proj"]
            for i in range(len(cls.compressed_model.model.layers)):

                layer = cls.compressed_model.model.layers[i]
                decompressed_layer = cls.decompressed_model.model.layers[i]

                attn = layer.self_attn 
                mlp = layer.mlp

                attn_decompressed = decompressed_layer.self_attn 
                mlp_decompressed = decompressed_layer.mlp

                for attn_attribute in attn_layers: 
                    attn_layer = getattr(attn, attn_attribute)
                    attn_layer_decomp = getattr(attn_decompressed, attn_attribute)

                    dequant = dequantize(x_q=attn_layer.weight, scale=attn_layer.weight_scale, zero_point=None, g_idx=None)
                    max_diff = torch.max(abs(dequant-attn_layer_decomp.weight))
                    assert max_diff == 0

                for mlp_attribute in mlp_layers: 
                    mlp_layer = getattr(mlp, mlp_attribute)
                    mlp_layer_decomp = getattr(mlp_decompressed, mlp_attribute)

                    dequant = dequantize(x_q=mlp_layer.weight, scale=mlp_layer.weight_scale, zero_point=None, g_idx=None)
                    max_diff = torch.max(abs(dequant-mlp_layer_decomp.weight))
                    assert max_diff == 0
        
        _run_dequant()
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
        shutil.rmtree(cls.test_dir)
        del cls.decompressed_model
        del cls.compressed_model
        torch.cuda.empty_cache()
