from compressed_tensors.linear.compressed_linear import CompressedLinear
from compressed_tensors.quantization.utils import iter_named_leaf_modules
from transformers import AutoModelForCausalLM

from llmcompressor.transformers.utils.helpers import load_quantized_model_decompressed


def test_load_quantized_model_decompressed():
    """run_compressed set to False module should be a Linear module"""

    MODEL_ID = "nm-testing/tinyllama-w8a8-compressed-hf-quantizer"

    model = load_quantized_model_decompressed(MODEL_ID)
    compressed_linear_counts = 0

    for _, submodule in iter_named_leaf_modules(
        model,
    ):
        if isinstance(submodule, CompressedLinear):
            compressed_linear_counts += 1

    assert compressed_linear_counts == 0


def test_load_quantized_model_compressed():
    """run_compressed set to True module should be a CompressedLinear module"""

    MODEL_ID = "nm-testing/tinyllama-w8a8-compressed-hf-quantizer"

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    compressed_linear_counts = 0

    # Some layers may be linear but not quantized. Ex. lm_head
    for _, submodule in iter_named_leaf_modules(
        model,
    ):
        if isinstance(submodule, CompressedLinear):
            compressed_linear_counts += 1

    assert compressed_linear_counts > 0
