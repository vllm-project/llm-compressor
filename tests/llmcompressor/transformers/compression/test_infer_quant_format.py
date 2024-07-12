import pytest
from compressed_tensors.config import SparsityCompressionConfig
from compressed_tensors.quantization import preset_name_to_scheme

from llmcompressor.transformers.compression.quantization_format import (
    infer_quantization_format,
)
from tests.llmcompressor.pytorch.helpers import LinearNet


@pytest.mark.parametrize(
    "preset,sparsity_structure,expected_format",
    [
        ["W8A8", "unstructured", "naive-quantized"],
        ["W8A16", "unstructured", "pack-quantized"],
        ["W8A16", "2:4", "marlin-24"],
        ["W4A16", "unstructured", "pack-quantized"],
        ["W4A16", "2:4", "marlin-24"],
        ["FP8", "unstructured", "naive-quantized"],
    ],
)
def test_infer_quant_format(preset, sparsity_structure, expected_format):
    sparsity_config = SparsityCompressionConfig(
        format="dense", sparsity_structure=sparsity_structure
    )
    quant_scheme = preset_name_to_scheme(preset, targets=["Linear"])

    dummy_model = LinearNet()
    for _, module in dummy_model.named_modules():
        module.quantization_scheme = quant_scheme

    inferred_format = infer_quantization_format(
        dummy_model, save_compressed=True, sparsity_config=sparsity_config
    )
    assert inferred_format.value == expected_format
