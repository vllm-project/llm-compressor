import pytest
import torch
from compressed_tensors.compressors import compress_module
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStatus,
)
from compressed_tensors.quantization.lifecycle.initialize import (
    initialize_module_for_quantization,
)

from llmcompressor.pipelines.sequential.decompression import decompressed_modules


def _compressed_linear():
    lin = torch.nn.Linear(128, 256, bias=False).to(torch.bfloat16)
    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(num_bits=4, group_size=64, symmetric=True),
    )
    lin.quantization_scheme = scheme
    initialize_module_for_quantization(lin, scheme)
    lin.quantization_status = QuantizationStatus.FROZEN
    compress_module(lin)
    lin._ct_input_format = scheme.format
    return lin


def test_decompress_then_recompress_roundtrip():
    lin = _compressed_linear()
    with decompressed_modules([lin], recompress=True):
        assert hasattr(lin, "weight")
        assert lin.quantization_status == QuantizationStatus.DECOMPRESSED
    assert not hasattr(lin, "weight")
    assert lin.quantization_status == QuantizationStatus.COMPRESSED


def test_recompress_false_leaves_dense():
    lin = _compressed_linear()
    with decompressed_modules([lin], recompress=False):
        assert hasattr(lin, "weight")
    assert hasattr(lin, "weight")


def test_dense_module_is_noop():
    lin = torch.nn.Linear(8, 8)
    with decompressed_modules([lin], recompress=True):
        pass
    assert hasattr(lin, "weight")


def test_fp8linear_seam_raises():
    class FP8Linear(torch.nn.Linear):
        pass

    lin = FP8Linear(8, 8)
    lin.quantization_status = QuantizationStatus.COMPRESSED
    with pytest.raises(NotImplementedError, match="Path A"):
        with decompressed_modules([lin], recompress=True):
            pass
