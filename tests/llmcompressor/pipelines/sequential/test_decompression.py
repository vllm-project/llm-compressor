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

from llmcompressor.core import Event, EventType, State
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.pipelines.sequential.decompression import (
    decompressed_modules,
    ensure_dense_for_nonsequential,
    stash_input_formats,
)


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
        assert hasattr(lin, "weight")  # dense during the block
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


def test_quantization_initializes_only_after_subgraph_decompression():
    model = torch.nn.Sequential(_compressed_linear(), _compressed_linear())
    original_scale = model[1].weight_scale.detach().clone()
    original_scheme = model[1].quantization_scheme
    stash_input_formats(model, recompress=True)
    modifier = QuantizationModifier(scheme="W8A8")
    state = State(model=model)

    modifier.on_initialize(state)

    assert not hasattr(model[0], "weight")
    assert torch.equal(model[1].weight_scale, original_scale)
    assert model[1].quantization_scheme is original_scheme
    assert not hasattr(model[0], "_ct_compressed_qparams")

    with decompressed_modules([model[0]], recompress=False):
        modifier.on_sequential_epoch_start(
            state,
            Event(type_=EventType.SEQUENTIAL_EPOCH_START),
            modules=[model[0]],
        )
        assert hasattr(model[0], "weight")
        assert model[0].quantization_scheme.weights.num_bits == 8
        assert hasattr(model[0], "weight_observer")

    assert not hasattr(model[1], "weight")
    assert torch.equal(model[1].weight_scale, original_scale)
    assert model[1].quantization_scheme is original_scheme


def test_nonsequential_initializes_after_full_decompression():
    model = torch.nn.Sequential(_compressed_linear(), _compressed_linear())
    stash_input_formats(model, recompress=True)
    modifier = QuantizationModifier(scheme="W8A8")
    state = State(model=model)
    modifier.on_initialize(state)

    ensure_dense_for_nonsequential(model)
    modifier.on_calibration_start(state, Event(type_=EventType.CALIBRATION_START))

    assert all(hasattr(module, "weight") for module in model)
    assert all(module.quantization_scheme.weights.num_bits == 8 for module in model)
    assert all(hasattr(module, "weight_observer") for module in model)


def test_fp8linear_seam_raises():
    class FP8Linear(torch.nn.Linear):
        pass

    lin = FP8Linear(8, 8)
    lin.quantization_status = QuantizationStatus.COMPRESSED
    with pytest.raises(NotImplementedError, match="Path A"):
        with decompressed_modules([lin], recompress=True):
            pass


@pytest.mark.skip(
    reason="Offload<->decompress composition needs a real GPU onload device; the "
    "CPU-only onload is degenerate. Covered end-to-end by the sequential pipeline "
    "test which sets up onload/offload via the real machinery."
)
def test_roundtrip_under_offload():
    pass
