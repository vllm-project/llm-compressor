import pytest
import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from safetensors.torch import save_file

from llmcompressor.entrypoints.model_free.process import ModelFreePtqConverter

@pytest.fixture
def mfptq():
    return ModelFreePtqConverter(scheme=_get_block_scheme(), ignore=[])

def _get_block_scheme() -> QuantizationScheme:
    return QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=8,
            type="float",
            strategy="block",
            symmetric=True,
            dynamic=False,
            block_structure=[16, 16],
        ),
    )


def test_validate_file_raises_for_non_2d_linear_weight(tmp_path, mfptq):
    path = tmp_path / "bad_shape.safetensors"
    save_file({"model.layers.0.mlp.down_proj.weight": torch.ones(128)}, str(path))

    with pytest.raises(ValueError, match="model.layers.0.mlp.down_proj.weight"):
        mfptq.validate_file({str(path): []}, None, "cpu")


def test_validate_file_does_not_raise_for_block_incompatible_shape(tmp_path, mfptq):
    path = tmp_path / "bad_block.safetensors"
    save_file(
        {"model.layers.0.mlp.down_proj.weight": torch.ones(17, 16)},
        str(path),
    )
    
    mfptq.validate_file({str(path): []}, None, "cpu")
