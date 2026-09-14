import pytest
import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
)
from compressed_tensors.utils.safetensors_load import (
    load_tensors_from_inverse_weight_map,
)
from safetensors.torch import save_file

from llmcompressor.entrypoints.model_free.converter import ModelFreePtqConverter


def _get_block_config() -> QuantizationConfig:
    scheme = QuantizationScheme(
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
    return QuantizationConfig(config_groups={"group_0": scheme}, ignore=[])


@pytest.fixture
def mfptq():
    return ModelFreePtqConverter(config=_get_block_config())


def test_validate_raises_for_non_2d_linear_weight(tmp_path, mfptq):
    path = tmp_path / "bad_shape.safetensors"
    save_file({"model.layers.0.mlp.down_proj.weight": torch.ones(128)}, str(path))

    tensors = load_tensors_from_inverse_weight_map({str(path): []}, device="meta")
    with pytest.raises(ValueError, match="model.layers.0.mlp.down_proj.weight"):
        mfptq.validate(tensors)


def test_validate_does_not_raise_for_block_incompatible_shape(tmp_path, mfptq):
    path = tmp_path / "bad_block.safetensors"
    save_file(
        {"model.layers.0.mlp.down_proj.weight": torch.ones(17, 16)},
        str(path),
    )

    tensors = load_tensors_from_inverse_weight_map({str(path): []}, device="meta")
    mfptq.validate(tensors)
