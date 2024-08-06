import math
import shutil

import pytest
import torch
from compressed_tensors import COMPRESSION_CONFIG_NAME
from compressed_tensors.compressors import ModelCompressor
from compressed_tensors.config import BitmaskConfig, DenseSparsityConfig
from compressed_tensors.quantization import QuantizationStatus
from transformers import AutoConfig

from llmcompressor.core import reset_session
from llmcompressor.pytorch.utils.helpers import tensor_sparsity
from llmcompressor.transformers import SparseAutoModelForCausalLM, oneshot
from llmcompressor.transformers.compression.sparsity_config import (
    SparsityConfigMetadata,
)


@pytest.mark.parametrize(
    "compressed,config,dtype",
    [
        [True, None, torch.float32],
        [False, DenseSparsityConfig(), torch.float16],
        [True, BitmaskConfig(), torch.bfloat16],
        [False, BitmaskConfig(), torch.float32],
        [False, None, torch.float16],
    ],
)
def test_sparse_model_reload(compressed, config, dtype, tmp_path):
    recipe_str = "tests/llmcompressor/transformers/obcq/recipes/test_tiny2.yaml"
    expected_sparsity = 0.5
    model_path = "Xenova/llama2.c-stories15M"
    device = "cuda:0"
    if not torch.cuda.is_available():
        device = "cpu"
    dataset = "open_platypus"
    concatenate_data = False
    num_calibration_samples = 64
    output_dir = tmp_path / "oneshot_out"
    splits = {"calibration": "train[:10%]"}
    one_of_sparse_weights = "model.layers.1.mlp.up_proj.weight"

    # create a sparse model
    oneshot(
        model=model_path,
        dataset=dataset,
        output_dir=output_dir,
        num_calibration_samples=num_calibration_samples,
        recipe=recipe_str,
        concatenate_data=concatenate_data,
        splits=splits,
        oneshot_device=device,
        precision=dtype,
        clear_sparse_session=False,
    )

    model = SparseAutoModelForCausalLM.from_pretrained(
        tmp_path / "oneshot_out", torch_dtype=dtype
    )

    # assert that sample layer has the intended sparsity
    assert math.isclose(
        tensor_sparsity(model.state_dict()[one_of_sparse_weights]),
        expected_sparsity,
        rel_tol=1e-3,
    )
    inferred_structure = SparsityConfigMetadata.infer_sparsity_structure()
    assert inferred_structure == "0:0"

    model.save_pretrained(
        tmp_path / "compress_out",
        sparsity_config=config,
        save_compressed=compressed,
    )

    config = AutoConfig.from_pretrained(tmp_path / "compress_out")
    compression_config = getattr(config, COMPRESSION_CONFIG_NAME, None)
    sparsity_config = ModelCompressor.parse_sparsity_config(compression_config)
    assert (
        sparsity_config["format"] == "dense"
        if (not compressed and config is None)
        else "sparse_bitmask"
    )
    assert sparsity_config[
        "global_sparsity"
    ] == SparsityConfigMetadata.infer_global_sparsity(model)
    assert sparsity_config["sparsity_structure"] == inferred_structure

    dense_model = SparseAutoModelForCausalLM.from_pretrained(
        tmp_path / "compress_out", torch_dtype="auto"
    )

    og_state_dict = model.state_dict()
    reconstructed_state_dict = dense_model.state_dict()
    assert len(og_state_dict) == len(reconstructed_state_dict)
    for key in og_state_dict.keys():
        dense_tensor = og_state_dict[key]
        reconstructed_tensor = reconstructed_state_dict[key]
        assert dense_tensor.dtype == reconstructed_tensor.dtype == dtype
        assert torch.equal(dense_tensor, reconstructed_tensor)

    shutil.rmtree(tmp_path)


@pytest.mark.parametrize(
    "skip_compression_stats,save_compressed",
    [[True, True], [True, False], [False, True], [False, False]],
)
def test_dense_model_save(tmp_path, skip_compression_stats, save_compressed):
    reset_session()

    model_path = "Xenova/llama2.c-stories15M"
    model = SparseAutoModelForCausalLM.from_pretrained(model_path)

    inferred_global_sparsity = SparsityConfigMetadata.infer_global_sparsity(model)
    assert math.isclose(inferred_global_sparsity, 0.0, rel_tol=1e-3)
    inferred_structure = SparsityConfigMetadata.infer_sparsity_structure()
    assert inferred_structure == "unstructured"

    model.save_pretrained(
        tmp_path / "dense_out",
        skip_compression_stats=skip_compression_stats,
        save_compressed=save_compressed,
    )

    # for models with 0% sparsity no sparsity config is saved regardless
    config = AutoConfig.from_pretrained(tmp_path / "dense_out")
    compression_config = getattr(config, COMPRESSION_CONFIG_NAME, None)
    sparsity_config = ModelCompressor.parse_sparsity_config(compression_config)
    assert sparsity_config is None

    shutil.rmtree(tmp_path)


@pytest.mark.parametrize(
    "format,dtype",
    [
        ["dense", torch.float32],
        ["dense", torch.float16],
        ["int_quantized", torch.float32],
        # [True, "int_quantized", torch.float16],
    ],
)
def test_quant_model_reload(format, dtype, tmp_path):
    recipe_str = (
        "tests/llmcompressor/transformers/compression/recipes/new_quant_simple.yaml"
    )
    model_path = "Xenova/llama2.c-stories15M"
    device = "cuda:0"
    if not torch.cuda.is_available():
        device = "cpu"
    dataset = "open_platypus"
    concatenate_data = False
    num_calibration_samples = 64
    output_dir = tmp_path / "oneshot_out"
    splits = {"calibration": "train[:10%]"}

    # create a quantized model
    oneshot(
        model=model_path,
        dataset=dataset,
        output_dir=output_dir,
        num_calibration_samples=num_calibration_samples,
        recipe=recipe_str,
        concatenate_data=concatenate_data,
        splits=splits,
        oneshot_device=device,
        precision=dtype,
    )

    model = SparseAutoModelForCausalLM.from_pretrained(
        tmp_path / "oneshot_out", torch_dtype=dtype
    )

    for _, module in model.named_modules():
        if hasattr(module, "quantization_scheme"):
            assert module.weight.dtype == dtype
            assert module.quantization_status == QuantizationStatus.FROZEN

    model.save_pretrained(
        tmp_path / "compress_out",
        quantization_format=format,
        save_compressed=True,
    )

    config = AutoConfig.from_pretrained(tmp_path / "compress_out")
    compression_config = getattr(config, COMPRESSION_CONFIG_NAME, None)
    quant_config = ModelCompressor.parse_quantization_config(compression_config)
    assert quant_config["format"] == format

    dense_model = SparseAutoModelForCausalLM.from_pretrained(
        tmp_path / "compress_out", torch_dtype="auto"
    )

    og_state_dict = model.state_dict()
    reconstructed_state_dict = dense_model.state_dict()
    assert len(og_state_dict) == len(reconstructed_state_dict)
    for key in og_state_dict.keys():
        dense_tensor = og_state_dict[key]
        reconstructed_tensor = reconstructed_state_dict[key]
        assert dense_tensor.dtype == reconstructed_tensor.dtype
        if key.endswith("weight") and format != "dense":
            # we don't expect an exact match for compressed
            diff = torch.abs(dense_tensor - reconstructed_tensor)
            assert not torch.any(diff > 0.01).item()
        else:
            assert torch.equal(dense_tensor, reconstructed_tensor)

    shutil.rmtree(tmp_path)
