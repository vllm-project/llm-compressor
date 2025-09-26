import math
import os
import shutil

import pytest
import torch
from accelerate import dispatch_model
from accelerate.accelerator import get_state_dict_offloaded_model
from compressed_tensors import QUANTIZATION_CONFIG_NAME, CompressionFormat
from compressed_tensors.compressors import ModelCompressor
from compressed_tensors.config import BitmaskConfig, DenseSparsityConfig
from compressed_tensors.quantization import (
    QuantizationConfig,
    QuantizationStatus,
    quantize,
)
from compressed_tensors.utils import align_module_device, update_offload_parameter
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.utils.quantization_config import CompressedTensorsConfig

from llmcompressor import oneshot
from llmcompressor.core import reset_session
from llmcompressor.pytorch.utils.helpers import tensor_sparsity
from llmcompressor.transformers.compression.compressed_tensors_utils import (
    get_model_compressor,
    modify_save_pretrained,
    untie_word_embeddings,
)
from llmcompressor.transformers.compression.sparsity_metadata_config import (
    SparsityConfigMetadata,
)
from tests.testing_utils import requires_gpu


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
    recipe_str = "tests/llmcompressor/transformers/sparsegpt/recipes/test_tiny2.yaml"
    expected_sparsity = 0.5
    model_path = "nm-testing/tinysmokellama-3.2"
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
        precision=dtype,
        clear_sparse_session=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
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
    compression_config = getattr(config, QUANTIZATION_CONFIG_NAME, None)
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

    dense_model = AutoModelForCausalLM.from_pretrained(
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

    if os.path.isdir(tmp_path):
        shutil.rmtree(tmp_path)


@pytest.mark.parametrize(
    "skip_compression_stats,save_compressed",
    [[True, True], [True, False], [False, True], [False, False]],
)
def test_dense_model_save(tmp_path, skip_compression_stats, save_compressed):
    reset_session()

    model_path = "nm-testing/tinysmokellama-3.2"
    model = AutoModelForCausalLM.from_pretrained(model_path)

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
    compression_config = getattr(config, QUANTIZATION_CONFIG_NAME, None)
    sparsity_config = ModelCompressor.parse_sparsity_config(compression_config)
    assert sparsity_config is None

    if os.path.isdir(tmp_path):
        shutil.rmtree(tmp_path)


@pytest.mark.parametrize(
    "format,dtype",
    [
        ["dense", torch.float32],
        ["dense", torch.float16],
        # TODO: Int8 Decompression fails for transformers>4.49
        # ["int_quantized", torch.float32],
    ],
)
def test_quant_model_reload(format, dtype, tmp_path):
    recipe_str = (
        "tests/llmcompressor/transformers/compression/recipes/new_quant_simple.yaml"
    )
    model_path = "nm-testing/tinysmokellama-3.2"
    device = "cuda:0" if not torch.cuda.is_available() else "cpu"
    dataset = "open_platypus"
    concatenate_data = False
    num_calibration_samples = 16
    splits = {"calibration": "train[:10%]"}

    # create a quantized model
    model = oneshot(
        model=model_path,
        dataset=dataset,
        num_calibration_samples=num_calibration_samples,
        recipe=recipe_str,
        concatenate_data=concatenate_data,
        splits=splits,
        precision=dtype,
        clear_sparse_session=False,
    )

    # Fetch the oneshot model
    og_state_dict = model.state_dict()
    save_path_compressed = tmp_path / "compressed"

    for name, module in model.named_modules():
        if hasattr(module, "quantization_scheme"):
            assert (
                module.weight.dtype == dtype
            ), f"Module {name} has incorrect weight dtype"
            assert (
                module.quantization_status == QuantizationStatus.FROZEN
            ), f"Module {name} has incorrect quantization status"

    # Save to disk
    model.save_pretrained(
        save_path_compressed,
        quantization_format=format,
        save_compressed=True,
    )

    # Verify config on disk
    config = AutoConfig.from_pretrained(save_path_compressed)
    compression_config = getattr(config, QUANTIZATION_CONFIG_NAME, None)
    quant_config = ModelCompressor.parse_quantization_config(compression_config)
    assert quant_config["format"] == format

    decompressed_model = AutoModelForCausalLM.from_pretrained(
        save_path_compressed,
        torch_dtype=dtype,
        quantization_config=CompressedTensorsConfig(run_compressed=False),
    )

    reconstructed_state_dict = decompressed_model.state_dict()
    assert len(og_state_dict) == len(reconstructed_state_dict)
    for key in og_state_dict.keys():
        dense_tensor = og_state_dict[key].to(device)
        reconstructed_tensor = reconstructed_state_dict[key].to(device)
        assert dense_tensor.dtype == reconstructed_tensor.dtype
        if key.endswith("weight") and format != "dense":
            # we don't expect an exact match for compressed
            diff = torch.abs(dense_tensor - reconstructed_tensor)
            assert not torch.any(diff > 0.01).item()
        else:
            assert torch.equal(dense_tensor, reconstructed_tensor)
    if os.path.isdir(tmp_path):
        shutil.rmtree(tmp_path)


@pytest.mark.parametrize(
    "offload,torch_dtype,tie_word_embeddings,device",
    [
        # dtype
        (False, torch.float16, False, "cpu"),
        (False, torch.float16, True, "cpu"),
        (False, torch.float32, False, "cpu"),
        (False, torch.float32, True, "cpu"),
        # offloading
        (True, torch.float16, False, "cpu"),
        (True, torch.float32, False, "cpu"),
        (True, torch.float16, True, "cpu"),
        (True, torch.float32, True, "cpu"),
    ],
)
def test_model_reload(offload, torch_dtype, tie_word_embeddings, device, tmp_path):
    model_path = "nm-testing/tinysmokellama-3.2"
    save_path = tmp_path / "save_path"

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype)
    if offload:
        model = dispatch_model(model, {"": device}, force_hooks=True)
    else:
        model = model.to(device)

    if not tie_word_embeddings:
        untie_word_embeddings(model)

    modify_save_pretrained(model)
    model.save_pretrained(save_path, safe_serialization=True)

    reloaded = AutoModelForCausalLM.from_pretrained(save_path, torch_dtype="auto")

    model_dict = get_state_dict_offloaded_model(model)
    reloaded_dict = get_state_dict_offloaded_model(reloaded)
    assert model_dict.keys() == reloaded_dict.keys()
    for key in model_dict:
        assert torch.equal(model_dict[key].cpu(), reloaded_dict[key].cpu())


@requires_gpu
@pytest.mark.parametrize(
    "offload,torch_dtype,tie_word_embeddings,device",
    [
        (False, torch.float32, False, "cuda:0"),
        (True, torch.float32, False, "cuda:0"),
        (True, torch.float16, True, "cuda:0"),
        (True, torch.float32, True, "cuda:0"),
    ],
)
def test_model_reload_gpu(offload, torch_dtype, tie_word_embeddings, device, tmp_path):
    test_model_reload(offload, torch_dtype, tie_word_embeddings, device, tmp_path)


@pytest.mark.parametrize(
    "offload,torch_dtype,tie_word_embeddings,device",
    [
        (False, torch.float16, False, "cpu"),
        (False, torch.float32, False, "cpu"),
        (True, torch.float32, False, "cpu"),
        (False, torch.float16, True, "cpu"),
        (False, torch.float32, True, "cpu"),
        (True, torch.float16, True, "cpu"),
        (True, torch.float32, True, "cpu"),
    ],
)
def test_model_shared_tensors(offload, torch_dtype, tie_word_embeddings, device):
    # load model
    model_path = "nm-testing/tinysmokellama-3.2"
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype)
    if offload:
        model = dispatch_model(model, {"": device}, force_hooks=True)
    else:
        model = model.to(device)

    if not tie_word_embeddings:
        untie_word_embeddings(model)

    # modify lm head
    with torch.no_grad(), align_module_device(model.lm_head):
        update_offload_parameter(model.lm_head, "weight", model.lm_head.weight + 1)

    # check that embed_tokens is not modified
    model_dict = get_state_dict_offloaded_model(model)
    lm_head = model_dict["lm_head.weight"]
    embed_tokens = model_dict["model.embed_tokens.weight"]
    if tie_word_embeddings:
        assert torch.equal(lm_head, embed_tokens)
    else:
        assert not torch.equal(lm_head, embed_tokens)


@requires_gpu
@pytest.mark.parametrize(
    "offload,torch_dtype,tie_word_embeddings,device",
    [
        (False, torch.float32, False, "cuda:0"),
        (False, torch.float32, True, "cuda:0"),
    ],
)
def test_model_shared_tensors_gpu(offload, torch_dtype, tie_word_embeddings, device):
    test_model_shared_tensors(offload, torch_dtype, tie_word_embeddings, device)


@requires_gpu
@pytest.mark.parametrize(
    "model_stub, recipe, sparse_format, quant_format",
    [
        (
            "nm-testing/tinysmokellama-3.2",
            "tests/llmcompressor/transformers/compression/recipes/sparse_24_fp8.yaml",
            CompressionFormat.sparse_24_bitmask.value,
            CompressionFormat.float_quantized.value,
        ),
    ],
)
def test_compressor_stacking(model_stub, recipe, sparse_format, quant_format, tmp_path):
    from llmcompressor.pytorch.model_load.helpers import get_session_model

    device = "cuda:0" if not torch.cuda.is_available() else "cpu"
    dataset = "open_platypus"
    concatenate_data = False
    num_calibration_samples = 64
    splits = {"calibration": "train[:10%]"}

    oneshot(
        model=model_stub,
        dataset=dataset,
        num_calibration_samples=num_calibration_samples,
        recipe=recipe,
        concatenate_data=concatenate_data,
        splits=splits,
    )

    # Fetch the oneshot model
    model = get_session_model()
    og_state_dict = model.state_dict()
    path = tmp_path / "compressed"

    # As HFQuantizer doesn't decompress the model, use the compressor to decompress
    # the model instead
    compressor = ModelCompressor.from_pretrained_model(
        model, sparsity_config_or_format=sparse_format, quantization_format=quant_format
    )

    assert (
        compressor.sparsity_compressor is not None
    ), "Sparse compressor not initialized"
    assert compressor.sparsity_config.format == sparse_format

    assert (
        compressor.quantization_compressor is not None
    ), "Quantization compressor not initialized"

    compressor.compress_model(model)
    compressor.decompress_model(model)
    compressor.quantization_config.quantization_status = QuantizationStatus.FROZEN

    # Verify the abs difference between the decompressed model
    # and the original model
    reconstructed_state_dict = model.state_dict()
    for key in reconstructed_state_dict.keys():
        dense_tensor = og_state_dict[key].to(device)
        reconstructed_tensor = reconstructed_state_dict[key].to(device)
        assert dense_tensor.dtype == reconstructed_tensor.dtype
        if key.endswith("weight") and quant_format != "dense":
            # we don't expect an exact match for compressed
            diff = torch.abs(dense_tensor - reconstructed_tensor)
            # maximum quantization error as a result of compression is ~0.025
            assert not torch.any(diff > 0.025), f"Max diff: {torch.max(diff)}"
        else:
            assert torch.equal(dense_tensor, reconstructed_tensor)

    # Recompress and save; validate correct formats used
    model.save_pretrained(path)
    config = AutoConfig.from_pretrained(path)
    compression_config = getattr(config, QUANTIZATION_CONFIG_NAME, None)
    quant_config = ModelCompressor.parse_quantization_config(compression_config)
    sparsity_config = ModelCompressor.parse_sparsity_config(compression_config)
    assert quant_config["format"] == quant_format
    assert sparsity_config["format"] == sparse_format

    if os.path.isdir(tmp_path):
        shutil.rmtree(tmp_path)


@pytest.mark.parametrize(
    "model_stub, recipe, sparse_format",
    [
        (
            "nm-testing/tinysmokellama-3.2",
            "tests/llmcompressor/transformers/compression/recipes/sparse_24.yaml",
            CompressionFormat.sparse_24_bitmask.value,
        ),
    ],
)
def test_sparse_24_compressor_is_lossless(model_stub, recipe, sparse_format, tmp_path):
    device = "cuda:0" if not torch.cuda.is_available() else "cpu"
    dataset = "open_platypus"
    concatenate_data = False
    num_calibration_samples = 64
    splits = {"calibration": "train[:10%]"}
    empty_model = AutoModelForCausalLM.from_pretrained(model_stub, torch_dtype="auto")

    model = oneshot(
        model=model_stub,
        dataset=dataset,
        num_calibration_samples=num_calibration_samples,
        recipe=recipe,
        concatenate_data=concatenate_data,
        splits=splits,
        clear_sparse_session=False,
    )

    og_state_dict = model.state_dict()
    path = tmp_path / "compressed"

    # Compress and save
    model.save_pretrained(
        path,
        save_compressed=True,
    )

    # Verify config on disk
    config = AutoConfig.from_pretrained(path)
    compression_config = getattr(config, QUANTIZATION_CONFIG_NAME, None)

    # As HFQuantizer doesn't decompress the model, use the compressor to decompress
    # the model instead
    compressor = ModelCompressor.from_compression_config(compression_config)

    assert (
        compressor.sparsity_compressor is not None
    ), "Sparse compressor not initialized"
    assert compressor.sparsity_config.format == sparse_format

    compressor.decompress(model_path=path, model=empty_model)

    # Verify the abs difference between the decompressed model
    # and the original model
    reconstructed_state_dict = empty_model.state_dict()
    assert len(og_state_dict) == len(reconstructed_state_dict)
    for key in og_state_dict.keys():
        dense_tensor = og_state_dict[key].to(device)
        reconstructed_tensor = reconstructed_state_dict[key].to(device)
        assert dense_tensor.dtype == reconstructed_tensor.dtype
        if key.endswith("weight"):
            assert torch.equal(dense_tensor, reconstructed_tensor)
    if os.path.isdir(tmp_path):
        shutil.rmtree(tmp_path)


def test_disable_sparse_compression_flag(tmp_path):
    two_four_sparse_model_id = "nm-testing/llama2.c-stories42M-pruned2.4"
    two_four_sparse_model = AutoModelForCausalLM.from_pretrained(
        two_four_sparse_model_id, torch_dtype="auto"
    )
    modify_save_pretrained(two_four_sparse_model)

    save_path = tmp_path / "no_sparse_compression_model"
    sparsity_config = SparsityConfigMetadata.from_pretrained(
        two_four_sparse_model,
        sparsity_structure="2:4",
    )
    two_four_sparse_model.save_pretrained(
        save_path, disable_sparse_compression=True, sparsity_config=sparsity_config
    )

    config = AutoConfig.from_pretrained(save_path)
    quantization_config = getattr(config, QUANTIZATION_CONFIG_NAME, None)

    assert quantization_config
    sparsity_config = quantization_config.get("sparsity_config")

    assert sparsity_config
    assert sparsity_config["format"] == "dense"
    if os.path.isdir(tmp_path):
        shutil.rmtree(tmp_path)


class DummyLinearModel(nn.Module):
    """
    A dummy linear model for testing purposes, simulating a quantized linear layer.
    """

    def __init__(self, weights, weight_scale=None, weight_zero_point=None):
        super().__init__()
        out_features, in_features = weights.shape

        # Linear layer without bias
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.linear.weight = nn.Parameter(weights, requires_grad=True)

        # Attach scale and zero-point if provided
        if weight_scale is not None:
            self.linear.weight_scale = nn.Parameter(
                torch.tensor(weight_scale), requires_grad=False
            )
        if weight_zero_point is not None:
            self.linear.weight_zero_point = nn.Parameter(
                torch.tensor(weight_zero_point), requires_grad=False
            )

    def forward(self, x):
        return self.linear(x)


def _create_quantization_config(
    w_bits=8,
    w_type="int",
    w_strategy="tensor",
    quantize_activations=False,
    a_bits=8,
    a_type="int",
    a_strategy="tensor",
):
    """
    Create a quantization configuration for testing.
    """
    config_dict = {
        "global_compression_ratio": 1.0,
        "quant_method": "compressed-tensors",
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": w_bits,
                    "strategy": w_strategy,
                    "symmetric": True,
                    "type": w_type,
                },
            }
        },
    }

    if quantize_activations:
        config_dict["config_groups"]["group_0"]["input_activations"] = {
            "num_bits": a_bits,
            "strategy": a_strategy,
            "symmetric": True,
            "type": a_type,
        }

    return QuantizationConfig.model_validate(config_dict)


def _quantization_config_from_string(config_str, q_type):
    """
    Parse quantization config from string and type.
    """
    w_bits = int(config_str[1])
    a_bits = int(config_str[3:])
    quantize_activations = a_bits < 16

    return _create_quantization_config(
        w_bits=w_bits,
        w_type=q_type,
        w_strategy="channel",
        quantize_activations=quantize_activations,
        a_bits=a_bits,
        a_type=q_type,
        a_strategy="tensor",
    )


def _make_24_sparse(tensor):
    """
    Apply 2:4 sparsity pattern to the given tensor.
    """
    reshaped_tensor = tensor.view(tensor.size(0), -1, 4)
    mask = torch.zeros_like(reshaped_tensor, dtype=torch.bool)
    mask[..., :2] = True
    sparsified_tensor = torch.where(
        mask, reshaped_tensor, torch.tensor(0.0, dtype=tensor.dtype)
    )
    return sparsified_tensor.view_as(tensor)


@pytest.mark.parametrize(
    "quant_style, quant_type, is_24, expected_quant_compressor, "
    "expected_sparsity_compressor",
    [
        ("W8A8", "int", False, "int-quantized", "dense"),
        ("W4A16", "int", False, "pack-quantized", "dense"),
        ("W8A16", "int", False, "pack-quantized", "dense"),
        ("W8A8", "int", True, "int-quantized", "sparse-24-bitmask"),
        ("W4A16", "int", True, "marlin-24", "dense"),
        ("W8A16", "int", True, "marlin-24", "dense"),
        ("W8A8", "float", False, "float-quantized", "dense"),
        ("W8A16", "float", False, "naive-quantized", "dense"),
        ("W8A8", "float", True, "float-quantized", "sparse-24-bitmask"),
        ("W8A16", "float", True, "naive-quantized", "dense"),
    ],
)
def test_correct_compressor_inferred(
    quant_style,
    quant_type,
    is_24,
    expected_quant_compressor,
    expected_sparsity_compressor,
):
    """
    Test if the correct compressor is inferred based on
    quantization and sparsity configurations.
    """
    weights = torch.rand(10, 4)
    if is_24:
        weights = _make_24_sparse(weights)
    else:
        weights[0, :] = torch.ones(
            4,
        )  # guarantee not 24 sparse

    quantization_config = _quantization_config_from_string(quant_style, quant_type)
    quantization_args = quantization_config.config_groups["group_0"].weights

    scale = (
        torch.ones((weights.shape[0], 1))
        if quantization_args.strategy == "channel"
        else torch.tensor([1.0])
    )
    zero_point = torch.zeros_like(scale)

    quantized_weights = quantize(
        weights, scale=scale, zero_point=zero_point, args=quantization_args
    )

    model = DummyLinearModel(quantized_weights, scale, zero_point)
    model.linear.quantization_scheme = quantization_config.config_groups["group_0"]
    model.linear.quantization_status = QuantizationStatus.FROZEN

    if is_24:
        sparsity_config = SparsityConfigMetadata.from_pretrained(
            model, sparsity_structure="2:4", compress=True
        )
    else:
        sparsity_config = None
    compressor = get_model_compressor(model, sparsity_config=sparsity_config)

    assert compressor.quantization_config.format == expected_quant_compressor

    if expected_sparsity_compressor == "dense":
        assert (
            compressor.sparsity_config is None
            or compressor.sparsity_config.format == expected_sparsity_compressor
        )
    else:
        assert compressor.sparsity_config.format == expected_sparsity_compressor
