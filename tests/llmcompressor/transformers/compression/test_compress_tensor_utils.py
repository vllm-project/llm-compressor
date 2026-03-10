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
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.utils.quantization_config import CompressedTensorsConfig

from llmcompressor import oneshot
from llmcompressor.core import reset_session
from llmcompressor.pytorch.utils.helpers import tensor_sparsity
from llmcompressor.transformers.compression.compressed_tensors_utils import (
    _graft_extra_weights,
    get_model_compressor,
    modify_save_pretrained,
)
from llmcompressor.transformers.compression.sparsity_metadata_config import (
    SparsityConfigMetadata,
)
from llmcompressor.utils import untie_word_embeddings
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
        tie_word_embeddings=False,
    )

    model = AutoModelForCausalLM.from_pretrained(tmp_path / "oneshot_out", dtype=dtype)

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
        tmp_path / "compress_out", dtype="auto"
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
        tie_word_embeddings=False,
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
        dtype=dtype,
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
    "offload,dtype,tie_word_embeddings,device",
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
def test_model_reload(offload, dtype, tie_word_embeddings, device, tmp_path):
    model_path = "nm-testing/tinysmokellama-3.2"
    save_path = tmp_path / "save_path"

    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype)
    if offload:
        model = dispatch_model(model, {"": device}, force_hooks=True)
    else:
        model = model.to(device)

    if not tie_word_embeddings:
        untie_word_embeddings(model)

    modify_save_pretrained(model)
    model.save_pretrained(save_path, safe_serialization=True)

    reloaded = AutoModelForCausalLM.from_pretrained(save_path, dtype="auto")

    model_dict = get_state_dict_offloaded_model(model)
    reloaded_dict = get_state_dict_offloaded_model(reloaded)
    assert model_dict.keys() == reloaded_dict.keys()
    for key in model_dict:
        assert torch.equal(model_dict[key].cpu(), reloaded_dict[key].cpu())


@requires_gpu
@pytest.mark.parametrize(
    "offload,dtype,tie_word_embeddings,device",
    [
        (False, torch.float32, False, "cuda:0"),
        (True, torch.float32, False, "cuda:0"),
        (True, torch.float16, True, "cuda:0"),
        (True, torch.float32, True, "cuda:0"),
    ],
)
def test_model_reload_gpu(offload, dtype, tie_word_embeddings, device, tmp_path):
    test_model_reload(offload, dtype, tie_word_embeddings, device, tmp_path)


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
    empty_model = AutoModelForCausalLM.from_pretrained(model_stub, dtype="auto")

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
        two_four_sparse_model_id, dtype="auto"
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


class _MockModel:
    """Minimal stand-in for a PreTrainedModel, providing the attributes
    that _graft_extra_weights inspects: ``name_or_path`` and ``named_modules``.

    *module_names* should be the set of dotted module paths that exist in the
    model (e.g. ``{"", "model", "model.layer"}``).  The empty string represents
    the root module and is always included.
    """

    def __init__(self, name_or_path: str, module_names: set[str]):
        self.name_or_path = name_or_path
        self._module_names = module_names | {""}

    def named_modules(self):
        for name in self._module_names:
            yield name, None


class TestGraftExtraWeights:
    """Tests for _graft_extra_weights() — the function that copies weights
    present in the source checkpoint but missing from the output directory."""

    def _make_source_checkpoint(self, tmp_path, keys, multi_shard=False):
        """Create a fake source checkpoint (local path) with the given keys."""
        import json

        from safetensors.torch import save_file

        source_dir = tmp_path / "source"
        source_dir.mkdir()

        tensors = {k: torch.randn(4, 4) for k in keys}

        if multi_shard:
            # Split into two shards
            half = len(keys) // 2 or 1
            shard1_keys = keys[:half]
            shard2_keys = keys[half:]

            shard1 = {k: tensors[k] for k in shard1_keys}
            shard2 = {k: tensors[k] for k in shard2_keys}

            save_file(shard1, str(source_dir / "model-00001-of-00002.safetensors"))
            save_file(shard2, str(source_dir / "model-00002-of-00002.safetensors"))

            weight_map = {}
            for k in shard1_keys:
                weight_map[k] = "model-00001-of-00002.safetensors"
            for k in shard2_keys:
                weight_map[k] = "model-00002-of-00002.safetensors"

            total_size = sum(
                t.nelement() * t.element_size() for t in tensors.values()
            )
            index = {
                "metadata": {"total_size": total_size},
                "weight_map": weight_map,
            }
            with open(source_dir / "model.safetensors.index.json", "w") as f:
                json.dump(index, f)
        else:
            save_file(tensors, str(source_dir / "model.safetensors"))

        return str(source_dir), tensors

    def _make_output_dir(self, tmp_path, keys, tensors, multi_shard=False):
        """Create a fake output directory with a subset of keys."""
        import json

        from safetensors.torch import save_file

        out_dir = tmp_path / "output"
        out_dir.mkdir()

        subset = {k: tensors[k] for k in keys}

        if multi_shard:
            save_file(subset, str(out_dir / "model-00001-of-00001.safetensors"))
            weight_map = {
                k: "model-00001-of-00001.safetensors" for k in keys
            }
            total_size = sum(
                t.nelement() * t.element_size() for t in subset.values()
            )
            index = {
                "metadata": {"total_size": total_size},
                "weight_map": weight_map,
            }
            with open(out_dir / "model.safetensors.index.json", "w") as f:
                json.dump(index, f)
        else:
            save_file(subset, str(out_dir / "model.safetensors"))

        return str(out_dir)

    def _mock_model(self, source_dir, module_names):
        """Create a _MockModel pointing at source_dir with the given modules."""
        return _MockModel(source_dir, set(module_names))

    def test_noop_when_all_keys_present(self, tmp_path):
        """No extra file should be created when source and output match."""
        all_keys = ["model.layer.weight", "model.layer.bias"]
        source_dir, tensors = self._make_source_checkpoint(tmp_path, all_keys)
        out_dir = self._make_output_dir(tmp_path, all_keys, tensors)

        mock = self._mock_model(source_dir, ["model", "model.layer"])
        _graft_extra_weights(mock, out_dir)

        assert not os.path.exists(os.path.join(out_dir, "extra_weights.safetensors"))
        assert not os.path.exists(
            os.path.join(out_dir, "model.safetensors.index.json")
        )

    def test_grafts_missing_keys_single_shard(self, tmp_path):
        """Extra keys from a single-shard source should be grafted."""
        from safetensors import safe_open

        all_keys = ["model.layer.weight", "mtp.0.layer.weight", "mtp.1.layer.weight"]
        output_keys = ["model.layer.weight"]

        source_dir, tensors = self._make_source_checkpoint(tmp_path, all_keys)
        out_dir = self._make_output_dir(tmp_path, output_keys, tensors)

        # Model has "model.layer" but NOT "mtp.0.layer" or "mtp.1.layer"
        mock = self._mock_model(source_dir, ["model", "model.layer"])
        _graft_extra_weights(mock, out_dir)

        # extra_weights.safetensors should exist
        extra_path = os.path.join(out_dir, "extra_weights.safetensors")
        assert os.path.exists(extra_path)

        with safe_open(extra_path, framework="pt") as f:
            extra_keys = set(f.keys())
        assert extra_keys == {"mtp.0.layer.weight", "mtp.1.layer.weight"}

        # An index should have been created (single-shard output + new shard)
        index_path = os.path.join(out_dir, "model.safetensors.index.json")
        assert os.path.exists(index_path)

        import json

        with open(index_path) as f:
            index = json.load(f)
        wm = index["weight_map"]
        assert wm["model.layer.weight"] == "model.safetensors"
        assert wm["mtp.0.layer.weight"] == "extra_weights.safetensors"
        assert wm["mtp.1.layer.weight"] == "extra_weights.safetensors"

    def test_grafts_missing_keys_multi_shard(self, tmp_path):
        """Extra keys from a multi-shard source with multi-shard output."""
        import json

        from safetensors import safe_open

        all_keys = [
            "model.layer0.weight",
            "model.layer1.weight",
            "mtp.0.layer.weight",
            "mtp.1.layer.weight",
        ]
        output_keys = ["model.layer0.weight", "model.layer1.weight"]

        source_dir, tensors = self._make_source_checkpoint(
            tmp_path, all_keys, multi_shard=True
        )
        out_dir = self._make_output_dir(
            tmp_path, output_keys, tensors, multi_shard=True
        )

        mock = self._mock_model(
            source_dir, ["model", "model.layer0", "model.layer1"]
        )
        _graft_extra_weights(mock, out_dir)

        extra_path = os.path.join(out_dir, "extra_weights.safetensors")
        assert os.path.exists(extra_path)

        with safe_open(extra_path, framework="pt") as f:
            extra_keys = set(f.keys())
        assert extra_keys == {"mtp.0.layer.weight", "mtp.1.layer.weight"}

        # Index should be updated (not created fresh)
        index_path = os.path.join(out_dir, "model.safetensors.index.json")
        with open(index_path) as f:
            index = json.load(f)

        wm = index["weight_map"]
        assert wm["mtp.0.layer.weight"] == "extra_weights.safetensors"
        assert wm["mtp.1.layer.weight"] == "extra_weights.safetensors"
        # Original keys still map to original shard
        assert wm["model.layer0.weight"] == "model-00001-of-00001.safetensors"

    def test_tensors_are_identical(self, tmp_path):
        """Grafted tensors should be bit-identical to source tensors."""
        from safetensors import safe_open

        all_keys = ["model.w", "extra.w"]
        output_keys = ["model.w"]

        source_dir, tensors = self._make_source_checkpoint(tmp_path, all_keys)
        out_dir = self._make_output_dir(tmp_path, output_keys, tensors)

        # "extra" module doesn't exist in the model
        mock = self._mock_model(source_dir, ["model"])
        _graft_extra_weights(mock, out_dir)

        extra_path = os.path.join(out_dir, "extra_weights.safetensors")
        with safe_open(extra_path, framework="pt") as f:
            grafted = f.get_tensor("extra.w")

        assert torch.equal(grafted, tensors["extra.w"])

    def test_total_size_correct_single_shard(self, tmp_path):
        """total_size in the created index should equal tensor byte counts,
        not file sizes on disk."""
        import json

        all_keys = ["model.w", "extra.w"]
        output_keys = ["model.w"]

        source_dir, tensors = self._make_source_checkpoint(tmp_path, all_keys)
        out_dir = self._make_output_dir(tmp_path, output_keys, tensors)

        mock = self._mock_model(source_dir, ["model"])
        _graft_extra_weights(mock, out_dir)

        index_path = os.path.join(out_dir, "model.safetensors.index.json")
        with open(index_path) as f:
            index = json.load(f)

        expected_total = sum(
            t.nelement() * t.element_size() for t in tensors.values()
        )
        assert index["metadata"]["total_size"] == expected_total

    def test_skips_renamed_keys_from_compression(self, tmp_path):
        """Keys renamed by compression (weight → weight_packed) must NOT be
        grafted back, even though they appear in source but not in output.
        The parent module still exists in the model, so we know the key was
        renamed rather than dropped."""
        # Source has original weight
        all_keys = [
            "model.layer.weight",
            "mtp.0.layer.weight",
        ]
        # Output has compressed variant (weight_packed) plus MTP is missing
        output_keys_on_disk = ["model.layer.weight_packed"]

        source_dir, tensors = self._make_source_checkpoint(tmp_path, all_keys)

        # Manually create output with the compressed key name
        from safetensors.torch import save_file

        out_dir = tmp_path / "output"
        out_dir.mkdir()
        save_file(
            {"model.layer.weight_packed": torch.randn(4, 4)},
            str(out_dir / "model.safetensors"),
        )

        # Model has "model.layer" (the module exists!) but NOT "mtp.0.layer"
        mock = self._mock_model(source_dir, ["model", "model.layer"])
        _graft_extra_weights(mock, str(out_dir))

        extra_path = os.path.join(str(out_dir), "extra_weights.safetensors")
        assert os.path.exists(extra_path)

        from safetensors import safe_open

        with safe_open(extra_path, framework="pt") as f:
            grafted_keys = set(f.keys())

        # Only MTP should be grafted, NOT the renamed model.layer.weight
        assert grafted_keys == {"mtp.0.layer.weight"}

    def test_noop_when_output_has_no_safetensors(self, tmp_path):
        """Must not graft when the output is in .bin format
        (safe_serialization=False).  Grafting here would silently replace the
        quantized weights with uncompressed originals."""
        all_keys = ["model.w", "mtp.w"]
        source_dir, tensors = self._make_source_checkpoint(tmp_path, all_keys)

        out_dir = tmp_path / "output"
        out_dir.mkdir()
        # Simulate safe_serialization=False: only .bin files
        torch.save(
            {"model.w": tensors["model.w"]},
            str(out_dir / "pytorch_model.bin"),
        )

        mock = self._mock_model(source_dir, ["model"])
        _graft_extra_weights(mock, str(out_dir))

        assert not os.path.exists(os.path.join(str(out_dir), "extra_weights.safetensors"))
        assert not os.path.exists(
            os.path.join(str(out_dir), "model.safetensors.index.json")
        )

    def test_noop_for_nonexistent_source(self, tmp_path):
        """Should silently skip when source can't be resolved."""
        from safetensors.torch import save_file

        out_dir = tmp_path / "output"
        out_dir.mkdir()
        save_file(
            {"model.w": torch.randn(4, 4)}, str(out_dir / "model.safetensors")
        )

        mock = _MockModel("/nonexistent/path/to/model", {"model"})
        _graft_extra_weights(mock, str(out_dir))
