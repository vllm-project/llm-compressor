import logging
import math
import shutil

import pytest
import torch
from accelerate import cpu_offload
from accelerate.accelerator import get_state_dict_offloaded_model
from compressed_tensors import QUANTIZATION_CONFIG_NAME
from compressed_tensors.compressors import ModelCompressor
from compressed_tensors.config import BitmaskConfig, DenseSparsityConfig
from compressed_tensors.quantization import QuantizationStatus
from compressed_tensors.utils import get_offloaded_device, update_prefix_dict
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.utils.quantization_config import CompressedTensorsConfig

from llmcompressor.core import reset_session
from llmcompressor.pytorch.utils.helpers import tensor_sparsity
from llmcompressor.transformers import oneshot
from llmcompressor.transformers.compression.sparsity_config import (
    SparsityConfigMetadata,
)
from llmcompressor.transformers.sparsification.compressed_tensors_utils import (
    modify_save_pretrained,
    patch_tied_tensors_bug,
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
    oneshot_calibrator = oneshot(
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

    # temporarily set the log level to error, to ignore printing out long missing
    # and unexpected key error messages (these are EXPECTED for quantized models)
    transformers_logger = logging.getLogger("transformers.modeling_utils")
    restore_log_level = transformers_logger.getEffectiveLevel()
    transformers_logger.setLevel(level=logging.ERROR)

    model = AutoModelForCausalLM.from_pretrained(
        tmp_path / "oneshot_out", torch_dtype=dtype
    )

    # restore transformers logging level now that model shell is loaded
    transformers_logger.setLevel(level=restore_log_level)

    # assert that sample layer has the intended sparsity
    assert math.isclose(
        tensor_sparsity(model.state_dict()[one_of_sparse_weights]),
        expected_sparsity,
        rel_tol=1e-3,
    )

    inferred_structure = SparsityConfigMetadata.infer_sparsity_structure(
        model, oneshot_calibrator.lifecycle.modifiers
    )
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

    shutil.rmtree(tmp_path)


@pytest.mark.parametrize(
    "skip_compression_stats,save_compressed",
    [[True, True], [True, False], [False, True], [False, False]],
)
def test_dense_model_save(tmp_path, skip_compression_stats, save_compressed):
    reset_session()

    model_path = "Xenova/llama2.c-stories15M"
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

    shutil.rmtree(tmp_path)


@pytest.mark.parametrize(
    "format,dtype",
    [
        ["dense", torch.float32],
        ["dense", torch.float16],
        ["int_quantized", torch.float32],
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
    num_calibration_samples = 1
    splits = {"calibration": "train[:10%]"}

    # create a quantized model
    oneshot_compressor = oneshot(
        model=model_path,
        dataset=dataset,
        num_calibration_samples=num_calibration_samples,
        recipe=recipe_str,
        concatenate_data=concatenate_data,
        splits=splits,
        oneshot_device=device,
        clear_sparse_session=False,
        precision=dtype,
    )

    # Fetch the oneshot model
    model = oneshot_compressor.model
    og_state_dict = model.state_dict()
    save_path_compressed = tmp_path / "compressed"

    for _, module in model.named_modules():
        if hasattr(module, "quantization_scheme"):
            assert module.weight.dtype == dtype
            assert module.quantization_status == QuantizationStatus.FROZEN

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
    shutil.rmtree(tmp_path)


# technically only tie_word_embeddings=False is supported right now
# setting to True is discouraged
@pytest.mark.parametrize(
    "offload,torch_dtype,tie_word_embeddings,device_map",
    [
        # dtype
        (False, torch.float16, False, "cpu"),
        (False, torch.float16, True, "cpu"),
        (False, torch.float32, False, "cpu"),
        (False, torch.float32, True, "cpu"),
        # offloading
        (True, torch.float16, False, "cpu"),
        (True, torch.float32, False, "cpu"),
        # (True, torch.float16, True, "cpu"),  # TODO: fails
        # (True, torch.float32, True, "cpu"),  # TODO: fails
    ],
)
def test_model_reload(offload, torch_dtype, tie_word_embeddings, device_map, tmp_path):
    model_path = "Xenova/llama2.c-stories15M"
    save_path = tmp_path / "save_path"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        tie_word_embeddings=tie_word_embeddings,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    if offload:
        model = cpu_offload(model)

    patch_tied_tensors_bug(model)
    modify_save_pretrained(model)
    model.save_pretrained(save_path, safe_serialization=True)

    reloaded = AutoModelForCausalLM.from_pretrained(
        save_path, torch_dtype="auto", device_map="cpu"
    )

    model_dict = get_state_dict_offloaded_model(model)
    reloaded_dict = get_state_dict_offloaded_model(reloaded)
    assert model_dict.keys() == reloaded_dict.keys()
    for key in model_dict:
        assert torch.equal(model_dict[key].cpu(), reloaded_dict[key].cpu())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires gpu")
@pytest.mark.parametrize(
    "offload,torch_dtype,tie_word_embeddings,device_map",
    [
        (False, torch.float32, False, "cuda:0"),
        (True, torch.float32, False, "cuda:0"),
        (True, torch.float16, True, "cuda:0"),
        (True, torch.float32, True, "cuda:0"),
    ],
)
def test_model_reload_gpu(
    offload, torch_dtype, tie_word_embeddings, device_map, tmp_path
):
    test_model_reload(offload, torch_dtype, tie_word_embeddings, device_map, tmp_path)


@pytest.mark.parametrize(
    "offload,torch_dtype,tie_word_embeddings,device_map",
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
def test_model_shared_tensors(
    offload, torch_dtype, tie_word_embeddings, device_map, tmp_path
):
    # load model
    model = AutoModelForCausalLM.from_pretrained(
        "Xenova/llama2.c-stories15M",
        torch_dtype=torch_dtype,
        tie_word_embeddings=tie_word_embeddings,
        device_map=device_map,
    )
    patch_tied_tensors_bug(model)

    if offload:
        model = cpu_offload(model)

    # modify lm head
    with torch.no_grad():
        if offload:
            model.lm_head._hf_hook.pre_forward(model.lm_head)

        model.lm_head.weight += 1

        if offload:
            device = get_offloaded_device(model.lm_head)
            update_prefix_dict(model.lm_head, "weight", model.lm_head.weight.to(device))
            model.lm_head._hf_hook.post_forward(model.lm_head, None)

    # check that embed_tokens is not modified
    model_dict = get_state_dict_offloaded_model(model)
    lm_head = model_dict["lm_head.weight"]
    embed_tokens = model_dict["model.embed_tokens.weight"]
    if tie_word_embeddings:
        assert torch.equal(lm_head, embed_tokens)
    else:
        assert not torch.equal(lm_head, embed_tokens)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires gpu")
@pytest.mark.parametrize(
    "offload,torch_dtype,tie_word_embeddings,device_map",
    [
        (False, torch.float32, False, "cuda:0"),
        (False, torch.float32, True, "cuda:0"),
    ],
)
def test_model_shared_tensors_gpu(
    offload, torch_dtype, tie_word_embeddings, device_map, tmp_path
):
    test_model_shared_tensors(
        offload, torch_dtype, tie_word_embeddings, device_map, tmp_path
    )
