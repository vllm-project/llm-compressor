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
from transformers import AutoConfig

from llmcompressor.core import reset_session
from llmcompressor.pytorch.utils.helpers import tensor_sparsity
from llmcompressor.transformers import SparseAutoModelForCausalLM, oneshot
from llmcompressor.transformers.compression.sparsity_config import (
    SparsityConfigMetadata,
)


@pytest.mark.parametrize(
    "compressed,s_config,dtype",
    [
        [True, None, torch.float32],
        [False, DenseSparsityConfig(), torch.float16],
        [True, BitmaskConfig(), torch.bfloat16],
        [False, BitmaskConfig(), torch.float32],
        [False, None, torch.float16],
    ],
)
def test_sparse_model_reload(compressed, s_config, dtype, tmp_path):
    recipe_str = "tests/llmcompressor/transformers/obcq/recipes/test_tiny2.yaml"
    expected_sparsity = 0.5
    model_path = "Xenova/llama2.c-stories15M"
    device = "cuda:0"
    if not torch.cuda.is_available():
        device = "cpu"
    dataset = "open_platypus"
    concatenate_data = False
    num_calibration_samples = 64
    splits = {"calibration": "train[:10%]"}
    one_of_sparse_weights = "model.layers.1.mlp.up_proj.weight"

    # create a sparse model
    oneshot(
        model=model_path,
        dataset=dataset,
        output_dir=tmp_path / "oneshot_out",
        num_calibration_samples=num_calibration_samples,
        recipe=recipe_str,
        concatenate_data=concatenate_data,
        splits=splits,
        oneshot_device=device,
        precision=dtype,
        clear_sparse_session=False,
        save_compressed=False,
    )

    # load uncompressed model
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

    # save (compressed)
    model.save_pretrained(
        tmp_path / "compress_out",
        sparsity_config=s_config,
        save_compressed=compressed,
    )
    reloaded_model = SparseAutoModelForCausalLM.from_pretrained(
        tmp_path / "compress_out", torch_dtype="auto"
    )

    if compressed and s_config is not None and not isinstance(s_config, DenseSparsityConfig):
        # check (compressed) config
        compression_config = reloaded_model.config.quantization_config
        sparsity_config = compression_config.sparsity_config.dict()
        assert sparsity_config is not None

        _format = sparsity_config["format"]
        sparsity = sparsity_config["global_sparsity"]
        structure = sparsity_config["sparsity_structure"]

        assert _format == "sparse-bitmask"
        assert sparsity == SparsityConfigMetadata.infer_global_sparsity(model)
        assert structure == inferred_structure

        # check (compressed) state dict values
        compressor = ModelCompressor.from_compression_config(compression_config)
        compressor.decompress(
            model_path=tmp_path / "compress_out", model=reloaded_model
        )
        og_state_dict = model.state_dict()
        reloaded_state_dict = reloaded_model.state_dict()
        assert len(og_state_dict) == len(reloaded_state_dict)
        for key in og_state_dict.keys():
            og_tensor = og_state_dict[key]
            reloaded_tensor = reloaded_state_dict[key]
            assert og_tensor.dtype == reloaded_tensor.dtype == dtype
            assert torch.equal(og_tensor, reloaded_tensor)

    else:
        assert not hasattr(reloaded_model, "quantization_config")

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
    compression_config = getattr(config, QUANTIZATION_CONFIG_NAME, None)
    sparsity_config = ModelCompressor.parse_sparsity_config(compression_config)
    assert sparsity_config is None

    shutil.rmtree(tmp_path)


@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.bfloat16],
)
def test_quant_model_reload(dtype, tmp_path):
    recipe_str = (
        "tests/llmcompressor/transformers/compression/recipes/new_quant_simple.yaml"
    )
    model_path = "Xenova/llama2.c-stories15M"
    device = "cuda:0" if not torch.cuda.is_available() else "cpu"
    dataset = "open_platypus"
    concatenate_data = False
    num_calibration_samples = 64
    splits = {"calibration": "train[:10%]"}

    # create a quantized, compressed model
    oneshot(
        model=model_path,
        dataset=dataset,
        output_dir=tmp_path / "compressed",
        num_calibration_samples=num_calibration_samples,
        recipe=recipe_str,
        concatenate_data=concatenate_data,
        splits=splits,
        oneshot_device=device,
        precision=dtype,
        save_compressed=True,
    )

    compressed_model = SparseAutoModelForCausalLM.from_pretrained(
        tmp_path / "compressed", torch_dtype=dtype
    )

    compression_config = compressed_model.config.quantization_config
    quant_config = ModelCompressor.parse_quantization_config(compression_config)
    assert quant_config["format"] == "int-quantized"


    # create a quantized, uncompressed model to compare against
    reset_session()
    oneshot(
        model=model_path,
        dataset=dataset,
        output_dir=tmp_path / "uncompressed",
        num_calibration_samples=num_calibration_samples,
        recipe=recipe_str,
        concatenate_data=concatenate_data,
        splits=splits,
        oneshot_device=device,
        precision=dtype,
        save_compressed=False,
    )

    uncompressed_model = SparseAutoModelForCausalLM.from_pretrained(
        tmp_path / "uncompressed", torch_dtype=dtype
    )

    # unfortunately, this is how we have to load compressed models in an
    # uncompressed way so we can load their state dict
    compressed_model = SparseAutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype
    ).to(dtype=torch.float32)
    compressor = ModelCompressor.from_compression_config(compression_config)
    compressor.decompress(
        model_path=tmp_path / "compressed", model=uncompressed_model
    )

    # compare loaded values
    og_state_dict = get_state_dict_offloaded_model(uncompressed_model)
    reconstructed_state_dict = get_state_dict_offloaded_model(compressed_model)
    for key in og_state_dict.keys():
        dense_tensor = og_state_dict[key]
        reconstructed_tensor = reconstructed_state_dict[key]
        if key.endswith("weight"):
            breakpoint()
            #assert dense_tensor.dtype == reconstructed_tensor.dtype
            # we don't expect an exact match for compressed
            diff = torch.abs(dense_tensor - reconstructed_tensor)
            assert not torch.any(diff > 0.01).item()

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

    model = SparseAutoModelForCausalLM.from_pretrained(
        model_path,
        tie_word_embeddings=tie_word_embeddings,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    if offload:
        model = cpu_offload(model)

    model.save_pretrained(save_path, safe_serialization=True)

    reloaded = SparseAutoModelForCausalLM.from_pretrained(
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
<<<<<<< HEAD


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
    model = SparseAutoModelForCausalLM.from_pretrained(
        "Xenova/llama2.c-stories15M",
        torch_dtype=torch_dtype,
        tie_word_embeddings=tie_word_embeddings,
        device_map=device_map,
    )
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
=======
>>>>>>> origin
