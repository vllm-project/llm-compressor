import gc
import math

import pytest
import torch
from transformers import AutoModelForCausalLM

from llmcompressor import oneshot
from llmcompressor.pytorch.utils.helpers import tensor_sparsity
from tests.testing_utils import parse_params, requires_gpu

CONFIGS_DIRECTORY = (
    "tests/llmcompressor/transformers/sparsegpt/sparsegpt_configs/sparse"
)
GPU_CONFIGS_DIRECTORY = (
    "tests/llmcompressor/transformers/sparsegpt/sparsegpt_configs/sparse/gpu"
)


@pytest.fixture
def _clear_cuda_cache():
    yield
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.integration
@pytest.mark.parametrize("config", parse_params(CONFIGS_DIRECTORY))
def test_sparsities(tmp_path, config):
    model = oneshot(
        model=config["model"],
        dataset=config["dataset"],
        recipe=config["recipe"],
        max_seq_length=128,
        num_calibration_samples=64,
        pad_to_max_length=False,
        output_dir=tmp_path,
    )

    layer_1_sparse = tensor_sparsity(model.model.layers[1].self_attn.k_proj.weight)
    assert math.isclose(layer_1_sparse.item(), config["sparsity"], rel_tol=1e-3)
    layer_2_dense = tensor_sparsity(model.model.layers[2].self_attn.k_proj.weight)
    assert math.isclose(layer_2_dense.item(), 0.0, rel_tol=1e-4)


# TODO: @Satrat and @dsikka, revisit if we want these nightly or weekly
@requires_gpu
@pytest.mark.integration
@pytest.mark.parametrize("config", parse_params(GPU_CONFIGS_DIRECTORY))
def test_sparsities_gpu(tmp_path, config, _clear_cuda_cache):
    model = AutoModelForCausalLM.from_pretrained(
        config["model"], device_map=config["device"], torch_dtype=torch.bfloat16
    )
    model = oneshot(
        model=model,
        dataset=config["dataset"],
        recipe=config["recipe"],
        max_seq_length=128,
        num_calibration_samples=64,
        pad_to_max_length=False,
        output_dir=tmp_path,
        precision="bfloat16",
    )

    layer_1_sparse = tensor_sparsity(model.model.layers[1].self_attn.k_proj.weight)
    assert math.isclose(layer_1_sparse.item(), config["sparsity"], rel_tol=1e-4)
    layer_2_dense = tensor_sparsity(model.model.layers[2].self_attn.k_proj.weight)
    assert math.isclose(layer_2_dense.item(), 0.0, abs_tol=1e-4)
