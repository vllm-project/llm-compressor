import os

import pytest
import torch
from compressed_tensors.compressors import ModelCompressor
from transformers import AutoConfig, AutoModelForCausalLM

from llmcompressor import oneshot, train
from llmcompressor.transformers.compression.compressed_tensors_utils import (
    get_model_compressor,
)
from tests.testing_utils import parse_params, requires_gpu

CONFIGS_DIRECTORY = "tests/llmcompressor/transformers/finetune/finetune_oneshot_configs"
GPU_CONFIGS_DIRECTORY = (
    "tests/llmcompressor/transformers/finetune/finetune_oneshot_configs/gpu"
)


def _test_oneshot_and_finetune(
    model, dataset, recipe, dataset_config_name, concat_txt, output, num_train_epochs
):
    splits = {"train": "train[:5%]", "calibration": "train[5%:10%]"}
    if dataset == "ultrachat-200k":
        splits = {"train": "train_gen[:5%]", "calibration": "train_gen[5%:10%]"}

    oneshot_args = dict(
        dataset=dataset,
        splits=splits,
        recipe=recipe,
        num_calibration_samples=64,
        dataset_config_name=dataset_config_name,
        concatenate_data=concat_txt,
        output_dir=output,
    )

    oneshot_model = oneshot(
        model=model,
        **oneshot_args,
        stage="test_oneshot_stage",
    )

    compressor = get_model_compressor(model=oneshot_model, save_compressed=True)
    if compressor is not None:
        compressor.decompress_model(oneshot_model)

    train_args = dict(
        num_train_epochs=num_train_epochs,
        precision="bfloat16",
        bf16=True,
    )
    train(
        model=oneshot_model,
        **oneshot_args,
        **train_args,
        stage="test_train_stage",
    )

    config_sparse_applied = ModelCompressor.parse_sparsity_config(
        AutoConfig.from_pretrained(
            os.path.join(output, "test_oneshot_stage")
        ).quantization_config
    )
    config_finetune_applied = ModelCompressor.parse_sparsity_config(
        AutoConfig.from_pretrained(
            os.path.join(output, "test_train_stage")
        ).quantization_config
    )
    # model is first sparsified, then finetuned, both should have the same sparsity
    assert config_sparse_applied["global_sparsity"] == pytest.approx(
        config_finetune_applied["global_sparsity"], abs=1e-5
    )


@pytest.mark.integration
@pytest.mark.parametrize("config", parse_params(CONFIGS_DIRECTORY))
def test_oneshot_and_finetune_small(config, tmp_path):
    model = config["model"]
    dataset = config["dataset"]
    recipe = config["recipe"]
    dataset_config_name = config.get("dataset_config_name")
    num_train_epochs = config["num_train_epochs"]
    concat_txt = config["concat_txt"]
    output = tmp_path / "finetune_output"

    _test_oneshot_and_finetune(
        model,
        dataset,
        recipe,
        dataset_config_name,
        concat_txt,
        output,
        num_train_epochs,
    )


@requires_gpu
@pytest.mark.integration
@pytest.mark.parametrize("config", parse_params(GPU_CONFIGS_DIRECTORY))
def test_oneshot_and_finetune_gpu(config, tmp_path):
    model = config["model"]
    dataset = config["dataset"]
    recipe = config["recipe"]
    dataset_config_name = config.get("dataset_config_name")
    num_train_epochs = config["num_train_epochs"]
    concat_txt = config["concat_txt"]
    output = tmp_path / "finetune_output"

    device = "cuda:0"
    model = AutoModelForCausalLM.from_pretrained(
        model, device_map=device, torch_dtype=torch.bfloat16
    )

    _test_oneshot_and_finetune(
        model,
        dataset,
        recipe,
        dataset_config_name,
        concat_txt,
        output,
        num_train_epochs,
    )
