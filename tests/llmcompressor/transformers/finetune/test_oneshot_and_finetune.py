import os
import shutil
import unittest

import pytest
from compressed_tensors.compressors import ModelCompressor
from parameterized import parameterized_class
from transformers import AutoConfig

from llmcompressor.transformers.sparsification.compressed_tensors_utils import (
    get_model_compressor,
)
from tests.testing_utils import parse_params, requires_gpu

CONFIGS_DIRECTORY = "tests/llmcompressor/transformers/finetune/finetune_oneshot_configs"
GPU_CONFIGS_DIRECTORY = (
    "tests/llmcompressor/transformers/finetune/finetune_oneshot_configs/gpu"
)


class TestOneshotAndFinetune(unittest.TestCase):
    def _test_oneshot_and_finetune(self):
        from llmcompressor import oneshot, train

        splits = {"train": "train[:5%]", "calibration": "train[5%:10%]"}
        if self.dataset == "ultrachat-200k":
            splits = {"train": "train_gen[:5%]", "calibration": "train_gen[5%:10%]"}

        oneshot_args = dict(
            dataset=self.dataset,
            splits=splits,
            recipe=self.recipe,
            num_calibration_samples=64,
            dataset_config_name=self.dataset_config_name,
            concatenate_data=self.concat_txt,
            output_dir=self.output,
        )

        oneshot_model = oneshot(
            model=self.model,
            **oneshot_args,
            stage="test_oneshot_stage",
        )

        compressor = get_model_compressor(model=oneshot_model, save_compressed=True)
        if compressor is not None:
            compressor.decompress_model(oneshot_model)

        train_args = dict(
            num_train_epochs=self.num_train_epochs,
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
                os.path.join(self.output, "test_oneshot_stage")
            ).quantization_config
        )
        config_finetune_applied = ModelCompressor.parse_sparsity_config(
            AutoConfig.from_pretrained(
                os.path.join(self.output, "test_train_stage")
            ).quantization_config
        )
        # model is first sparsified, then finetuned, both should have the same sparsity
        assert config_sparse_applied["global_sparsity"] == pytest.approx(
            config_finetune_applied["global_sparsity"], abs=1e-5
        )

    def tearDown(self):
        # TODO: we get really nice stats from finetune that we should log
        # stored in results.json
        if os.path.isdir(self.output):
            shutil.rmtree(self.output)


@pytest.mark.integration
@parameterized_class(parse_params(CONFIGS_DIRECTORY))
class TestOneshotAndFinetuneSmall(TestOneshotAndFinetune):
    model = None
    dataset = None
    recipe = None
    dataset_config_name = None
    num_train_epochs = None
    concat_txt = None

    def setUp(self):
        import torch

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.output = "./finetune_output"

    def test_oneshot_then_finetune_small(self):
        self._test_oneshot_and_finetune()


@requires_gpu
@pytest.mark.integration
@parameterized_class(parse_params(GPU_CONFIGS_DIRECTORY))
class TestOneshotAndFinetuneGPU(TestOneshotAndFinetune):
    model = None
    dataset = None
    recipe = None
    dataset_config_name = None
    num_train_epochs = None
    concat_txt = None

    def setUp(self):
        import torch
        from transformers import AutoModelForCausalLM

        self.device = "cuda:0"
        self.output = "./finetune_output"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model, device_map=self.device, torch_dtype=torch.bfloat16
        )

    def test_oneshot_then_finetune_gpu(self):
        self._test_oneshot_and_finetune()
