import os
import shutil
import unittest

import pytest
from compressed_tensors.compressors import ModelCompressor
from parameterized import parameterized_class
from transformers import AutoConfig

from tests.testing_utils import parse_params, requires_gpu

CONFIGS_DIRECTORY = "tests/llmcompressor/transformers/finetune/finetune_oneshot_configs"
GPU_CONFIGS_DIRECTORY = (
    "tests/llmcompressor/transformers/finetune/finetune_oneshot_configs/gpu"
)


class TestOneshotAndFinetune(unittest.TestCase):
    def _test_post_train_and_finetune(self):
        from llmcompressor.transformers import apply

        splits = {"train": "train[:30%]", "calibration": "train[30%:40%]"}
        if self.dataset == "ultrachat-200k":
            splits = {"train": "train_gen[:30%]", "calibration": "train_gen[30%:40%]"}

        apply(
            model=self.model,
            dataset=self.dataset,
            run_stages=True,
            output_dir=self.output,
            recipe=self.recipe,
            num_train_epochs=self.num_train_epochs,
            concatenate_data=self.concat_txt,
            splits=splits,
            post_train_device=self.device,
            precision="bfloat16",
            bf16=True,
            dataset_config_name=self.dataset_config_name,
        )

        config_os = ModelCompressor.parse_sparsity_config(
            AutoConfig.from_pretrained(
                os.path.join(self.output, "stage_test_post_train")
            ).quantization_config
        )
        config_ft = ModelCompressor.parse_sparsity_config(
            AutoConfig.from_pretrained(
                os.path.join(self.output, "stage_test_post_train")
            ).quantization_config
        )
        assert config_ft["global_sparsity"] >= config_os["global_sparsity"]

    def tearDown(self):
        # TODO: we get really nice stats from finetune that we should log
        # stored in results.json
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

    def test_post_train_then_finetune_small(self):
        self._test_post_train_and_finetune()


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

    def test_post_train_then_finetune_gpu(self):
        self._test_post_train_and_finetune()
