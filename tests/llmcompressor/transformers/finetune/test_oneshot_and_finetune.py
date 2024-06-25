import shutil
import unittest

import pytest
from parameterized import parameterized_class

from tests.testing_utils import parse_params, requires_gpu, requires_torch

CONFIGS_DIRECTORY = "tests/llmcompressor/transformers/finetune/finetune_oneshot_configs"
GPU_CONFIGS_DIRECTORY = (
    "tests/llmcompressor/transformers/finetune/finetune_oneshot_configs/gpu"
)


class TestOneshotAndFinetune(unittest.TestCase):
    def _test_oneshot_and_finetune(self):
        from llmcompressor.transformers import apply

        splits = {"train": "train[:50%]", "calibration": "train[50%:60%]"}
        if self.dataset == "ultrachat-200k":
            splits = {"train": "train_gen[:50%]", "calibration": "train_gen[50%:60%]"}

        apply(
            model=self.model,
            dataset=self.dataset,
            run_stages=True,
            output_dir=self.output,
            recipe=self.recipe,
            num_train_epochs=self.num_train_epochs,
            concatenate_data=self.concat_txt,
            splits=splits,
            oneshot_device=self.device,
            precision="bfloat16",
            bf16=True,
            dataset_config_name=self.dataset_config_name,
        )

    def tearDown(self):
        # TODO: we get really nice stats from finetune that we should log
        # stored in results.json
        shutil.rmtree(self.output)


@requires_torch
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


@requires_torch
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

        from llmcompressor.transformers import SparseAutoModelForCausalLM

        self.device = "auto"
        self.output = "./finetune_output"

        self.model = SparseAutoModelForCausalLM.from_pretrained(
            self.model, device_map=self.device, torch_dtype=torch.bfloat16
        )

    def test_oneshot_then_finetune_gpu(self):
        self._test_oneshot_and_finetune()
