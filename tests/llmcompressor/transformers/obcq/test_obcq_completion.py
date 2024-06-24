import shutil
import unittest

import pytest
from parameterized import parameterized_class

from tests.testing_utils import parse_params, requires_gpu, requires_torch

CONFIGS_DIRECTORY = "tests/llmcompressor/transformers/obcq/obcq_configs/completion"
GPU_CONFIGS_DIRECTORY = (
    "tests/llmcompressor/transformers/obcq/obcq_configs/completion/gpu"
)


class TestOBCQCompletion(unittest.TestCase):
    """
    Test for oneshot for quantization and quantization + sparsity. Sparsity-only tests
    can be found under `test_obcq_sparsity.py`
    """

    def labeled_dataloader(self, dataset_name, model_name):
        from torch.utils.data import DataLoader
        from transformers import DefaultDataCollator, AutoTokenizer

        from llmcompressor.transformers.finetune.data import TextGenerationDataset
        from llmcompressor.transformers.finetune.data.data_args import (
            DataTrainingArguments,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        data_args = DataTrainingArguments(
            dataset=dataset_name,
            max_seq_length=512,
            pad_to_max_length=False,
        )
        dataset_manager = TextGenerationDataset.load_from_registry(
            data_args.dataset,
            data_args=data_args,
            split="train",
            tokenizer=tokenizer,
        )
        calib_dataset = dataset_manager.tokenize_and_process(
            dataset_manager.get_raw_dataset()
        )
        data_loader = DataLoader(
            calib_dataset, batch_size=1, collate_fn=DefaultDataCollator()
        )

        return data_loader

    def _test_oneshot_completion(self, model_name: str = None):
        import torch

        from llmcompressor.pytorch.model_load.helpers import get_session_model
        from llmcompressor.pytorch.utils import tensors_to_device
        from llmcompressor.transformers import oneshot

        oneshot(
            model=self.model,
            dataset=self.dataset,
            oneshot_device=self.device,
            recipe=self.recipe,
            max_seq_length=512,
            num_calibration_samples=self.num_samples,
            pad_to_max_length=False,
            output_dir=self.output,
            clear_sparse_session=False,
            precision="bfloat16",
            bf16=True,
        )

        first_tiny_model = get_session_model()

        dataset = "open_platypus"

        iter = 10
        if model_name:
            dataloader = self.labeled_dataloader(dataset, model_name)
        else:
            dataloader = self.labeled_dataloader(dataset, self.model)

        total_new_ppl = 0.0
        for idx, sample in enumerate(dataloader):
            if idx >= iter:
                break

            with torch.no_grad():
                new_output = first_tiny_model(
                    **(tensors_to_device(sample, self.device))
                )
            new_ppl = torch.exp(new_output.loss)
            total_new_ppl += new_ppl

        avg_new_ppl = total_new_ppl / iter
        self.assertLess(avg_new_ppl, self.perplexity)

    def tearDown(self):
        shutil.rmtree(self.output)


@requires_torch
@pytest.mark.integration
@parameterized_class(parse_params(CONFIGS_DIRECTORY))
class TestOBCQCompletionSmall(TestOBCQCompletion):
    model = None
    dataset = None
    recipe = None
    sparsity = None
    num_samples = None
    perplexity = None

    def setUp(self):
        import torch

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.output = "./oneshot_output"

    def test_obcq_completion_small(self):
        self._test_oneshot_completion()


@requires_torch
@requires_gpu
@pytest.mark.integration
@parameterized_class(parse_params(GPU_CONFIGS_DIRECTORY))
class TestOBCQCompletionGPU(TestOBCQCompletion):
    model = None
    dataset = None
    recipe = None
    sparsity = None
    device = None
    num_samples = None
    perplexity = None

    def setUp(self):
        import torch

        from llmcompressor.transformers import SparseAutoModelForCausalLM

        self.model_name = None
        self.output = "./oneshot_output"

        # Temporary fix as oneshot seems to not work with zoo: models
        # Need to keep th model name for the perplexity calculation post oneshot
        self.model_name = self.model
        self.model = SparseAutoModelForCausalLM.from_pretrained(
            self.model, device_map=self.device, torch_dtype=torch.bfloat16
        )

    def test_oneshot_completion_gpu(self):
        self._test_oneshot_completion(model_name=self.model_name)
