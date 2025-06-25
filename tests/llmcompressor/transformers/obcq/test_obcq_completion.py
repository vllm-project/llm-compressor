import os
import shutil
import unittest

import pytest
from parameterized import parameterized_class

from tests.testing_utils import parse_params, requires_gpu

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
        from transformers import AutoTokenizer, DefaultDataCollator

        from llmcompressor.args import DatasetArguments
        from llmcompressor.transformers.finetune.data import TextGenerationDataset

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        dataset_args = DatasetArguments(
            dataset=dataset_name,
            max_seq_length=512,
            pad_to_max_length=False,
        )
        dataset_manager = TextGenerationDataset.load_from_registry(
            dataset_args.dataset,
            dataset_args=dataset_args,
            split=f"train[:{self.num_samples}]",
            processor=tokenizer,
        )
        calib_dataset = dataset_manager()
        data_loader = DataLoader(
            calib_dataset, batch_size=1, collate_fn=DefaultDataCollator()
        )

        return data_loader

    def _test_oneshot_completion(self, model_name: str = None):
        import torch

        from llmcompressor import oneshot
        from llmcompressor.pytorch.model_load.helpers import get_session_model
        from llmcompressor.pytorch.utils import tensors_to_device
        from llmcompressor.transformers.sparsification.compressed_tensors_utils import (
            get_model_compressor,  # avoid circular import
        )

        oneshot(
            model=self.model,
            dataset=self.dataset,
            splits={"calibration": f"train[:{self.num_samples}]"},
            recipe=self.recipe,
            max_seq_length=512,
            num_calibration_samples=self.num_samples,
            pad_to_max_length=False,
            output_dir=self.output,
            precision="bfloat16",
        )

        first_tiny_model = get_session_model()
        compressor = get_model_compressor(
            model=first_tiny_model,
            save_compressed=True,
            skip_sparsity_compression_stats=False,
        )
        if compressor is not None:
            compressor.decompress_model(first_tiny_model)

        dataset = "open_platypus"

        iter = 10
        if model_name:
            dataloader = self.labeled_dataloader(dataset, model_name)
        else:
            dataloader = self.labeled_dataloader(dataset, self.model)

        total_new_ppl = 0.0
        model_device = next(first_tiny_model.parameters()).device
        for idx, sample in enumerate(dataloader):
            if idx >= iter:
                break

            with torch.no_grad():
                new_output = first_tiny_model(
                    **(tensors_to_device(sample, model_device))
                )
            new_ppl = torch.exp(new_output.loss)
            total_new_ppl += new_ppl

        avg_new_ppl = total_new_ppl / iter
        self.assertLess(avg_new_ppl, self.perplexity)

    def tearDown(self):
        if os.path.isdir(self.output):
            shutil.rmtree(self.output)


@requires_gpu
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
        from transformers import AutoModelForCausalLM

        self.model_name = None
        self.output = "./oneshot_output"

        self.model_name = self.model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model, device_map=self.device, torch_dtype=torch.bfloat16
        )

    def test_oneshot_completion_gpu(self):
        self._test_oneshot_completion(model_name=self.model_name)
