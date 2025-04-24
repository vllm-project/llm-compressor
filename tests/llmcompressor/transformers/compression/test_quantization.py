import os
import shutil
import tempfile
import unittest

import pytest
import torch
from compressed_tensors.quantization.utils import is_module_quantized
from parameterized import parameterized_class
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DefaultDataCollator

from llmcompressor import oneshot
from llmcompressor.args import DatasetArguments
from llmcompressor.pytorch.utils import tensors_to_device
from llmcompressor.transformers.finetune.data import TextGenerationDataset
from tests.testing_utils import parse_params, requires_gpu

CONFIGS_DIRECTORY = "tests/llmcompressor/transformers/compression/configs"


@requires_gpu
@pytest.mark.integration
@parameterized_class(parse_params(CONFIGS_DIRECTORY))
class TestQuantizationMatches(unittest.TestCase):
    new_recipe = None
    ppl_threshold = None
    model_stub = None
    dataset = "ultrachat-200k"
    output = "tiny_llama_out"
    max_seq_length = 512
    weight_dtype = torch.float16
    num_eval = 64

    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp()

        cls.model = AutoModelForCausalLM.from_pretrained(
            cls.model_stub, torch_dtype=cls.weight_dtype, device_map="cuda:0"
        )
        model = cls._run_oneshot(
            cls.model,
            cls.new_recipe,
            cls.dataset,
            os.path.join(cls.test_dir, cls.output),
        )
        cls.session_model = model

    @classmethod
    def tearDownClass(cls):
        if os.path.isdir(cls.test_dir):
            shutil.rmtree(cls.test_dir)
        del cls.model
        torch.cuda.empty_cache()

    @staticmethod
    def _run_oneshot(model, recipe, dataset, output_dir):
        num_calibration_samples = 64
        max_seq_length = 512
        pad_to_max_length = False

        model = oneshot(
            model=model,
            dataset=dataset,
            output_dir=output_dir,
            max_seq_length=max_seq_length,
            num_calibration_samples=num_calibration_samples,
            recipe=recipe,
            pad_to_max_length=pad_to_max_length,
            splits={"calibration": "train_gen[:1%]"},
            save_compressed=False,
        )
        return model

    def _get_quant_info(self, model):
        quant_info_weights = {}
        quant_info_inputs = {}
        for name, module in model.named_modules():
            if is_module_quantized(module):
                if module.quantization_scheme.weights is not None:
                    quant_info_weights[name] = (
                        module.weight_scale,
                        module.weight_zero_point,
                        module.weight,
                    )

                if module.quantization_scheme.input_activations is not None:
                    is_dynamic = module.quantization_scheme.input_activations.dynamic
                    if not is_dynamic:
                        quant_info_inputs[name] = (
                            module.input_scale,
                            module.input_zero_point,
                        )

        return quant_info_weights, quant_info_inputs

    def test_quantization_reload(self):
        model_reloaded = AutoModelForCausalLM.from_pretrained(
            os.path.join(self.test_dir, self.output),
            torch_dtype="auto",
            device_map="cuda:0",
        )

        og_weights, og_inputs = self._get_quant_info(self.model)
        reloaded_weights, reloaded_inputs = self._get_quant_info(model_reloaded)

        for name, (o_scale, o_zp, o_weight) in og_weights.items():
            n_scale, n_zp, n_weight = reloaded_weights[name]
            assert o_scale.dtype == n_scale.dtype == self.weight_dtype
            assert torch.equal(o_scale, n_scale)
            assert o_zp.dtype == n_zp.dtype
            assert torch.equal(o_zp, n_zp)

            # we don't expect an exact match here because o_weight still has the
            # original weight and n_weight has been fake_quantized
            assert n_weight.dtype == o_weight.dtype == self.weight_dtype

        for name, (o_scale, o_zp) in og_inputs.items():
            n_scale, n_zp = reloaded_inputs[name]
            assert o_scale.dtype == n_scale.dtype == self.weight_dtype
            assert torch.equal(o_scale, n_scale)
            assert o_zp.dtype == n_zp.dtype
            assert torch.equal(o_zp, n_zp)

    def _get_dataloader(self, dataset_args, tokenizer):
        dataset_manager = TextGenerationDataset.load_from_registry(
            dataset_args.dataset,
            dataset_args=dataset_args,
            split="train_gen[:5%]",
            processor=tokenizer,
        )
        calib_dataset = dataset_manager()
        data_loader = DataLoader(
            calib_dataset,
            batch_size=1,
            collate_fn=DefaultDataCollator(),
            sampler=torch.utils.data.RandomSampler(calib_dataset),
        )

        return data_loader

    @torch.no_grad()
    def test_perplexity(self):
        if self.ppl_threshold is None:
            pytest.skip("Skipping perplexity calculation.")
        tokenizer = AutoTokenizer.from_pretrained(self.model_stub)
        dataset_args = DatasetArguments(
            dataset="ultrachat-200k",
            max_seq_length=self.max_seq_length,
        )
        dataloader = self._get_dataloader(dataset_args, tokenizer)

        total_ppl = 0.0
        total_non_nan = 0
        for idx, sample in enumerate(dataloader):
            if idx >= self.num_eval:
                break
            output = self.model(**tensors_to_device(sample, "cuda:0"))
            if torch.isnan(output.loss):
                continue
            total_ppl += torch.exp(output.loss).item()
            total_non_nan += 1

        avg_ppl = total_ppl / total_non_nan
        assert avg_ppl <= self.ppl_threshold
