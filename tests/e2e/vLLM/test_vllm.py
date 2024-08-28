import shutil
import unittest

import pytest
from datasets import load_dataset
from parameterized import parameterized_class
from transformers import AutoTokenizer

from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.transformers import SparseAutoModelForCausalLM, oneshot
from tests.testing_utils import parse_params, requires_gpu, requires_torch

try:
    from vllm import LLM, SamplingParams

    vllm_installed = True
except ImportError:
    vllm_installed = False

# Defines the file paths to the directories containing the test configs
# for each of the quantization schemes
WNA16 = "tests/e2e/vLLM/configs/WNA16"
FP8 = "tests/e2e/vLLM/configs/FP8"
INT8 = "tests/e2e/vLLM/configs/INT8"


@requires_gpu
@requires_torch
@pytest.mark.skipif(not vllm_installed, reason="vLLM is not installed, skipping test")
@parameterized_class(parse_params([WNA16, FP8, INT8]))
class TestvLLM(unittest.TestCase):
    model = None
    scheme = None
    dataset_id = None
    dataset_split = None
    recipe = None

    def setUp(self):
        print("========== RUNNING ==============")
        print(self.scheme)

        self.save_dir = None
        self.device = "cuda:0"
        self.oneshot_kwargs = {}
        self.num_calibration_samples = 256
        self.max_seq_length = 1048
        self.prompts = [
            "The capital of France is",
            "The president of the US is",
            "My name is",
        ]

    def test_vllm(self):
        # Load model.
        loaded_model = SparseAutoModelForCausalLM.from_pretrained(
            self.model, device_map=self.device, torch_dtype="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model)

        def preprocess(example):
            return {
                "text": tokenizer.apply_chat_template(
                    example["messages"],
                    tokenize=False,
                )
            }

        def tokenize(sample):
            return tokenizer(
                sample["text"],
                padding=False,
                max_length=self.max_seq_length,
                truncation=True,
                add_special_tokens=False,
            )

        if self.dataset_id:
            ds = load_dataset(self.dataset_id, split=self.dataset_split)
            ds = ds.shuffle(seed=42).select(range(self.num_calibration_samples))
            ds = ds.map(preprocess)
            ds = ds.map(tokenize, remove_columns=ds.column_names)
            self.oneshot_kwargs["dataset"] = ds
            self.oneshot_kwargs["max_seq_length"] = self.max_seq_length
            self.oneshot_kwargs["num_calibration_samples"] = (
                self.num_calibration_samples
            )

        self.save_dir = self.model.split("/")[1] + f"-{self.scheme}"
        self.oneshot_kwargs["model"] = loaded_model
        if self.recipe:
            self.oneshot_kwargs["recipe"] = self.recipe
        else:
            # Test assumes that if a recipe was not provided, using
            # a compatible preset sceme from:
            # https://github.com/neuralmagic/compressed-tensors/blob/main/src/compressed_tensors/quantization/quant_scheme.py
            self.oneshot_kwargs["recipe"] = QuantizationModifier(
                targets="Linear", scheme=self.scheme, ignore=["lm_head"]
            )

        # Apply quantization.
        print("ONESHOT KWARGS", self.oneshot_kwargs)
        oneshot(
            **self.oneshot_kwargs,
            output_dir=self.save_dir,
            clear_sparse_session=True,
            oneshot_device=self.device,
        )
        tokenizer.save_pretrained(self.save_dir)
        # Run vLLM with saved model
        print("================= RUNNING vLLM =========================")
        sampling_params = SamplingParams(temperature=0.80, top_p=0.95)
        llm = LLM(model=self.save_dir)
        outputs = llm.generate(self.prompts, sampling_params)
        print("================= vLLM GENERATION ======================")
        for output in outputs:
            assert output
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print("PROMPT", prompt)
            print("GENERATED TEXT", generated_text)

    def tearDown(self):
        shutil.rmtree(self.save_dir)
