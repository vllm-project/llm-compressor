import shutil
import unittest

import pytest
from datasets import load_dataset
from parameterized import parameterized_class
from transformers import AutoTokenizer

from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.transformers import SparseAutoModelForCausalLM, oneshot
from tests.testing_utils import parse_params, requires_gpu, requires_torch

CONFIGS_DIRECTORY = "tests/e2e/vLLM/configs"

try:
    from vllm import LLM, SamplingParams

    vllm_installed = True
except ImportError:
    vllm_installed = False


@requires_gpu
@requires_torch
@pytest.mark.skipif(not vllm_installed, reason="vLLM is not installed, skipping test")
@parameterized_class(parse_params(CONFIGS_DIRECTORY))
class TestvLLM(unittest.TestCase):
    model = None
    scheme = None
    dataset_id = None
    dataset_split = None

    def setUp(self):
        print("========== RUNNING ==============")
        print(self.scheme)

        self.save_dir = None
        self.oneshot_kwargs = {}
        self.num_calibration_samples = 512
        self.max_seq_length = 2048
        self.prompts = [
            "The capital of France is",
            "The president of the US is",
            "My name is",
        ]

    def test_vllm(self):
        # Load model.
        loaded_model = SparseAutoModelForCausalLM.from_pretrained(
            self.model, device_map="auto", torch_dtype="auto"
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
        self.oneshot_kwargs["recipe"] = QuantizationModifier(
            targets="Linear", scheme=self.scheme, ignore=["lm_head"]
        )

        # Apply quantization.
        print("ONESHOT KWARGS", self.oneshot_kwargs)
        oneshot(
            **self.oneshot_kwargs, output_dir=self.save_dir, clear_sparse_session=True
        )

        # Confirm generations of the quantized model look sane.
        print("========== SAMPLE GENERATION ==============")
        input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(
            "cuda"
        )
        output = loaded_model.generate(input_ids, max_new_tokens=20)
        print(tokenizer.decode(output[0]))
        print("==========================================")

        # Run vLLM with saved model
        print("================= RUNNING vLLM =========================")
        sampling_params = SamplingParams(temperature=0.80, top_p=0.95)
        llm = LLM(model=self.save_dir)
        outputs = llm.generate(self.prompts, sampling_params)
        print("========== vLLM GENERATION ==============")
        print(outputs)
        assert outputs

    def tearDown(self):
        shutil.rmtree(self.save_dir)
