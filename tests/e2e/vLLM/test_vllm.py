import shutil
import unittest

import pytest
from datasets import load_dataset
from parameterized import parameterized_class
from transformers import AutoTokenizer

from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.transformers import SparseAutoModelForCausalLM, oneshot
from tests.testing_utils import (
    parse_params,
    preprocess_tokenize_dataset,
    requires_gpu,
    requires_torch,
)

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
ACTORDER = "tests/e2e/vLLM/configs/actorder"
CONFIGS = [WNA16, FP8, INT8, ACTORDER]


@requires_gpu
@requires_torch
@pytest.mark.skipif(not vllm_installed, reason="vLLM is not installed, skipping test")
@parameterized_class(parse_params(CONFIGS))
class TestvLLM(unittest.TestCase):
    """
    The following test quantizes a model using a preset scheme or recipe,
    runs the model using vLLM, and then pushes the model to the hub for
    future use. Each test case is focused on a specific quantization type
    (e.g W4A16 with grouped quantization, W4N16 with channel quantization).
    To add a new test case, a new config has to be added to one of the folders
    listed in the `CONFIGS` folder. If the test case is for a data type not listed
    in `CONFIGS`, a new folder can be created and added to the list. The tests
    run on a cadence defined by the `cadence` field. Each config defines the model
    to quantize. Optionally, a dataset id and split can be provided for calibration.
    Finally, all config files must list a scheme. The scheme can be a preset scheme
    from https://github.com/neuralmagic/compressed-tensors/blob/main/src/compressed_tensors/quantization/quant_scheme.py
    or another identifier which can be used for the particular test case. If a recipe
    is not provided, it is assumed that the scheme provided is a preset scheme and will
    be used for quantization. Otherwise, the recipe will always be used if given.
    """  # noqa: E501

    model = None
    scheme = None
    dataset_id = None
    dataset_config = None
    dataset_split = None
    recipe = None
    save_dir = None

    def setUp(self):
        print("========== RUNNING ==============")
        print(self.scheme)

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

        if self.dataset_id:
            ds = load_dataset(
                self.dataset_id, name=self.dataset_config, split=self.dataset_split
            )
            ds = ds.shuffle(seed=42).select(range(self.num_calibration_samples))
            ds = preprocess_tokenize_dataset(ds, tokenizer, self.max_seq_length)
            self.oneshot_kwargs["dataset"] = ds
            self.oneshot_kwargs["max_seq_length"] = self.max_seq_length
            self.oneshot_kwargs["num_calibration_samples"] = (
                self.num_calibration_samples
            )

        if self.save_dir is None:
            self.save_dir = self.model.split("/")[1] + f"-{self.scheme}"

        self.oneshot_kwargs["model"] = loaded_model
        if self.recipe:
            self.oneshot_kwargs["recipe"] = self.recipe
        else:
            # Test assumes that if a recipe was not provided, using
            # a compatible preset sceme
            self.oneshot_kwargs["recipe"] = QuantizationModifier(
                targets="Linear", scheme=self.scheme, ignore=["lm_head"]
            )

        # Apply quantization.
        print("ONESHOT KWARGS", self.oneshot_kwargs)
        oneshot(
            **self.oneshot_kwargs,
            clear_sparse_session=True,
            oneshot_device=self.device,
        )
        self.oneshot_kwargs["model"].save_pretrained(self.save_dir)
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

        print("================= UPLOADING TO HUB ======================")
        self.oneshot_kwargs["model"].push_to_hub(f"nm-testing/{self.save_dir}-e2e")
        tokenizer.push_to_hub(f"nm-testing/{self.save_dir}-e2e")

    def tearDown(self):
        if self.save_dir is not None:
            shutil.rmtree(self.save_dir)
