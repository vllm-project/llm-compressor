import shutil
import unittest
from typing import Callable

import pytest
from parameterized import parameterized, parameterized_class

from tests.e2e.e2e_utils import run_oneshot_for_e2e_testing
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
ACTORDER = "tests/e2e/vLLM/configs/actorder"
WNA16_2of4 = "tests/e2e/vLLM/configs/WNA16_2of4"
CONFIGS = [WNA16, FP8, INT8, ACTORDER, WNA16_2of4]


def gen_test_name(testcase_func: Callable, param_num: int, param: dict) -> str:
    return "_".join(
        [
            testcase_func.__name__,
            parameterized.to_safe_name(
                param.get("testconfig_path", "").split("configs/")[-1]
            ),
            param.get("cadence", "").lower(),
        ]
    )


@requires_gpu
@requires_torch
@pytest.mark.skipif(not vllm_installed, reason="vLLM is not installed, skipping test")
@parameterized_class(parse_params(CONFIGS), class_name_func=gen_test_name)
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
        self.max_seq_length = 2048
        self.prompts = [
            "The capital of France is",
            "The president of the US is",
            "My name is",
        ]

    def test_vllm(self):
        # Run vLLM with saved model
        import torch

        save_dir = run_oneshot_for_e2e_testing(
            model=self.model,
            device=self.device,
            oneshot_kwargs=self.oneshot_kwargs,
            num_calibration_samples=self.num_calibration_samples,
            max_seq_length=self.max_seq_length,
            scheme=self.scheme,
            dataset_id=self.dataset_id,
            dataset_config=self.dataset_config,
            dataset_split=self.dataset_split,
            recipe=self.recipe,
        )

        print("================= RUNNING vLLM =========================")
        sampling_params = SamplingParams(temperature=0.80, top_p=0.95)
        if "W4A16_2of4" in self.scheme:
            # required by the kernel
            llm = LLM(model=save_dir, dtype=torch.float16)
        else:
            llm = LLM(model=save_dir)
        outputs = llm.generate(self.prompts, sampling_params)
        print("================= vLLM GENERATION ======================")
        for output in outputs:
            assert output
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print("PROMPT", prompt)
            print("GENERATED TEXT", generated_text)
        self.save_dir = save_dir

    def tearDown(self):
        if self.save_dir is not None:
            shutil.rmtree(self.save_dir)
