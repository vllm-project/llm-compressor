import os
import re
import shutil
from pathlib import Path
from typing import Callable

import pytest
import yaml
from huggingface_hub import HfApi
from loguru import logger

from llmcompressor.core import active_session
from tests.e2e.e2e_utils import run_oneshot_for_e2e_testing
from tests.examples.utils import requires_gpu_count

try:
    from vllm import LLM, SamplingParams

    vllm_installed = True
except ImportError:
    vllm_installed = False
    logger.warning("vllm is not installed. This test will be skipped")

HF_MODEL_HUB_NAME = "nm-testing"

TEST_DATA_FILE = os.environ.get("TEST_DATA_FILE", "")

EXPECTED_SAVED_FILES = [
    "config.json",
    r"^model(?:-\d{5}-of-\d{5})?\.safetensors$",
    "recipe.yaml",
    "tokenizer.json",
]


@pytest.fixture
def record_config_file(record_testsuite_property: Callable[[str, object], None]):
    test_data_file_name = TEST_DATA_FILE.split("configs/")[-1]
    record_testsuite_property("TEST_DATA_FILE_NAME", test_data_file_name)


# Will run each test case in its own process through run_tests.sh
# emulating vLLM CI testing
@requires_gpu_count(1)
@pytest.mark.skipif(not vllm_installed, reason="vLLM is not installed, skipping test")
class TestvLLM:
    """
    The following test quantizes a model using a preset scheme or recipe,
    runs the model using vLLM, and then pushes the model to the hub for
    future use. Each test case is focused on a specific quantization type
    (e.g W4A16 with grouped quantization, W4N16 with channel quantization).
    To add a new test case, a new config has to be added to the `configs` folder.
    The tests run on a cadence defined by the `cadence` field. Each config defines
    the model to quantize. Optionally, a dataset id and split can be provided for
    calibration. Finally, all config files must list a scheme. The scheme can be a
    preset scheme from
    https://github.com/neuralmagic/compressed-tensors/blob/main/src/compressed_tensors/quantization/quant_scheme.py
    or another identifier which can be used for the particular test case. If a recipe
    is not provided, it is assumed that the scheme provided is a preset scheme and will
    be used for quantization. Otherwise, the recipe will always be used if given.
    """  # noqa: E501

    def set_up(self):
        eval_config = yaml.safe_load(Path(TEST_DATA_FILE).read_text(encoding="utf-8"))

        if os.environ.get("CADENCE", "commit") != eval_config.get("cadence"):
            pytest.skip("Skipping test; cadence mismatch")

        self.model = eval_config["model"]
        self.scheme = eval_config.get("scheme")
        self.dataset_id = eval_config.get("dataset_id")
        self.dataset_config = eval_config.get("dataset_config")
        self.dataset_split = eval_config.get("dataset_split")
        self.recipe = eval_config.get("recipe")
        self.quant_type = eval_config.get("quant_type")
        self.save_dir = eval_config.get("save_dir")

        logger.info("========== RUNNING ==============")
        logger.info(self.scheme)

        self.device = "cuda:0"
        self.num_calibration_samples = 256
        self.max_seq_length = 2048
        self.prompts = [
            "The capital of France is",
            "The president of the US is",
            "My name is",
        ]
        self.api = HfApi()

    @pytest.mark.usefixtures("record_config_file")
    def test_vllm(self):
        # Run vLLM with saved model
        import torch

        self.set_up()
        if not self.save_dir:
            self.save_dir = self.model.split("/")[1] + f"-{self.scheme}"
        oneshot_model, tokenizer = run_oneshot_for_e2e_testing(
            model=self.model,
            device=self.device,
            num_calibration_samples=self.num_calibration_samples,
            max_seq_length=self.max_seq_length,
            scheme=self.scheme,
            dataset_id=self.dataset_id,
            dataset_config=self.dataset_config,
            dataset_split=self.dataset_split,
            recipe=self.recipe,
            quant_type=self.quant_type,
        )

        # check that session contains recipe
        self._check_session_contains_recipe()

        logger.info("================= SAVING TO DISK ======================")
        oneshot_model.save_pretrained(self.save_dir)
        tokenizer.save_pretrained(self.save_dir)
        recipe_path = os.path.join(self.save_dir, "recipe.yaml")

        # check that expected files exist
        self._check_save_dir_has_expected_files()

        # Use the session to fetch the recipe;
        # Reset session for next test case
        session = active_session()
        recipe_yaml_str = session.get_serialized_recipe()
        with open(recipe_path, "w") as fp:
            fp.write(recipe_yaml_str)
        session.reset()

        logger.info("================= UPLOADING TO HUB ======================")

        stub = f"{HF_MODEL_HUB_NAME}/{self.save_dir}-e2e"

        self.api.create_repo(
            repo_id=stub,
            exist_ok=True,
            repo_type="model",
            private=False,
        )

        self.api.upload_folder(
            repo_id=stub,
            folder_path=self.save_dir,
        )

        logger.info("================= RUNNING vLLM =========================")

        sampling_params = SamplingParams(temperature=0.80, top_p=0.95)
        if "W4A16_2of4" in self.scheme:
            # required by the kernel
            llm = LLM(model=self.save_dir, dtype=torch.float16)
        else:
            llm = LLM(model=self.save_dir)
        outputs = llm.generate(self.prompts, sampling_params)

        logger.info("================= vLLM GENERATION ======================")
        for output in outputs:
            assert output
            prompt = output.prompt
            generated_text = output.outputs[0].text

            logger.info("PROMPT")
            logger.info(prompt)
            logger.info("GENERATED TEXT")
            logger.info(generated_text)

        self.tear_down()

    def tear_down(self):
        import torch
        torch.cuda.empty_cache()

        if self.save_dir is not None:
            shutil.rmtree(self.save_dir)

    def _check_session_contains_recipe(self) -> None:
        session = active_session()
        recipe_yaml_str = session.get_serialized_recipe()
        assert recipe_yaml_str is not None

    def _check_save_dir_has_expected_files(self):
        files = os.listdir(self.save_dir)
        logger.debug("Saved files: ", files)

        matched_patterns = set()

        for expected in EXPECTED_SAVED_FILES:
            # Find all files matching the expected pattern
            matches = [
                file
                for file in files
                if (
                    re.fullmatch(expected, file)
                    if expected.startswith("^")
                    else file == expected
                )
            ]
            if len(matches) > 0:
                matched_patterns.add(expected)

        assert len(matched_patterns) == len(EXPECTED_SAVED_FILES), (
            "expected: ",
            EXPECTED_SAVED_FILES,
            "\n saved: ",
            list(matched_patterns),
        )
