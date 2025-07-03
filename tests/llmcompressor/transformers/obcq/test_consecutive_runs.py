import os
import shutil
import unittest
from pathlib import Path

import pytest
import yaml
from parameterized import parameterized_class
from transformers import AutoModelForCausalLM
from transformers.utils.quantization_config import CompressedTensorsConfig

from llmcompressor.recipe import Recipe
from llmcompressor.transformers.utils import is_model_ct_quantized_from_path
from tests.testing_utils import parse_params, requires_gpu

CONFIGS_DIRECTORY = "tests/llmcompressor/transformers/obcq/obcq_configs/consec_runs"
GPU_CONFIGS_DIRECTORY = (
    "tests/llmcompressor/transformers/obcq/obcq_configs/consec_runs/gpu"
)


class TestConsecutiveRuns(unittest.TestCase):
    quantization_config = CompressedTensorsConfig(run_compressed=False)

    def _test_consecutive_runs(
        self, tolerance: float, num_calibration_samples: int = 16
    ):
        import math

        from llmcompressor import oneshot
        from llmcompressor.core import active_session
        from llmcompressor.pytorch.utils.helpers import tensor_sparsity
        from llmcompressor.utils.pytorch import qat_active

        # test recipe with 50% sparsity, quantization and smoothquant
        oneshot(
            model=self.model,
            dataset=self.dataset,
            num_calibration_samples=num_calibration_samples,
            recipe=self.first_recipe,
            output_dir=self.output_first,
        )

        first_model = AutoModelForCausalLM.from_pretrained(
            self.output_first,
            torch_dtype="auto",
            quantization_config=self.quantization_config,
        )

        layer_0_sparse = tensor_sparsity(
            first_model.model.layers[0].self_attn.k_proj.weight
        )
        assert math.isclose(layer_0_sparse.item(), 0.5, rel_tol=tolerance)
        assert qat_active(first_model)

        session = active_session()
        session.reset()

        # reload saved model and increase sparsity to 0.7
        oneshot(
            model=self.output_first,
            dataset=self.dataset,
            num_calibration_samples=num_calibration_samples,
            recipe=self.second_recipe,
            output_dir=self.output_second,
        )

        second_model = AutoModelForCausalLM.from_pretrained(
            self.output_second,
            quantization_config=self.quantization_config,
            torch_dtype="auto",
        )

        layer_0_sparse = tensor_sparsity(
            second_model.model.layers[0].self_attn.k_proj.weight
        )
        assert math.isclose(layer_0_sparse.item(), 0.7, rel_tol=tolerance)
        assert qat_active(second_model)

        recipe_path = self.output_second / "recipe.yaml"
        recipe_data = yaml.safe_load(recipe_path.read_text())
        stage_keys = recipe_data.keys()
        self.assertEqual(len(stage_keys), 2)
        self.assertIn("test_stage_0", stage_keys)
        self.assertIn("test_stage_1", stage_keys)

        # check saved modifier names are same
        stage0_modifier_names = list(
            list(recipe_data["test_stage_0"].values())[0].keys()
        )
        exp_stage0_modifier_names = [
            mod.__class__.__name__
            for mod in Recipe.create_instance(self.first_recipe).modifiers
        ]
        stage1_modifier_names = list(
            list(recipe_data["test_stage_1"].values())[0].keys()
        )
        exp_stage1_modifier_names = [
            mod.__class__.__name__
            for mod in Recipe.create_instance(self.second_recipe).modifiers
        ]
        self.assertEqual(stage0_modifier_names, exp_stage0_modifier_names)
        self.assertEqual(stage1_modifier_names, exp_stage1_modifier_names)

    def tearDown(self):
        if os.path.isdir(self.output):
            shutil.rmtree(self.output)


@pytest.mark.integration
@parameterized_class(parse_params(CONFIGS_DIRECTORY))
class TestConsecutiveRunsSmall(TestConsecutiveRuns):
    model = None
    first_recipe = None
    second_recipe = None
    dataset = None

    def setUp(self):
        import torch

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.output = "./oneshot_output"
        self.output_first = Path(self.output) / "test_1"
        self.output_second = Path(self.output) / "test_2"

    def test_consecutive_runs_small(self):
        self._test_consecutive_runs(tolerance=1e-3)


@requires_gpu
@pytest.mark.integration
@parameterized_class(parse_params(GPU_CONFIGS_DIRECTORY))
class TestConsecutiveRunsGPU(TestConsecutiveRuns):
    # Will be populated using the config files
    model = None
    first_recipe = None
    second_recipe = None
    dataset = None
    device = None

    def setUp(self):
        from transformers import AutoModelForCausalLM

        self.assertFalse(
            is_model_ct_quantized_from_path(self.model),
            "The provided model is quantized. Please use a dense model.",
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model, device_map=self.device, torch_dtype="auto"
        )

        self.output = "./oneshot_output"
        self.output_first = Path(self.output) / "test_1"
        self.output_second = Path(self.output) / "test_2"

    def test_consecutive_runs_gpu(self):
        self._test_consecutive_runs(tolerance=1e-0)
