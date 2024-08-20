import shutil
import unittest
from pathlib import Path
import os

import pytest
import yaml
from parameterized import parameterized_class

from tests.testing_utils import parse_params, requires_gpu, requires_torch

CONFIGS_DIRECTORY = "tests/llmcompressor/transformers/obcq/obcq_configs/consec_runs"
GPU_CONFIGS_DIRECTORY = (
    "tests/llmcompressor/transformers/obcq/obcq_configs/consec_runs/gpu"
)


class TestConsecutiveRuns(unittest.TestCase):
    def _test_consecutive_runs(
        self, tolerance: float, num_calibration_samples: int = 16
    ):
        import math

        from llmcompressor.core import active_session
        from llmcompressor.pytorch.model_load.helpers import get_session_model
        from llmcompressor.pytorch.utils.helpers import tensor_sparsity
        from llmcompressor.transformers import oneshot
        from llmcompressor.utils.pytorch import qat_active

        # test recipe with 50% sparsity, quantization and smoothquant
        oneshot(
            model=self.model,
            dataset=self.dataset,
            num_calibration_samples=num_calibration_samples,
            recipe=self.first_recipe,
            output_dir=self.output_first,
            oneshot_device=self.device,
            clear_sparse_session=False,
        )
        first_tiny_model = get_session_model()
        layer_0_sparse = tensor_sparsity(
            first_tiny_model.model.layers[0].self_attn.k_proj.weight
        )
        assert math.isclose(layer_0_sparse.item(), 0.5, rel_tol=tolerance)
        assert qat_active(first_tiny_model)

        session = active_session()
        session_recipe = session.lifecycle.recipe_container.compiled_recipe
        stages = [stage.group for stage in session_recipe.stages]
        self.assertEqual(len(stages), 1)
        session.reset()

        # reload saved model and up sparsity to 0.7
        oneshot(
            model=self.output_first,
            dataset=self.dataset,
            num_calibration_samples=num_calibration_samples,
            recipe=self.second_recipe,
            output_dir=self.output_second,
            oneshot_device="cuda:1"
            clear_sparse_session=False,
        )

        second_tiny_model = get_session_model()
        layer_0_sparse = tensor_sparsity(
            second_tiny_model.model.layers[0].self_attn.k_proj.weight
        )
        assert math.isclose(layer_0_sparse.item(), 0.7, rel_tol=tolerance)
        assert qat_active(second_tiny_model)

        session = active_session()
        session_recipe = session.lifecycle.recipe_container.compiled_recipe
        stages = [stage.group for stage in session_recipe.stages]
        self.assertEqual(len(stages), 2)

        recipe_path = self.output_second / "recipe.yaml"
        recipe_data = yaml.safe_load(recipe_path.read_text())
        stage_keys = recipe_data.keys()
        self.assertEqual(len(stage_keys), 2)
        self.assertIn("test_stage_0", stage_keys)
        self.assertIn("test_stage_1", stage_keys)

    def tearDown(self):
        shutil.rmtree(self.output)


@requires_torch
@pytest.mark.integration
@parameterized_class(parse_params(CONFIGS_DIRECTORY))
class TestConsecutiveRunsSmall(TestConsecutiveRuns):
    model = None
    first_recipe = None
    second_recipe = None
    dataset = None

    def setUp(self):
        import torch
        cwd = os.getcwd()

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.output = f"{cwd}/oneshot_output"
        self.output_first = Path(self.output) / "test_1"
        self.output_second = Path(self.output) / "test_2"

    def test_consecutive_runs_small(self):
        self._test_consecutive_runs(tolerance=1e-3)


# TODO: @Satrat and @dsikka, revisit if we want these nightly or weekly
@requires_gpu
@requires_torch
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
        from llmcompressor.transformers import SparseAutoModelForCausalLM

        self.model = SparseAutoModelForCausalLM.from_pretrained(
            self.model, device_map=self.device
        )
        cwd = os.getcwd()
        self.output = f"{cwd}/oneshot_output"
        self.output_first = Path(self.output) / "test_1"
        self.output_second = Path(self.output) / "test_2"

    def test_consecutive_runs_gpu(self):
        self._test_consecutive_runs(tolerance=1e-0, num_calibration_samples=16)
