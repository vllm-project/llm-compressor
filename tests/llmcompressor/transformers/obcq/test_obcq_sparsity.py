import math
import shutil
import unittest

import pytest
from parameterized import parameterized_class

from tests.testing_utils import parse_params, requires_gpu, requires_torch

CONFIGS_DIRECTORY = "tests/llmcompressor/transformers/obcq/obcq_configs/sparse"
GPU_CONFIGS_DIRECTORY = "tests/llmcompressor/transformers/obcq/obcq_configs/sparse/gpu"


@requires_torch
@pytest.mark.integration
@parameterized_class(parse_params(CONFIGS_DIRECTORY))
class TestSparsities(unittest.TestCase):
    model = None
    dataset = None
    recipe = None
    sparsity = None

    def setUp(self):
        import torch

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.output = "./oneshot_output"

    def test_sparsities(self):
        from llmcompressor.pytorch.model_load.helpers import get_session_model
        from llmcompressor.pytorch.utils.helpers import tensor_sparsity
        from llmcompressor.transformers import oneshot

        oneshot(
            model=self.model,
            dataset=self.dataset,
            oneshot_device=self.device,
            recipe=self.recipe,
            max_seq_length=128,
            num_calibration_samples=64,
            pad_to_max_length=False,
            clear_sparse_session=False,
            output_dir=self.output,
        )

        model = get_session_model()

        layer_1_sparse = tensor_sparsity(model.model.layers[1].self_attn.k_proj.weight)
        assert math.isclose(layer_1_sparse.item(), self.sparsity, rel_tol=1e-4)
        layer_2_dense = tensor_sparsity(model.model.layers[2].self_attn.k_proj.weight)
        assert math.isclose(layer_2_dense.item(), 0.0, rel_tol=1e-4)

    def tearDown(self):
        import torch

        shutil.rmtree(self.output)
        torch.cuda.empty_cache()


# TODO: @Satrat and @dsikka, revisit if we want these nightly or weekly
@requires_gpu
@requires_torch
@pytest.mark.integration
@parameterized_class(parse_params(GPU_CONFIGS_DIRECTORY))
class TestSparsitiesGPU(unittest.TestCase):
    model = None
    dataset = None
    recipe = None
    sparsity = None
    device = None

    def setUp(self):
        import torch
        from transformers import AutoModelForCausalLM

        self.output = "./oneshot_output"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model, device_map=self.device, torch_dtype=torch.bfloat16
        )

    def test_sparsities_gpu(self):
        from llmcompressor.pytorch.model_load.helpers import get_session_model
        from llmcompressor.pytorch.utils.helpers import tensor_sparsity
        from llmcompressor.transformers import oneshot

        oneshot(
            model=self.model,
            dataset=self.dataset,
            oneshot_device=self.device,
            recipe=self.recipe,
            max_seq_length=128,
            num_calibration_samples=64,
            pad_to_max_length=False,
            clear_sparse_session=False,
            output_dir=self.output,
            precision="bfloat16",
            bf16=True,
        )

        model = get_session_model()

        layer_1_sparse = tensor_sparsity(model.model.layers[1].self_attn.k_proj.weight)
        assert math.isclose(layer_1_sparse.item(), self.sparsity, rel_tol=1e-4)
        layer_2_dense = tensor_sparsity(model.model.layers[2].self_attn.k_proj.weight)
        assert math.isclose(layer_2_dense.item(), 0.0, abs_tol=1e-4)

    def tearDown(self):
        import torch

        shutil.rmtree(self.output)
        torch.cuda.empty_cache()
