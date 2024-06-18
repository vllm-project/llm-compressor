# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import shutil
import unittest

import pytest

from parameterized import parameterized_class
from tests.testing_utils import parse_params, requires_gpu, requires_torch


CONFIGS_DIRECTORY = "tests/sparseml/transformers/obcq/obcq_configs/sparse"
GPU_CONFIGS_DIRECTORY = "tests/sparseml/transformers/obcq/obcq_configs/sparse/gpu"


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
        from sparseml.pytorch.model_load.helpers import get_session_model
        from sparseml.pytorch.utils.helpers import tensor_sparsity
        from sparseml.transformers import oneshot

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

        from sparseml.transformers import SparseAutoModelForCausalLM

        self.output = "./oneshot_output"

        if "zoo:" in self.model:
            self.model = SparseAutoModelForCausalLM.from_pretrained(
                self.model, device_map=self.device, torch_dtype=torch.bfloat16
            )

    def test_sparsities_gpu(self):
        from sparseml.pytorch.model_load.helpers import get_session_model
        from sparseml.pytorch.utils.helpers import tensor_sparsity
        from sparseml.transformers import oneshot

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
