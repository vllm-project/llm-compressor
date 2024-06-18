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


import unittest
from pathlib import Path

import pytest

import sparseml
from compressed_tensors.compressors.utils import tensor_follows_mask_structure
from parameterized import parameterized_class
from tests.testing_utils import parse_params, requires_torch


MASK_STRUCTURE_CONFIGS_DIRECTORY = (
    "tests/sparseml/transformers/obcq/obcq_configs/consec_runs/mask_structure"
)


@requires_torch
@pytest.mark.integration
@parameterized_class(parse_params(MASK_STRUCTURE_CONFIGS_DIRECTORY))
class TestMaskStructurePreserved(unittest.TestCase):
    """
    Tests that the mask structure is preserved across multiple runs of oneshot
    initial model is pruned using a mask_structure, and then the pruned model
    is further pruned and quantized.
    """

    model = None
    initial_pruning_only_recipe = None
    initial_sparsity = None
    recipe_mask_structure = None
    dataset = None
    subsequent_prune_and_quant_recipe = None
    final_sparsity = None

    def setUp(self) -> None:
        import torch

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.output = "./oneshot_output"
        self.output_first = Path(self.output) / "test_1"
        self.output_second = Path(self.output) / "test_2"

    def test_mask_structure_preserved(self):
        """
        Checks that the mask structure is preserved across runs of oneshot
        between the initial pruning and the subsequent pruning + quantization
        """
        import math

        import torch

        from sparseml.pytorch.model_load.helpers import get_session_model
        from sparseml.pytorch.utils.helpers import tensor_sparsity
        from sparseml.transformers import oneshot
        from sparseml.utils.pytorch import qat_active

        tolerance = 1e-3
        num_calibration_samples = 16

        oneshot(
            model=self.model,
            dataset=self.dataset,
            num_calibration_samples=num_calibration_samples,
            recipe=self.initial_pruning_only_recipe,
            output_dir=self.output_first,
            oneshot_device=self.device,
            clear_sparse_session=False,
        )
        first_tiny_model = get_session_model()
        targetted_layer = first_tiny_model.model.layers[0].self_attn.k_proj
        target_layer_sparsity = tensor_sparsity(targetted_layer.weight)
        initial_mask = first_tiny_model.model.layers[0].self_attn.k_proj.weight == 0

        # sparsity is as expected, i.e close to self.initial_sparsity
        assert math.isclose(
            target_layer_sparsity.item(), self.initial_sparsity, rel_tol=tolerance
        )
        # mask structure is as expected, i.e same as self.recipe_mask_structure
        assert tensor_follows_mask_structure(initial_mask, self.recipe_mask_structure)

        sparseml.reset_session()

        oneshot(
            model=self.output_first,
            dataset=self.dataset,
            num_calibration_samples=num_calibration_samples,
            recipe=self.subsequent_prune_and_quant_recipe,
            output_dir=self.output_second,
            oneshot_device=self.device,
            clear_sparse_session=False,
        )

        second_tiny_model = get_session_model()

        # model is loaded
        assert second_tiny_model is not None

        targetted_layer = second_tiny_model.model.layers[0].self_attn.k_proj
        target_layer_sparsity = tensor_sparsity(targetted_layer.weight)

        # sparsity is as expected, i.e close to self.final_sparsity
        assert math.isclose(
            target_layer_sparsity.item(), self.final_sparsity, rel_tol=tolerance
        )
        # qat should be active, second recipe has quantization
        assert qat_active(second_tiny_model)

        # original mask structure is preserved, additional zeros are
        # added on top of the initial mask
        final_mask = targetted_layer.weight == 0
        assert torch.all(initial_mask <= final_mask)
