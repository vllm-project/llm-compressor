import math
from collections import namedtuple

import pytest
import torch
from compressed_tensors.utils import tensor_follows_mask_structure

from llmcompressor import oneshot
from llmcompressor.core import reset_session
from llmcompressor.pytorch.utils.helpers import tensor_sparsity
from llmcompressor.utils.pytorch import qat_active
from tests.testing_utils import parse_params

MASK_STRUCTURE_CONFIGS_DIRECTORY = (
    "tests/llmcompressor/transformers/sparsegpt/sparsegpt_configs/mask_structure"
)

_TestArgs = namedtuple(
    "_TestArgs",
    [
        "model",
        "initial_pruning_only_recipe",
        "initial_sparsity",
        "recipe_mask_structure",
        "dataset",
        "subsequent_prune_and_quant_recipe",
        "final_sparsity",
    ],
)


@pytest.fixture(params=parse_params(MASK_STRUCTURE_CONFIGS_DIRECTORY))
def args_to_test(request):
    config = request.param
    # config:
    # {model, initial_pruning_only_recipe, initial_sparsity, recipe_mask_structure,
    #  dataset, subsequent_prune_and_quant_recipe, final_sparsity}
    return _TestArgs(
        config.get("model"),
        config.get("initial_pruning_only_recipe"),
        config.get("initial_sparsity"),
        config.get("recipe_mask_structure"),
        config.get("dataset"),
        config.get("subsequent_prune_and_quant_recipe"),
        config.get("final_sparsity"),
    )


@pytest.mark.integration
def test_mask_structure_preserved(args_to_test, tmp_path):
    tolerance = 1e-3
    num_calibration_samples = 16

    output_first = tmp_path / "test_1"
    output_second = tmp_path / "test_2"

    first_tiny_model = oneshot(
        model=args_to_test.model,
        dataset=args_to_test.dataset,
        num_calibration_samples=num_calibration_samples,
        recipe=args_to_test.initial_pruning_only_recipe,
        output_dir=output_first,
        save_compressed=False,
    )
    targetted_layer = first_tiny_model.model.layers[0].self_attn.k_proj
    target_layer_sparsity = tensor_sparsity(targetted_layer.weight)
    initial_mask = first_tiny_model.model.layers[0].self_attn.k_proj.weight == 0

    # sparsity is as expected, i.e close to self.initial_sparsity
    assert math.isclose(
        target_layer_sparsity.item(), args_to_test.initial_sparsity, rel_tol=tolerance
    )
    # mask structure is as expected, i.e same as self.recipe_mask_structure
    assert tensor_follows_mask_structure(
        initial_mask, args_to_test.recipe_mask_structure
    )

    reset_session()

    second_tiny_model = oneshot(
        model=output_first,
        dataset=args_to_test.dataset,
        num_calibration_samples=num_calibration_samples,
        recipe=args_to_test.subsequent_prune_and_quant_recipe,
        output_dir=output_second,
        save_compressed=False,
    )

    # model is loaded
    assert second_tiny_model is not None

    targetted_layer = second_tiny_model.model.layers[0].self_attn.k_proj
    target_layer_sparsity = tensor_sparsity(targetted_layer.weight)

    # sparsity is as expected, i.e close to self.final_sparsity
    assert math.isclose(
        target_layer_sparsity.item(), args_to_test.final_sparsity, rel_tol=tolerance
    )
    # qat should be active, second recipe has quantization
    assert qat_active(second_tiny_model)

    # original mask structure is preserved, additional zeros are
    # added on top of the initial mask
    final_mask = targetted_layer.weight == 0
    assert torch.all(initial_mask <= final_mask)
