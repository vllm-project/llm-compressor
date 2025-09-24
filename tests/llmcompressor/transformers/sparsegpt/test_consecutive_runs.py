import math

import pytest
import yaml
from transformers import AutoModelForCausalLM
from transformers.utils.quantization_config import CompressedTensorsConfig

from llmcompressor import oneshot
from llmcompressor.core import active_session
from llmcompressor.pytorch.utils.helpers import tensor_sparsity
from llmcompressor.recipe import Recipe
from llmcompressor.transformers.utils import is_model_ct_quantized_from_path
from llmcompressor.utils.pytorch import qat_active
from tests.testing_utils import parse_params, requires_gpu

CONFIGS_DIRECTORY = (
    "tests/llmcompressor/transformers/sparsegpt/sparsegpt_configs/consec_runs"
)
GPU_CONFIGS_DIRECTORY = (
    "tests/llmcompressor/transformers/sparsegpt/sparsegpt_configs/consec_runs/gpu"
)


@pytest.mark.integration
@pytest.mark.parametrize("config", parse_params(CONFIGS_DIRECTORY))
def test_consecutive_runs_small(config, tmp_path):
    _test_consecutive_runs(
        config["model"],
        config["dataset"],
        config["first_recipe"],
        config["second_recipe"],
        1e-3,
        tmp_path,
    )


@requires_gpu
@pytest.mark.integration
@pytest.mark.parametrize("config", parse_params(GPU_CONFIGS_DIRECTORY))
def test_consecutive_runs_gpu(config, tmp_path):
    assert not is_model_ct_quantized_from_path(
        config["model"]
    ), "The provided model is quantized. Please use a dense model."
    model = AutoModelForCausalLM.from_pretrained(
        config["model"], device_map=config["device"], torch_dtype="auto"
    )

    _test_consecutive_runs(
        model,
        config["dataset"],
        config["first_recipe"],
        config["second_recipe"],
        1e-0,
        tmp_path,
    )


def _test_consecutive_runs(
    model, dataset, first_recipe, second_recipe, tolerance, tmp_path
):
    output_first = tmp_path / "test_1"
    output_second = tmp_path / "test_2"
    num_calibration_samples = 16
    quantization_config = CompressedTensorsConfig(run_compressed=False)

    # test recipe with 50% sparsity, quantization and smoothquant
    oneshot(
        model=model,
        dataset=dataset,
        num_calibration_samples=num_calibration_samples,
        recipe=first_recipe,
        output_dir=output_first,
    )

    first_model = AutoModelForCausalLM.from_pretrained(
        output_first,
        torch_dtype="auto",
        quantization_config=quantization_config,
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
        model=output_first,
        dataset=dataset,
        num_calibration_samples=num_calibration_samples,
        recipe=second_recipe,
        output_dir=output_second,
    )

    second_model = AutoModelForCausalLM.from_pretrained(
        output_second,
        quantization_config=quantization_config,
        torch_dtype="auto",
    )

    layer_0_sparse = tensor_sparsity(
        second_model.model.layers[0].self_attn.k_proj.weight
    )
    assert math.isclose(layer_0_sparse.item(), 0.7, rel_tol=tolerance)
    assert qat_active(second_model)

    recipe_path = output_second / "recipe.yaml"
    recipe_data = yaml.safe_load(recipe_path.read_text())
    stage_keys = recipe_data.keys()
    assert len(stage_keys) == 2
    assert "test_stage_0" in stage_keys
    assert "test_stage_1" in stage_keys

    # check saved modifier names are same
    stage0_modifier_names = list(list(recipe_data["test_stage_0"].values())[0].keys())
    exp_stage0_modifier_names = [
        mod.__class__.__name__ for mod in Recipe.create_instance(first_recipe).modifiers
    ]
    stage1_modifier_names = list(list(recipe_data["test_stage_1"].values())[0].keys())
    exp_stage1_modifier_names = [
        mod.__class__.__name__
        for mod in Recipe.create_instance(second_recipe).modifiers
    ]
    assert stage0_modifier_names == exp_stage0_modifier_names
    assert stage1_modifier_names == exp_stage1_modifier_names
