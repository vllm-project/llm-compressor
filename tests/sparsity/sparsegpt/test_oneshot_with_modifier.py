import pytest

from llmcompressor import oneshot
from llmcompressor.modifiers.pruning.sparsegpt.base import SparseGPTModifier
from tests.testing_utils import parse_params, requires_gpu

CONFIGS_DIRECTORY = (
    "tests/llmcompressor/transformers/sparsegpt/sparsegpt_configs/sparsity_generic"
)


@requires_gpu
@pytest.mark.integration
@pytest.mark.parametrize("config", parse_params(CONFIGS_DIRECTORY))
def test_oneshot_with_modifier_object(tmp_path, config):
    output_dir = tmp_path / "oneshot_out"
    recipe_str = [SparseGPTModifier(sparsity=0.5, targets=[r"re:model.layers.\d+$"])]

    concatenate_data = False
    num_calibration_samples = 64
    splits = {"calibration": "train[:10%]"}

    oneshot(
        model=config["model"],
        dataset=config["dataset"],
        output_dir=output_dir,
        num_calibration_samples=num_calibration_samples,
        recipe=recipe_str,
        concatenate_data=concatenate_data,
        splits=splits,
    )
