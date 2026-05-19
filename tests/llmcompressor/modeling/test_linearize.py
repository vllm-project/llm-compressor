import os
from pathlib import Path

import pytest
import torch
from compressed_tensors.utils import patch_attr
from safetensors import safe_open
from transformers import AutoModelForCausalLM
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4PreTrainedModel,
)

from llmcompressor.modeling.moe.context import (
    moe_calibration_context,
)
from llmcompressor.modeling.moe.linearize import (
    load_quantizable_moe,
)
from tests.testing_utils import requires_gpu


@pytest.fixture
def patch_deepseek_fp32_modules():
    """
    Monkey patch to force DeepseekV4 models to load in bfloat16.

    BUG: norms should be loaded in float32, but usually aren't due to the base
    model having a quant_config which overrides this. Loading in float32 actually
    breaks the model definition (it expects bfloat16). Let's force load in bfloat16.
    """
    with patch_attr(DeepseekV4PreTrainedModel, "_keep_in_fp32_modules_strict", set()):
        yield


@torch.no_grad()
@requires_gpu
@pytest.mark.parametrize(
    "model_stub,exp_keys",
    [
        (
            "inference-optimization/DSV4-tiny-empty",
            [
                "model.layers.0.mlp.experts.2.up_proj.weight",
                "model.layers.1.mlp.experts.0.gate_proj.weight",
                "model.layers.2.mlp.experts.1.down_proj.weight",
            ],
        )
    ],
)
def test_linearize_moe_model(
    model_stub, exp_keys, tmp_path, patch_deepseek_fp32_modules
):
    save_dir = tmp_path / "offload_dir"
    os.mkdir(save_dir)

    input_ids = torch.randint(1024, size=(1, 64), device="cuda")
    model = AutoModelForCausalLM.from_pretrained(model_stub, device_map="cuda")
    true_outputs = model(input_ids=input_ids).logits
    assert torch.any(true_outputs != 0), "Bad source of truth, all zeros"
    del model

    with load_quantizable_moe():
        model2 = AutoModelForCausalLM.from_pretrained(model_stub, device_map="cuda")

    select_exp_outputs = model2(input_ids=input_ids).logits
    assert torch.nn.functional.mse_loss(true_outputs, select_exp_outputs) < 1e-2

    with moe_calibration_context():
        all_exp_outputs = model2(input_ids=input_ids).logits
        assert torch.nn.functional.mse_loss(true_outputs, all_exp_outputs) < 1e-2

    model2.save_pretrained(save_dir)
    assert keys_exist(save_dir, exp_keys)


def keys_exist(model_path: Path, keys: list[str]) -> bool:
    """
    Utility to check that expected expert keys exist in a saved model.

    Args:
        model_path: Path to the saved model directory
        expected_patterns: List of key patterns to check for

    Returns:
        True if all expected patterns are found in the model checkpoint
    """
    safetensor_files = list(model_path.glob("*.safetensors"))
    all_keys = set()
    keys = set(keys)

    for st_file in safetensor_files:
        with safe_open(st_file, framework="pt", device="cpu") as f:
            all_keys.update(f.keys())

    return keys <= all_keys
