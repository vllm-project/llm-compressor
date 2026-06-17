"""
Test that fused observers work correctly with sequential calibration.

This tests the fix for fused observer handling where sequential calibration
with TENSOR_GROUP quantization would fail because only some fused observers
would be observed (e.g., q_proj observed but k_proj and v_proj skipped).
"""

import torch

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier


def test_fused_observers_with_linear_sequential_targets(tmp_path):
    """
    Test that oneshot with sequential_targets properly handles fused observers.

    This test uses TENSOR_GROUP quantization (like NVFP4) which requires q/k/v
    observers to be fused (share global_scale). With sequential_targets=["Linear"],
    the sequential pipeline processes each Linear individually.
    """
    from transformers import AutoModelForCausalLM

    model_id = "nm-testing/tinysmokellama-3.2"
    output = tmp_path / "quantized_output"

    # Recipe with TENSOR_GROUP quantization (like NVFP4)
    recipe = QuantizationModifier(
        targets=["Linear"],
        scheme="NVFP4",  # Uses TENSOR_GROUP strategy
    )

    # Run oneshot with sequential_targets=["Linear"]
    # This triggers sequential calibration at the Linear layer level
    # On main, this would fail with:
    # "All fused observers must be run before get_qparams"
    # because when processing q_proj, k_proj and v_proj don't get observed
    oneshot(
        model=model_id,
        dataset="open_platypus",
        recipe=recipe,
        output_dir=output,
        num_calibration_samples=4,
        splits={"calibration": "train[:4]"},
        sequential_targets=["Linear"],  # Sequential calibration at Linear level
    )

    # Load the quantized model to verify
    model = AutoModelForCausalLM.from_pretrained(output)

    # Verify all fused modules have qparams
    # Check the first layer's attention q/k/v projections
    first_attn = model.model.layers[0].self_attn

    # All should have global_scale (TENSOR_GROUP qparam)
    assert hasattr(
        first_attn.q_proj, "weight_global_scale"
    ), "q_proj should have weight_global_scale"
    assert hasattr(
        first_attn.k_proj, "weight_global_scale"
    ), "k_proj should have weight_global_scale (FAILS on main)"
    assert hasattr(
        first_attn.v_proj, "weight_global_scale"
    ), "v_proj should have weight_global_scale (FAILS on main)"

    # Global scales should be equal (fused)
    q_scale = first_attn.q_proj.weight_global_scale
    k_scale = first_attn.k_proj.weight_global_scale
    v_scale = first_attn.v_proj.weight_global_scale

    assert torch.allclose(q_scale, k_scale), "q and k should have same global_scale"
    assert torch.allclose(q_scale, v_scale), "q and v should have same global_scale"
