import pytest
import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq import AWQModifier

# AWQModifier only computes/applies smoothing scales - it must be paired with a
# QuantizationMixin modifier (e.g. QuantizationModifier) to actually produce a
# compressed/quantized checkpoint.
recipe_str = """
quant_stage:
    quant_modifiers:
        AWQModifier:
            duo_scaling: "both"
        QuantizationModifier:
            ignore: ["lm_head"]
            config_groups:
                group_0:
                    targets: ["Linear"]
                    weights:
                        num_bits: 4
                        type: "int"
                        symmetric: true
                        strategy: "group"
                        group_size: 32
"""

# Note: tinysmokeqwen3 has hidden_size=64, so most Linear layers (q/k/v/gate/up
# proj) have in_features=64. A group strategy with group_size=128 (e.g. the
# W4A16_ASYM preset) cannot divide evenly and raises an unflatten error, so this
# variant uses "channel" strategy (per-output-channel, no group_size) to cover
# a distinct quantization strategy from the group-based recipes below.
recipe_modifier_full = [
    AWQModifier(duo_scaling="both"),
    QuantizationModifier(
        ignore=["lm_head"],
        config_groups={
            "group_0": QuantizationScheme(
                targets=["Linear"],
                weights=QuantizationArgs(
                    num_bits=4, symmetric=False, strategy="channel"
                ),
            )
        },
    ),
]

recipe_modifier_config_groups = [
    AWQModifier(duo_scaling="both"),
    QuantizationModifier(
        ignore=["lm_head"],
        config_groups={
            "group_0": QuantizationScheme(
                targets=["Linear"],
                weights=QuantizationArgs(num_bits=4, strategy="group", group_size=32),
            )
        },
    ),
]


@pytest.mark.parametrize(
    "recipe",
    [
        recipe_str,
        recipe_modifier_full,
        recipe_modifier_config_groups,
    ],
)
def test_oneshot_application(recipe, tmp_path):
    output = tmp_path / "oneshot_output"
    model_id = "nm-testing/tinysmokeqwen3"
    dataset = "open_platypus"
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    # Load original model for numerical comparison
    original_model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Create test input
    test_text = "The quick brown fox jumps over the lazy dog"
    inputs = tokenizer(test_text, return_tensors="pt").to(device)

    # Get original model output
    with torch.no_grad():
        original_output = original_model(**inputs).logits

    # Quantize model
    oneshot(
        model=model_id,
        dataset=dataset,
        output_dir=output,
        recipe=recipe,
        num_calibration_samples=9,
        splits="train[:9]",
        max_seq_length=512,
    )
    model_loaded = AutoModelForCausalLM.from_pretrained(output, device_map=device)

    # Check that the model is quantized
    # decompress() will attach a quantization_config to the model
    # as we decompress right away
    quantization_config = model_loaded.config.quantization_config.quantization_config
    assert quantization_config is not None

    # check config is set properly
    assert "lm_head" in quantization_config.ignore

    # Check a specific layer is quantized
    targetted_linear_layer = model_loaded.model.layers[0].self_attn.q_proj
    assert hasattr(targetted_linear_layer, "quantization_scheme")

    # Check lm-head is not quantized
    not_targetted = model_loaded.lm_head
    assert not hasattr(not_targetted, "quantization_scheme")

    # Numerical validation: check MSE
    with torch.no_grad():
        quantized_output = model_loaded(**inputs).logits

    mse = torch.nn.functional.mse_loss(quantized_output, original_output).item()

    # MSE threshold - quantization should not degrade quality too much
    mse_threshold = 0.05
    assert mse < mse_threshold, (
        f"MSE {mse:.6f} exceeds threshold {mse_threshold}. "
        f"Quantization degraded model quality too much."
    )

    # Cleanup
    del original_model, model_loaded
    if torch.accelerator.is_available():
        torch.accelerator.empty_cache()
