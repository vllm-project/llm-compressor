import pytest
import torch
from compressed_tensors.quantization import (
    ActivationOrdering,
    QuantizationArgs,
    QuantizationScheme,
)
from compressed_tensors.quantization.quant_args import QuantizationType
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier

recipe_str = """
quant_stage:
    quant_modifiers:
        GPTQModifier:
            ignore: ["lm_head"]
            config_groups:
                group_0:
                    weights:
                        num_bits: 4
                        type: "int"
                        symmetric: true
                        strategy: "channel"
                    targets: ["re:.*model.layers.2.self_attn.q_proj$"]
"""

recipe_modifier_full = GPTQModifier(
    ignore=["lm_head"],
    config_groups={
        "group_0": QuantizationScheme(
            targets=["re:.*model.layers.2.self_attn.q_proj$"],
            weights=QuantizationArgs(num_bits=4, strategy="channel"),
        )
    },
)

recipe_modifier_full_actorder_weight = GPTQModifier(
    ignore=["lm_head"],
    config_groups={
        "group_0": QuantizationScheme(
            targets=["re:.*model.layers.2.self_attn.q_proj$"],
            weights=QuantizationArgs(
                num_bits=4, strategy="channel", actorder=ActivationOrdering.WEIGHT
            ),
        )
    },
)

recipe_modifier_full_group = GPTQModifier(
    ignore=["lm_head"],
    config_groups={
        "group_0": QuantizationScheme(
            targets=["re:.*model.layers.2.self_attn.q_proj$"],
            weights=QuantizationArgs(num_bits=4, strategy="group", group_size=32),
        )
    },
)

recipe_modifier_shorthand_a = GPTQModifier(
    ignore=["lm_head"],
    config_groups={
        "group_0": QuantizationScheme(
            targets=["re:.*model.layers.2.self_attn.q_proj$"],
            weights=QuantizationArgs(num_bits=4, strategy="group", group_size=32),
        )
    },
)

recipe_modifier_shorthand_b = GPTQModifier(
    ignore=["lm_head"],
    config_groups={
        "group_0": QuantizationScheme(
            targets=["re:.*model.layers.2.self_attn.q_proj$"],
            weights=QuantizationArgs(num_bits=4, strategy="group", group_size=32),
        )
    },
)

# Test group quantization variants
recipe_modifier_group_actorder_weight = GPTQModifier(
    ignore=["lm_head"],
    config_groups={
        "group_0": QuantizationScheme(
            targets=["re:.*model.layers.2.self_attn.q_proj$"],
            weights=QuantizationArgs(
                num_bits=4,
                strategy="group",
                group_size=32,
                actorder=ActivationOrdering.WEIGHT,
            ),
        )
    },
)

recipe_modifier_group_actorder_group = GPTQModifier(
    ignore=["lm_head"],
    config_groups={
        "group_0": QuantizationScheme(
            targets=["re:.*model.layers.2.self_attn.q_proj$"],
            weights=QuantizationArgs(
                num_bits=4,
                strategy="group",
                group_size=32,
                actorder=ActivationOrdering.GROUP,
            ),
        )
    },
)

# Test block quantization variants
recipe_modifier_full_block = GPTQModifier(
    ignore=["lm_head"],
    config_groups={
        "group_0": QuantizationScheme(
            targets=["re:.*model.layers.2.self_attn.q_proj$"],
            weights=QuantizationArgs(
                num_bits=8,
                type=QuantizationType.FLOAT,
                strategy="block",
                block_structure=[2, 8],
            ),
        )
    },
)

recipe_modifier_block_actorder_weight = GPTQModifier(
    ignore=["lm_head"],
    config_groups={
        "group_0": QuantizationScheme(
            targets=["re:.*model.layers.2.self_attn.q_proj$"],
            weights=QuantizationArgs(
                num_bits=8,
                type=QuantizationType.FLOAT,
                strategy="block",
                block_structure=[2, 8],
                actorder=ActivationOrdering.WEIGHT,
            ),
        )
    },
)

# Exercises the modifier-level CHANNEL + actorder=WEIGHT path, complementary
# to recipe_modifier_full_actorder_weight (yaml-level). Post-CT #682 the
# QuantizationArgs validator allows actorder on non-group strategies except
# for actorder=GROUP, so the yaml form also validates; both paths are kept
# to cover the modifier-level resolve in GPTQModifier.resolve_quantization_config.
recipe_modifier_channel_actorder_weight = GPTQModifier(
    ignore=["lm_head"],
    actorder=ActivationOrdering.WEIGHT,
    config_groups={
        "group_0": QuantizationScheme(
            targets=["re:.*model.layers.2.self_attn.q_proj$"],
            weights=QuantizationArgs(num_bits=4, strategy="channel"),
        )
    },
)


@pytest.mark.parametrize(
    "recipe",
    [
        recipe_str,
        recipe_modifier_full,
        recipe_modifier_full_actorder_weight,
        recipe_modifier_full_group,
        recipe_modifier_shorthand_a,
        recipe_modifier_shorthand_b,
        recipe_modifier_group_actorder_weight,
        recipe_modifier_group_actorder_group,
        recipe_modifier_full_block,
        recipe_modifier_block_actorder_weight,
        recipe_modifier_channel_actorder_weight,
    ],
)
def test_oneshot_application(recipe, tmp_path):
    output = tmp_path / "oneshot_output"
    model_id = "nm-testing/tinysmokellama-3.2"
    dataset = "open_platypus"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Load original model for numerical comparison
    original_model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map=device
    )
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
        splits={"calibration": "train[:9]"},
    )
    model_loaded = AutoModelForCausalLM.from_pretrained(output, device_map=device)

    # Check that the model is quantized
    # decompress() will attach a quantization_config to the model
    # as we decompress right away
    quantization_config = model_loaded.config.quantization_config.quantization_config
    assert quantization_config is not None

    # check config is set properly
    assert "lm_head" in quantization_config.ignore
    assert len(quantization_config.config_groups) == 1
    quant_scheme = quantization_config.config_groups["group_0"]
    assert isinstance(quant_scheme, QuantizationScheme)
    assert quant_scheme.targets == ["re:.*model.layers.2.self_attn.q_proj$"]
    weight_args = quantization_config.config_groups["group_0"].weights
    assert isinstance(weight_args, QuantizationArgs)
    assert weight_args.num_bits == 4 or weight_args.num_bits == 8

    # Check a specific layer is quantized
    targetted_linear_layer = model_loaded.model.layers[2].self_attn.q_proj
    assert hasattr(targetted_linear_layer, "quantization_scheme")

    # Check lm-head is not quantized
    not_targetted = model_loaded.lm_head
    assert not hasattr(not_targetted, "quantization_scheme")

    # Verify g_idx behavior for activation ordering
    if weight_args.actorder == ActivationOrdering.GROUP:
        # GROUP actorder should save g_idx
        assert hasattr(
            targetted_linear_layer, "weight_g_idx"
        ), "GROUP actorder should have g_idx"
    elif weight_args.actorder == ActivationOrdering.WEIGHT:
        # WEIGHT actorder should NOT save g_idx (identity mapping)
        assert not hasattr(
            targetted_linear_layer, "weight_g_idx"
        ), "WEIGHT actorder should not have g_idx"

    # Numerical validation: check MSE
    with torch.no_grad():
        quantized_output = model_loaded(**inputs).logits

    mse = torch.nn.functional.mse_loss(quantized_output, original_output).item()

    # MSE threshold - quantization should not degrade quality too much
    mse_threshold = 0.015
    assert mse < mse_threshold, (
        f"MSE {mse:.6f} exceeds threshold {mse_threshold}. "
        f"Quantization degraded model quality too much."
    )

    # Cleanup
    del original_model, model_loaded
    torch.cuda.empty_cache()
