import os

import pytest
import torch
from transformers import AutoModelForCausalLM

from llmcompressor.core import State
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.transformers.compression.compressed_tensors_utils import (
    untie_word_embeddings,
)
from tests.testing_utils import requires_gpu


@requires_gpu
@pytest.mark.skipif(
    (not os.getenv("HF_TOKEN")),
    reason="Skipping correctness tests requiring gated model access",
)
def test_quantization_with_automatic_untie():
    """
    Test that quantization with automatic untie_word_embeddings produces the same
    results as manually calling untie_word_embeddings first.

    This test verifies the functionality added to QuantizationMixin where
    _untie_if_target_shared is called during start_calibration to automatically
    handle shared input/output embeddings when they are targeted for quantization.
    """
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    scheme = "W8A8"
    targets = ["Linear"]  # This targets lm_head (Linear layer)

    # Test 1: Apply quantization WITHOUT manually untieing first
    # (relies on automatic untieing in start_calibration)
    model_auto = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="cuda", torch_dtype=torch.float32
    )

    # Verify embeddings are initially tied
    input_embed = model_auto.get_input_embeddings()
    output_embed = model_auto.get_output_embeddings()
    initial_tied = input_embed.weight is output_embed.weight

    state_auto = State(model=model_auto)
    modifier_auto = QuantizationModifier(scheme=scheme, targets=targets)

    input_data = {k: v.to("cuda") for k, v in model_auto.dummy_inputs.items()}

    with torch.no_grad():
        baseline_output = model_auto(**input_data)

    # Initialize and start calibration (should automatically untie if needed)
    modifier_auto.on_initialize(state_auto)
    modifier_auto.on_start(state_auto, None)

    with torch.no_grad():
        output_auto = model_auto(**input_data)

    # Verify that embeddings were untied
    auto_untied = input_embed.weight is not output_embed.weight

    # Clean up
    del model_auto, state_auto, modifier_auto
    torch.cuda.empty_cache()

    # Test 2: Apply quantization WITH manual untieing first
    model_manual = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="cuda", torch_dtype=torch.float32
    )

    # Manually untie before quantization
    untie_word_embeddings(model_manual)

    state_manual = State(model=model_manual)
    modifier_manual = QuantizationModifier(scheme=scheme, targets=targets)

    # Initialize and start calibration
    modifier_manual.on_initialize(state_manual)
    modifier_manual.on_start(state_manual, None)

    with torch.no_grad():
        output_manual = model_manual(**input_data)

    # Verify that automatic and manual untieing produce identical results
    # The outputs should be identical (or very close due to numerical precision)
    mse = torch.nn.MSELoss()(output_auto.logits, output_manual.logits)

    # For quantization, we expect very low MSE between the two approaches
    # (they should be essentially identical)
    assert mse < 1e-6, (
        f"Automatic untie produced different results than manual untie. "
        f"MSE: {mse:.2e}. Initial embeddings tied: {initial_tied}, "
        f"Auto untied: {auto_untied}"
    )

    # Also verify that both are close to baseline (within quantization error)
    mse_auto_baseline = torch.nn.MSELoss()(output_auto.logits, baseline_output.logits)
    mse_manual_baseline = torch.nn.MSELoss()(
        output_manual.logits, baseline_output.logits
    )

    # MSE from baseline should be similar for both approaches
    assert abs(mse_auto_baseline - mse_manual_baseline) < 1e-5, (
        f"Auto and manual untie have different accuracy relative to baseline. "
        f"Auto MSE: {mse_auto_baseline:.2e}, Manual MSE: {mse_manual_baseline:.2e}"
    )


@requires_gpu
@pytest.mark.skipif(
    (not os.getenv("HF_TOKEN")),
    reason="Skipping correctness tests requiring gated model access",
)
def test_quantization_untie_only_when_targeted():
    """
    Test that embeddings are only untied when they are actually targeted for quantization.

    This verifies that the _untie_if_target_shared logic correctly checks if embeddings
    are in the target list before untieing.
    """
    model_id = "meta-llama/Llama-3.2-1B-Instruct"

    # Test with targets that don't include embeddings
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="cuda", torch_dtype=torch.float32
    )

    # Verify embeddings are initially tied
    input_embed = model.get_input_embeddings()
    output_embed = model.get_output_embeddings()
    initial_tied = input_embed.weight is output_embed.weight

    state = State(model=model)
    # Only target attention projection layers, not embeddings
    modifier = QuantizationModifier(
        scheme="W8A8", targets=[r"re:.*q_proj$", r"re:.*k_proj$"]
    )

    modifier.on_initialize(state)
    modifier.on_start(state, None)

    # Verify embeddings are still tied (since they weren't targeted)
    still_tied = input_embed.weight is output_embed.weight

    # If they were initially tied and not targeted, they should remain tied
    if initial_tied:
        assert still_tied, (
            "Embeddings were untied even though they were not in the target list"
        )


@requires_gpu
@pytest.mark.skipif(
    (not os.getenv("HF_TOKEN")),
    reason="Skipping correctness tests requiring gated model access",
)
def test_quantization_with_ignore_list():
    """
    Test that embeddings remain tied when they are in the ignore list,
    even if they match the target pattern.
    """
    model_id = "meta-llama/Llama-3.2-1B-Instruct"

    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="cuda", torch_dtype=torch.float32
    )

    # Verify embeddings are initially tied
    input_embed = model.get_input_embeddings()
    output_embed = model.get_output_embeddings()
    initial_tied = input_embed.weight is output_embed.weight

    state = State(model=model)
    # Target all Linear layers but ignore lm_head
    modifier = QuantizationModifier(
        scheme="W8A8", targets=["Linear"], ignore=["lm_head"]
    )

    modifier.on_initialize(state)
    modifier.on_start(state, None)

    # Verify embeddings are still tied (lm_head was ignored)
    still_tied = input_embed.weight is output_embed.weight

    # When lm_head is ignored, it shouldn't match in resolved_targets,
    # so embeddings should remain tied (assuming embed_tokens isn't Linear)
    if initial_tied:
        assert still_tied, (
            "Embeddings were untied even though lm_head was in ignore list"
        )