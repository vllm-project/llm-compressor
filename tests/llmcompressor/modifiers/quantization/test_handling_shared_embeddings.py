import os

import pytest
import torch
from transformers import AutoModelForCausalLM

from llmcompressor.core import State
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform import QuIPModifier, SpinQuantModifier
from tests.testing_utils import requires_gpu

torch.manual_seed(0)

_EXP_MSE = 8e-3


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
    model_id = "nm-testing/tinysmokellama-3.2"

    # Test 1: Apply quantization WITHOUT manually untieing first
    # (relies on automatic untieing in start_calibration)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="cuda", torch_dtype=torch.bfloat16
    )

    # Verify embeddings are initially tied
    input_embed = model.get_input_embeddings()
    output_embed = model.get_output_embeddings()
    assert (
        input_embed.weight is output_embed.weight
    ), "test setup failed, need to test a model with shared input/output embeddings"

    state = State(model=model)
    modifier = QuantizationModifier(
        scheme="W8A8",
        targets=[r"re:.*q_proj$", r"re:.*k_proj$", r"re:.*lm_head$"],
    )

    input_data = {k: v.to("cuda") for k, v in model.dummy_inputs.items()}

    with torch.no_grad():
        baseline_output = model(**input_data)

    # Initialize and start calibration (should automatically untie if needed)
    modifier.on_initialize(state)
    modifier.on_start(state, None)

    with torch.no_grad():
        output = model(**input_data)

    assert torch.nn.MSELoss()(output.logits, baseline_output.logits) <= _EXP_MSE

    # Verify that embeddings were untied
    assert input_embed.weight is not output_embed.weight, (
        "expected input_embed.weight to be different from output_embed.weight"
        + "but found that they are still the same"
    )


@requires_gpu
@pytest.mark.skipif(
    (not os.getenv("HF_TOKEN")),
    reason="Skipping correctness tests requiring gated model access",
)
def test_quantization_untie_only_when_targeted():
    """
    Test that embeddings are only untied when they
    are actually targeted for quantization.

    This verifies that the _untie_if_target_shared logic
    correctly checks if embeddings are in the target list before untieing.
    """
    model_id = "nm-testing/tinysmokellama-3.2"

    # Test with targets that don't include embeddings
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="cuda", torch_dtype=torch.bfloat16
    )

    # Verify embeddings are initially tied
    input_embed = model.get_input_embeddings()
    output_embed = model.get_output_embeddings()
    assert (
        input_embed.weight is output_embed.weight
    ), "test setup failed, need to test a model with shared input/output embeddings"

    state = State(model=model)
    # Only target attention projection layers, not embeddings
    modifier = QuantizationModifier(
        scheme="W8A8", targets=[r"re:.*q_proj$", r"re:.*k_proj$"]
    )

    input_data = {k: v.to("cuda") for k, v in model.dummy_inputs.items()}

    with torch.no_grad():
        baseline_output = model(**input_data)

    modifier.on_initialize(state)
    modifier.on_start(state, None)

    with torch.no_grad():
        output = model(**input_data)

    assert torch.nn.MSELoss()(output.logits, baseline_output.logits) <= _EXP_MSE

    # Verify embeddings are still tied (since they weren't targeted)
    assert (
        input_embed.weight is output_embed.weight
    ), "Embeddings were untied even though they were not in the target list"


@requires_gpu
@pytest.mark.skipif(
    (not os.getenv("HF_TOKEN")),
    reason="Skipping correctness tests requiring gated model access",
)
@pytest.mark.parametrize("rotations", [["R1"], ["R2"], ["R4"]])
def test_spinquant_with_tied_embeddings(rotations):
    """
    Test that SpinQuant with rotations properly handles tied embeddings.

    This test verifies that:
    1. SpinQuant can be applied to a model with tied embeddings
    2. When SpinQuant is applied, embeddings are automatically untied
    3. SpinQuant works correctly with the new untie functionality
    """
    model_id = "nm-testing/tinysmokellama-3.2"

    # Test with R1 rotation (should untie embeddings)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="cuda", torch_dtype=torch.bfloat16
    )

    # Verify embeddings are initially tied
    input_embed = model.get_input_embeddings()
    output_embed = model.get_output_embeddings()
    assert (
        input_embed.weight is output_embed.weight
    ), "test setup failed, need to test a model with shared input/output embeddings"

    state = State(model=model)

    spinquant_modifier = SpinQuantModifier(
        rotations=rotations, transform_type="random-hadamard"
    )

    input_data = {k: v.to("cuda") for k, v in model.dummy_inputs.items()}

    with torch.no_grad():
        baseline_output = model(**input_data)

    spinquant_modifier.on_initialize(state)
    spinquant_modifier.on_start(state, None)

    # Verify embeddings were untied by SpinQuant
    assert (
        input_embed.weight is not output_embed.weight
    ), f"SpinQuant {rotations} should have untied embeddings but they are still tied"

    with torch.no_grad():
        output = model(**input_data)

    assert torch.nn.MSELoss()(output.logits, baseline_output.logits) <= _EXP_MSE


@requires_gpu
@pytest.mark.skipif(
    (not os.getenv("HF_TOKEN")),
    reason="Skipping correctness tests requiring gated model access",
)
@pytest.mark.parametrize(
    "rotations",
    [
        ["v"],
        # ["u"], ["v", "u"]
    ],
)
def test_quip_with_tied_embeddings(rotations):
    """
    Test that QuIP with rotations properly handles tied embeddings.

    This test verifies that:
    1. QuIP can be applied to a model with tied embeddings
    2. When QuIP targets lm_head (which shares weights with embeddings),
       embeddings are automatically untied

    No accuracy checks are done because inverting the
    random-matrix is too innacurate
    """
    model_id = "nm-testing/tinysmokellama-3.2"

    # Test with QuIP rotations
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="cuda", torch_dtype=torch.bfloat16
    )

    # Verify embeddings are initially tied
    input_embed = model.get_input_embeddings()
    output_embed = model.get_output_embeddings()
    assert (
        input_embed.weight is output_embed.weight
    ), "test setup failed, need to test a model with shared input/output embeddings"

    state = State(model=model)

    # Apply QuIP with specified rotations
    # Set ignore=[] to ensure lm_head is targeted (default is ignore="lm_head")
    # Use random-matrix since lm_head has dimension which may not be power of 2
    quip_modifier = QuIPModifier(
        rotations=rotations, transform_type="random-matrix", ignore=[]
    )

    # Initialize QuIP
    quip_modifier.on_initialize(state)

    # Embeddings should still be tied after initialization
    assert (
        input_embed.weight is output_embed.weight
    ), "Embeddings should still be tied after on_initialize"

    # Start QuIP (should untie embeddings if lm_head is targeted)
    quip_modifier.on_start(state, None)

    # Verify embeddings were untied by QuIP during on_start
    assert (
        input_embed.weight is not output_embed.weight
    ), "QuIP should have untied embeddings but they are still tied"


@requires_gpu
@pytest.mark.skipif(
    (not os.getenv("HF_TOKEN")),
    reason="Skipping correctness tests requiring gated model access",
)
@pytest.mark.parametrize(
    "rotations",
    [
        ["v"],
    ],
)
def test_quip_untie_only_when_targeted(rotations):
    """
    Test that QuIP only unties embeddings when they are actually targeted.

    This verifies the surgical nature of the fix: when lm_head is in the ignore
    list (the default), embeddings should remain tied since they won't be
    transformed.
    """
    model_id = "nm-testing/tinysmokellama-3.2"

    # Test with QuIP with default ignore (includes lm_head)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="cuda", torch_dtype=torch.bfloat16
    )

    # Verify embeddings are initially tied
    input_embed = model.get_input_embeddings()
    output_embed = model.get_output_embeddings()
    assert (
        input_embed.weight is output_embed.weight
    ), "test setup failed, need to test a model with shared input/output embeddings"

    state = State(model=model)

    # Apply QuIP with default ignore (lm_head is ignored by default)
    quip_modifier = QuIPModifier(
        rotations=rotations,
        transform_type="random-matrix",
        # ignore lm_head is the default
    )

    # Initialize and start QuIP
    quip_modifier.on_initialize(state)
    quip_modifier.on_start(state, None)

    # Verify embeddings are still tied (since lm_head was ignored)
    assert (
        input_embed.weight is output_embed.weight
    ), "Embeddings should still be tied when lm_head is in the ignore list"
