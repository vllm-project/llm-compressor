"""
Architecture-support oneshot tests for the expert pruning modifiers.

Each parameterization runs oneshot with ``REAPPruningModifier``,
``HOPEPruningModifier``, ``RazorPruningModifier``, or ``MoNePruningModifier``
on a tiny real MoE checkpoint, validating that the whole pipeline -- MoE layer
detection, calibration hooks, saliency accumulation, and the structural
collapse -- runs end to end on that architecture and leaves a usable model.

Two architectures are covered, exercising the two router layouts the modifiers
must handle:

- ``inference-optimization/gemma-4-1B-0.8B-tiny`` (gemma4): a standard MoE
  whose router is a normed, scaled projection returning softmaxed
  probabilities (no bare ``weight``), with a ``per_expert_scale``, and whose
  router/experts are direct children of the decoder layer.
- ``inference-optimization/NemotronH-0.3B-A0.3B`` (nemotron_h): a latent MoE
  whose experts consume a down-projection of the block input while the router
  scores the block input itself, with a sigmoid score function, an
  ``e_score_correction_bias``, and a single (inert) routing group.

These are correctness-of-support smoke tests, not golden-value tests: the
checkpoints are tiny, so the exact pruning choice is not asserted. Each test
checks that oneshot prunes the expected number of experts from every MoE layer,
records it in the expert-sparsity config, and leaves a model that still runs.
"""

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from llmcompressor import oneshot
from llmcompressor.modifiers.pruning.hope import HOPEPruningModifier
from llmcompressor.modifiers.pruning.mone import MoNePruningModifier
from llmcompressor.modifiers.pruning.razor import RazorPruningModifier
from llmcompressor.modifiers.pruning.reap import REAPPruningModifier
from llmcompressor.utils import load_context
from tests.testing_utils import requires_gpu

# calibration is kept small: these tests validate that the pipeline runs on
# each architecture, not the quality of the pruning decision
NUM_CALIBRATION_SAMPLES = 32
MAX_SEQ_LENGTH = 128
SPARSITY = 0.25


class ModelSpec:
    """A tiny MoE checkpoint and its expected post-pruning expert counts."""

    def __init__(self, model_id: str, num_experts: int, num_moe_layers: int):
        self.model_id = model_id
        self.num_experts = num_experts
        self.num_moe_layers = num_moe_layers
        self.retained = num_experts - int(num_experts * SPARSITY)


MODELS = {
    "gemma4": ModelSpec(
        "inference-optimization/gemma-4-1B-0.8B-tiny",
        num_experts=16,
        num_moe_layers=6,
    ),
    "nemotron_h": ModelSpec(
        "inference-optimization/NemotronH-0.3B-A0.3B",
        num_experts=32,
        num_moe_layers=2,
    ),
}

MODIFIERS = {
    "reap": REAPPruningModifier,
    "hope": HOPEPruningModifier,
    "razor": RazorPruningModifier,
    "mone": MoNePruningModifier,
}


@pytest.mark.integration
@requires_gpu
@pytest.mark.parametrize("model_key", list(MODELS), ids=list(MODELS))
@pytest.mark.parametrize("modifier_key", list(MODIFIERS), ids=list(MODIFIERS))
def test_expert_pruning_supports_architecture(model_key, modifier_key):
    """oneshot prunes every MoE layer of the checkpoint to its retained experts,
    records the per-layer counts in the expert-sparsity config, and leaves a
    model that still runs."""
    torch.manual_seed(42)
    spec = MODELS[model_key]
    modifier_cls = MODIFIERS[modifier_key]

    with load_context():
        model = AutoModelForCausalLM.from_pretrained(
            spec.model_id, dtype=torch.bfloat16, device_map="auto_offload"
        )

    oneshot(
        model=model,
        dataset="open_platypus",
        splits=f"train[:{NUM_CALIBRATION_SAMPLES}]",
        recipe=modifier_cls(sparsity=SPARSITY),
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
        max_seq_length=MAX_SEQ_LENGTH,
        shuffle_calibration_samples=False,
        pipeline="sequential",
        moe_calibrate_all_experts=False,
        propagate_error=False,
    )

    # the expert-sparsity config records the method and the per-layer retained
    # expert counts (None for the model's dense / non-MoE layers)
    sparsity_config = model._moe_sparsity_config
    assert sparsity_config is not None, "oneshot recorded no expert-sparsity config"
    assert sparsity_config.method == modifier_cls.sparsity_method

    pruned_counts = [c for c in sparsity_config.num_experts if c is not None]
    assert len(pruned_counts) == spec.num_moe_layers, (
        f"expected {spec.num_moe_layers} pruned MoE layers, got "
        f"{len(pruned_counts)}: {sparsity_config.num_experts}"
    )
    assert all(count == spec.retained for count in pruned_counts), (
        f"expected every MoE layer pruned to {spec.retained} experts, got "
        f"{pruned_counts}"
    )

    # the pruned model still runs and produces finite logits
    with torch.no_grad():
        input_ids = torch.tensor([[1, 2, 3, 4, 5]], device=model.device)
        logits = model(input_ids).logits
    assert torch.isfinite(logits).all(), "pruned model produced non-finite logits"


@pytest.mark.integration
def test_model_specs_match_configs():
    """The expected expert counts hard-coded above still match the checkpoints'
    configs (a cheap guard against the tiny models being re-published with
    different shapes)."""
    for spec in MODELS.values():
        config = AutoConfig.from_pretrained(spec.model_id).get_text_config()
        num_experts = getattr(config, "num_experts", None) or getattr(
            config, "n_routed_experts", None
        )
        assert num_experts == spec.num_experts, (
            f"{spec.model_id}: config reports {num_experts} experts, "
            f"test expects {spec.num_experts}"
        )
