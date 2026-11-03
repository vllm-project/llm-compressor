"""
End-to-end oneshot test for expert pruning modifiers on a real MoE checkpoint.

Each parameterization runs oneshot with ``REAPPruningModifier`` or
``MoNePruningModifier`` on ``Qwen/Qwen1.5-MoE-A2.7B`` over 2048 ``perfectblend``
calibration samples. To keep calibration cheap, the model is cut down to its
first three decoder layers before oneshot (decoder layers are independent, so
the retained layers see exactly the activations they would see in the full
model; the per-expert token counts come out bit-identical and the saliency up
to float32 reduction-order noise). The un-pruned truncated checkpoint is saved
and the test then verifies that:

1. the retained/pruned expert choice is identical across
   - oneshot, recovered from the model's post-pruning router weights,
   - ``ExpertPruner``, built from the saliency report which oneshot wrote, and
   - the expected values recorded in ``test_e2e_expected.json``, and
2. the per-expert saliency statistics of the analyzed layers match the values
   recorded in ``test_e2e_expected.json``.

Router-row recovery relies on pruning only ever slicing and reordering router
rows, so each row of a pruned router is a bitwise copy of a row of the
un-pruned checkpoint's router. Saliency is compared with a tolerance, since it
is accumulated in float32 over bf16 activations whose exact values depend on
the execution hardware.

The expected values live in ``test_e2e_expected.json``, recorded from a
reference run of this very test (single A100-80GB, bf16 weights); to
regenerate them, read the written report and the recovered expert choices back
into that file. REAP and MoNe rank experts with different saliency metrics, so
each modifier has its own recorded values. The retained/pruned boundary is
decisive (the smallest saliency gap between a retained and a pruned expert
across all layers is ~0.2%), so the expert lists are only expected to
reproduce on hardware whose bf16 statistics match the reference run's to
better than that; re-record the values when that assumption breaks.
"""

import json
from pathlib import Path

import pytest
import torch
from compressed_tensors.utils.safetensors_load import (
    get_checkpoint_files,
    get_weight_map,
)
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

from llmcompressor import oneshot
from llmcompressor.entrypoints.converter import ExpertPruner
from llmcompressor.modifiers.pruning.mone import MoNePruningModifier
from llmcompressor.modifiers.pruning.reap import REAPPruningModifier
from llmcompressor.utils import load_context
from tests.testing_utils import requires_gpu

MODEL = "Qwen/Qwen1.5-MoE-A2.7B"

# a large calibration set makes the accumulated per-expert statistics (and
# with them the pruning decisions) robust
NUM_CALIBRATION_SAMPLES = 2048
MAX_SEQ_LENGTH = 512

# Qwen1.5-MoE-A2.7B (qwen2_moe) has 60 experts per layer with top-4 routing;
# 25% sparsity prunes the 15 least-salient experts per layer
SPARSITY = 0.25
NUM_EXPERTS = 60
NUM_RETAINED = NUM_EXPERTS - int(NUM_EXPERTS * SPARSITY)

# only the first three decoder layers of the model are kept (and analyzed)
NUM_LAYERS = 3

# the truncated checkpoint keeps the hub checkpoint's 2D per-expert weights
# (mlp.experts.<i>.<proj>) behind mlp.gate routers, which is the layout both
# the reload for oneshot and ExpertPruner must match on
ROUTER_PATTERN = r"mlp\.gate\.weight$"
EXPERT_PATTERN = r"mlp\.experts\.\d+\.(gate|up|down)_proj"

# saliency is accumulated in float32 over bf16 activations, whose exact values
# depend on the execution hardware; allow for reduction-order differences
SALIENCY_RTOL = 1e-2
SALIENCY_ATOL = 1e-2

# expected expert choices and per-expert statistics, recorded from a reference
# run of this test (see the module docstring)
EXPECTED_PATH = Path(__file__).parent / "test_e2e_expected.json"
EXPECTED = json.loads(EXPECTED_PATH.read_text())


def _save_truncated_checkpoint(directory):
    """Save ``Qwen1.5-MoE-A2.7B`` cut down to its first ``NUM_LAYERS`` decoder
    layers as an un-pruned checkpoint.

    Building the model from a config with a reduced ``num_hidden_layers``
    loads only the retained layers' weights. The linearized experts serialize
    in the hub checkpoint's native 2D per-expert layout, so the saved directory
    is a standard qwen2_moe checkpoint which ``load_context`` re-loads in
    linearized form for oneshot. The processor is saved alongside the weights:
    oneshot infers it from the model's path, and without it the calibration
    data would be tokenized by a fallback processor.
    """
    config = AutoConfig.from_pretrained(MODEL)
    config.num_hidden_layers = NUM_LAYERS
    config.layer_types = config.layer_types[:NUM_LAYERS]
    with load_context():
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, config=config, dtype=torch.bfloat16, device_map="auto_offload"
        )
    assert len(model.model.layers) == NUM_LAYERS

    model.save_pretrained(directory)
    AutoProcessor.from_pretrained(MODEL).save_pretrained(directory)

    del model
    torch.accelerator.empty_cache()


@pytest.fixture(scope="module")
def truncated_model_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("truncated_model")
    _save_truncated_checkpoint(directory)
    return directory


def _router_name(layer: int) -> str:
    return f"model.layers.{layer}.mlp.gate.weight"


def _original_router_weights(directory) -> dict[int, torch.Tensor]:
    """Router weights of the un-pruned truncated checkpoint, by layer."""
    model_files = get_checkpoint_files(directory)
    weight_map = get_weight_map(model_files)
    routers = {}
    for layer in range(NUM_LAYERS):
        name = _router_name(layer)
        with safe_open(model_files[weight_map[name]], framework="pt") as f:
            routers[layer] = f.get_tensor(name)
    return routers


def _match_rows(rows: torch.Tensor, original: torch.Tensor) -> list[int]:
    """For each row of ``rows``, the index of its exact match in ``original``.

    Expert pruning only ever slices and reorders router rows, so each row of a
    pruned router is a bitwise copy of exactly one original row.
    """
    matches = (original.unsqueeze(0) == rows.unsqueeze(1)).all(dim=-1)
    assert bool(
        (matches.sum(dim=1) == 1).all()
    ), "each router row must match exactly one row of the checkpoint"
    return matches.to(torch.uint8).argmax(dim=1).tolist()


def _oneshot_expert_choice(
    model: torch.nn.Module,
    original: dict[int, torch.Tensor],
) -> tuple[list[list[int]], list[list[int]]]:
    """Recover the experts each layer kept and dropped from the post-oneshot
    model, by matching router rows against the un-pruned checkpoint.

    REAP shrinks each pruned layer's router to its retained experts, while MoNe
    keeps the full-width router reordered to ``[retained..., dropped...]``, so
    the first ``NUM_RETAINED`` matched indices are the retained experts in both
    cases and (for MoNe) the remaining matched indices are the dropped ones.
    """
    retained: list[list[int]] = []
    pruned: list[list[int]] = []
    for layer in range(NUM_LAYERS):
        router = model.get_submodule(f"model.layers.{layer}.mlp.gate")
        order = _match_rows(router.weight.detach().to("cpu"), original[layer])
        retained.append(order[:NUM_RETAINED])
        if len(order) == NUM_EXPERTS:
            pruned.append(order[NUM_RETAINED:])
        else:
            assert len(order) == NUM_RETAINED, (
                f"layer {layer}: pruned router has {len(order)} rows, expected "
                f"{NUM_RETAINED} (REAP) or {NUM_EXPERTS} (MoNe)"
            )
            pruned.append(sorted(set(range(NUM_EXPERTS)) - set(order)))
    return retained, pruned


@pytest.mark.integration
@requires_gpu
@pytest.mark.parametrize(
    "modifier_cls, expected",
    [
        (REAPPruningModifier, EXPECTED["reap"]),
        (MoNePruningModifier, EXPECTED["mone"]),
    ],
    ids=["reap", "mone"],
)
def test_oneshot_expert_selection(
    modifier_cls,
    expected,
    truncated_model_dir,
    tmp_path,
):
    """The expert choice recovered from the pruned model, the choice
    ``ExpertPruner`` makes from the saliency report, and the report's
    statistics all match the values recorded for ``modifier_cls``."""
    torch.manual_seed(42)
    torch.get_device_module().manual_seed_all(42)

    expected_retained = expected["retained"]
    expected_pruned = expected["pruned"]
    expected_saliency = expected["saliency"]

    # sanity-check the recorded expectations: retained and pruned experts
    # partition each layer's experts
    for layer in range(NUM_LAYERS):
        assert sorted(expected_retained[layer] + expected_pruned[layer]) == list(
            range(NUM_EXPERTS)
        ), f"layer {layer}: retained and pruned experts must partition all experts"

    report_path = str(tmp_path / "saliency_report.json")

    model = None
    try:
        with load_context():
            model = AutoModelForCausalLM.from_pretrained(
                truncated_model_dir, dtype=torch.bfloat16, device_map="auto_offload"
            )

        oneshot(
            model=model,
            dataset="perfectblend",
            splits=f"train[:{NUM_CALIBRATION_SAMPLES}]",
            recipe=modifier_cls(sparsity=SPARSITY, report_path=report_path),
            num_calibration_samples=NUM_CALIBRATION_SAMPLES,
            max_seq_length=MAX_SEQ_LENGTH,
            shuffle_calibration_samples=False,
            pipeline="sequential",
            moe_calibrate_all_experts=False,
            propagate_error=False,
        )

        # 1a. the pruning oneshot applied to the model
        original = _original_router_weights(truncated_model_dir)
        retained, pruned = _oneshot_expert_choice(model, original)
        for layer in range(NUM_LAYERS):
            assert (
                retained[layer] == expected_retained[layer]
            ), f"layer {layer}: oneshot retained experts differ"
            assert (
                pruned[layer] == expected_pruned[layer]
            ), f"layer {layer}: oneshot pruned experts differ"

        # the model's expert-sparsity config records the per-layer expert counts
        sparsity_config = model._moe_sparsity_config
        assert sparsity_config.method == modifier_cls.sparsity_method
        assert sparsity_config.num_experts == [NUM_RETAINED] * NUM_LAYERS

        # 1b. the choice ExpertPruner makes from the same saliency report
        pruner = ExpertPruner.from_pretrained(
            truncated_model_dir,
            sparsity=SPARSITY,
            metric="saliency",
            uniform=True,
            router_pattern=ROUTER_PATTERN,
            expert_pattern=EXPERT_PATTERN,
            saliency_report_path=report_path,
        )
        for layer in range(NUM_LAYERS):
            assert (
                pruner.retained_experts[_router_name(layer)] == expected_retained[layer]
            ), f"layer {layer}: ExpertPruner retained experts differ"

        # 2. the saliency statistics of the analyzed layers
        with open(report_path) as f:
            report = json.load(f)
        assert (
            len(report["saliency"]) == NUM_LAYERS
        ), "expected saliency statistics for every layer of the truncated model"
        for layer in range(NUM_LAYERS):
            torch.testing.assert_close(
                torch.tensor(report["saliency"][layer], dtype=torch.float64),
                torch.tensor(expected_saliency[layer], dtype=torch.float64),
                rtol=SALIENCY_RTOL,
                atol=SALIENCY_ATOL,
                msg=lambda m, i=layer: f"layer {i}: saliency differs\n{m}",
            )
    finally:
        del model
        torch.accelerator.empty_cache()
