"""
End-to-end test for MoNe expert collapse on a real Qwen3 MoE checkpoint.

Verifies that an oneshot MoNe run produces a checkpoint whose ``config.json`` and
weights follow the MoNe serialization design:

* ``config.json`` contains ``quantization_config.moe_sparsity_config`` with
  ``method="novice"`` and a per-decoder-layer ``num_experts`` list, and
* each collapsed MoE layer serializes its retained experts (left-shifted) plus a
  single ``experts.novices`` tensor, while the router keeps its full width.
"""

import glob
import json
import os
import tempfile

import pytest
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM

from llmcompressor import oneshot
from llmcompressor.modeling.moe.linearize import repack_moe
from llmcompressor.modifiers.pruning.mone import MoNePruningModifier
from llmcompressor.utils import load_context
from tests.testing_utils import requires_gpu

QWEN_MODEL = "inference-optimization/Qwen3-1.6B-A0.9B"
NUM_SAMPLES = 16
MAX_SEQ_LENGTH = 512
SPARSITY = 0.25


@pytest.mark.integration
@requires_gpu
def test_mone_e2e_qwen3_serialization():
    torch.manual_seed(42)
    torch.get_device_module().manual_seed_all(42)

    with load_context():
        model = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL, dtype=torch.bfloat16, device_map="auto_offload"
        )

    num_experts = model.config.num_experts
    expected_retained = num_experts - int(num_experts * SPARSITY)

    oneshot(
        model=model,
        dataset="perfectblend",
        splits=f"train[:{NUM_SAMPLES}]",
        recipe=MoNePruningModifier(sparsity=SPARSITY),
        num_calibration_samples=NUM_SAMPLES,
        max_seq_length=MAX_SEQ_LENGTH,
        shuffle_calibration_samples=False,
        pipeline="sequential",
        moe_calibrate_all_experts=False,
    )

    # router width is preserved and novices installed on every collapsed layer
    collapsed_layers = 0
    for module in model.modules():
        novices = getattr(module, "novices", None)
        if isinstance(novices, torch.nn.Parameter):
            collapsed_layers += 1
            assert novices.shape[0] == num_experts - expected_retained
    assert collapsed_layers > 0

    repack_moe(model)

    with tempfile.TemporaryDirectory() as save_dir:
        model.save_pretrained(save_dir, save_compressed=True)

        with open(os.path.join(save_dir, "config.json")) as f:
            config_data = json.load(f)

        msc = config_data["quantization_config"]["moe_sparsity_config"]
        assert msc["method"] == "novice"

        per_layer = msc["num_experts"]
        assert len(per_layer) == model.config.num_hidden_layers
        collapsed_entries = [n for n in per_layer if n is not None]
        assert len(collapsed_entries) == collapsed_layers
        assert all(n == expected_retained for n in collapsed_entries)

        # checkpoint weights: retained experts packed (left-shifted) + novices
        novice_keys, gate_up_keys = [], []
        for path in glob.glob(os.path.join(save_dir, "*.safetensors")):
            with safe_open(path, framework="pt") as handle:
                for key in handle.keys():
                    if key.endswith("experts.novices"):
                        novice_keys.append(key)
                    elif key.endswith("experts.gate_up_proj"):
                        gate_up_keys.append(key)

        assert len(novice_keys) == collapsed_layers
        assert len(gate_up_keys) == collapsed_layers
