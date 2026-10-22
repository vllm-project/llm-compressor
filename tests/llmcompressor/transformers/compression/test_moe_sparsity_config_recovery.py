"""
Regression tests for recovering the quantization config when saving a pre-quantized MoE
model whose experts were collapsed (MoNe) / pruned (REAP) and repacked into fused
compressed modules.

On the quantized-forward save path (``layerwise_decompression=False``) the fused experts
(``CompressedFusedLinear``) carry no ``quantization_scheme``, so
``ModelCompressor.from_pretrained_model`` infers no config. Previously a degenerate empty
config was serialized (``format="fakequant"``, empty ``config_groups``), producing an
unloadable checkpoint. These tests cover the retargeting that rebuilds the config from the
original checkpoint's scheme against the fused layout.
"""

import re

import pytest

from llmcompressor.transformers.compression.compressed_tensors_utils import (
    _as_quantization_config_dict,
    _retarget_expert_groups_to_fused,
)


def _nvfp4_group(target: str) -> dict:
    return {
        "format": "nvfp4-pack-quantized",
        "targets": [target],
        "weights": {"num_bits": 4, "type": "float", "group_size": 16},
    }


def test_retarget_single_group_matches_all_fused_experts():
    # per-expert target, as stored in a pre-collapse nvfp4 checkpoint
    groups = {"group_0": _nvfp4_group(r"re:.*mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$")}
    fused = [
        f"model.layers.{i}.mlp.experts.{proj}"
        for i in range(3, 78)
        for proj in ("gate_up_proj", "down_proj")
    ]

    kept = _retarget_expert_groups_to_fused(groups, fused)

    assert list(kept) == ["group_0"]
    (target,) = kept["group_0"]["targets"]
    pattern = re.compile(target[len("re:") :])
    assert all(pattern.search(name) for name in fused)
    # original per-expert form must be gone (no `experts.<idx>.`)
    assert r"\d+" not in target


def test_retarget_drops_group_whose_layers_are_absent():
    # mixed-precision: nvfp4 on layers 3-44, fp8 on layer 45 (an MTP layer that
    # transformers does not rebuild, so no layer-45 module exists in the output)
    groups = {
        "group_0": _nvfp4_group(
            r"re:.*\.layers\.(?:[3-9]|[1-3][0-9]|4[0-4])\.mlp\.experts\..*(gate|up|down)_proj$"
        ),
        "group_1": {
            "format": "float-quantized",
            "targets": [r"re:.*\.layers\.45\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$"],
            "weights": {"num_bits": 8},
        },
    }
    fused = [
        f"model.language_model.layers.{i}.mlp.experts.{proj}"
        for i in range(3, 45)
        for proj in ("gate_up_proj", "down_proj")
    ]

    kept = _retarget_expert_groups_to_fused(groups, fused)

    # layer-45 (fp8) group targets nothing in the fused output -> dropped
    assert list(kept) == ["group_0"]
    pattern = re.compile(kept["group_0"]["targets"][0][len("re:") :])
    assert all(pattern.search(name) for name in fused)


def test_retarget_leaves_non_expert_targets_untouched():
    groups = {"group_0": {"format": "x", "targets": ["re:.*self_attn\\.q_proj$", "Linear"]}}
    kept = _retarget_expert_groups_to_fused(
        groups, ["model.layers.0.mlp.experts.down_proj"]
    )
    # nothing matched the fused experts, so the group is dropped, but non-expert targets
    # must not have been corrupted in the attempt
    assert kept == {}


@pytest.mark.parametrize(
    "raw,expected",
    [
        ({"config_groups": {"g": 1}}, {"config_groups": {"g": 1}}),
        ({"quantization_config": {"config_groups": {"g": 2}}}, {"config_groups": {"g": 2}}),
        (None, None),
        ({"no_groups": True}, None),
    ],
)
def test_as_quantization_config_dict(raw, expected):
    assert _as_quantization_config_dict(raw) == expected


def test_as_quantization_config_dict_unwraps_to_dict_wrapper():
    class Wrapper:
        def to_dict(self):
            return {"quantization_config": {"config_groups": {"g": 3}}}

    assert _as_quantization_config_dict(Wrapper()) == {"config_groups": {"g": 3}}
