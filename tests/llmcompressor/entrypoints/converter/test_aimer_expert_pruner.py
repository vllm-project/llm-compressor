# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for ``AIMERExpertPruner``."""

import json

import pytest
import torch
from compressed_tensors.entrypoints.convert import convert_checkpoint
from compressed_tensors.utils.safetensors_load import (
    get_checkpoint_files,
    get_weight_map,
)
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from llmcompressor.entrypoints.converter import AIMERExpertPruner
from llmcompressor.entrypoints.converter.aimer_expert_pruner import _aimer_scores

EXPERT_PATTERN_2D = r"mlp\.experts\.\d+\.(gate|up|down)_proj"
HIDDEN = 8
INTERMEDIATE = 16


def _expert(scale: float, constant_magnitude: bool = False) -> dict[str, torch.Tensor]:
    """Random gate/up/down weights of one expert. With ``constant_magnitude``
    every weight is ``+-scale``, the maximum AIMER score of 1."""
    shapes = {
        "gate_proj": (INTERMEDIATE, HIDDEN),
        "up_proj": (INTERMEDIATE, HIDDEN),
        "down_proj": (HIDDEN, INTERMEDIATE),
    }
    weights = {}
    for proj, shape in shapes.items():
        weight = torch.randn(shape)
        if constant_magnitude:
            weight = weight.sign()
        weights[proj] = scale * weight
    return weights


def _write_checkpoint(dirpath, experts_by_layer, layout="2d", dtype=torch.float32):
    """Write a synthetic MoE checkpoint from per-layer lists of expert weights.

    Every expert also carries a large down_proj bias and a floating-point
    auxiliary tensor (a weight scale), which must not affect its score."""
    tensors = {}
    for layer, experts in enumerate(experts_by_layer):
        prefix = f"model.layers.{layer}.mlp"
        tensors[f"{prefix}.gate.weight"] = torch.randn(len(experts), HIDDEN)
        if layout == "2d":
            for e, weights in enumerate(experts):
                for proj, weight in weights.items():
                    tensors[f"{prefix}.experts.{e}.{proj}.weight"] = weight
                down = f"{prefix}.experts.{e}.down_proj"
                tensors[f"{down}.bias"] = torch.full((HIDDEN,), 100.0)
                tensors[f"{down}.weight_scale"] = torch.full((HIDDEN, 1), 100.0)
        else:  # 3d stacked, gate and up concatenated
            tensors[f"{prefix}.experts.gate_up_proj"] = torch.stack(
                [torch.cat([w["gate_proj"], w["up_proj"]]) for w in experts]
            )
            tensors[f"{prefix}.experts.down_proj"] = torch.stack(
                [w["down_proj"] for w in experts]
            )
            for aux in ("down_proj_bias", "down_proj_scale"):
                tensors[f"{prefix}.experts.{aux}"] = torch.full(
                    (len(experts), HIDDEN), 100.0
                )
    tensors = {name: t.to(dtype).contiguous() for name, t in tensors.items()}
    save_file(tensors, str(dirpath / "model.safetensors"))

    config = {"num_experts": len(experts_by_layer[0]), "num_experts_per_tok": 2}
    (dirpath / "config.json").write_text(json.dumps(config))


def _pruner(path, layout, **kwargs) -> AIMERExpertPruner:
    if layout == "2d":
        kwargs.setdefault("expert_pattern", EXPERT_PATTERN_2D)
    return AIMERExpertPruner.from_pretrained(path, **kwargs)


def _scores(path, pruner: AIMERExpertPruner) -> dict[str, torch.Tensor]:
    model_files = get_checkpoint_files(str(path))
    return _aimer_scores(
        list(pruner.retained_experts),
        pruner.expert_to_router,
        pruner.expert_indices,
        pruner.is_3d,
        get_weight_map(model_files),
        model_files,
    )


def _reference_aimer(weights: dict[str, torch.Tensor]) -> float:
    """AIMER of one expert, as defined in the paper: over the flattened and
    concatenated projection weights, ``||w||_1 / (sqrt(N) * ||w||_2)``."""
    w = torch.cat([weight.flatten() for weight in weights.values()]).double()
    return (w.abs().sum() / (w.numel() ** 0.5 * w.norm())).item()


@pytest.mark.parametrize("layout", ["2d", "3d"])
def test_scores_match_reference(tmp_path, layout):
    """Experts are scored by their negated AIMER score, independent of the
    checkpoint layout, and biases and auxiliary tensors are not scored."""
    torch.manual_seed(0)
    experts_by_layer = [
        [_expert(scale=0.1 * (e + 1)) for e in range(8)] for _ in range(2)
    ]
    _write_checkpoint(tmp_path, experts_by_layer, layout)

    scores = _scores(tmp_path, _pruner(tmp_path, layout, sparsity=0.25))

    for layer, experts in enumerate(experts_by_layer):
        expected = [-_reference_aimer(weights) for weights in experts]
        actual = scores[f"model.layers.{layer}.mlp.gate.weight"]
        assert actual.tolist() == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("layout", ["2d", "3d"])
def test_convert_checkpoint_prunes_highest_aimer_experts(tmp_path, layout):
    """The highest-AIMER experts are pruned, regardless of their magnitude:
    experts 1 and 6 have constant-magnitude weights (AIMER = 1) and the
    smallest scale, while the others have Gaussian weights (AIMER ~ 0.8)."""
    torch.manual_seed(0)
    pruned_experts = {1, 6}
    experts_by_layer = [
        [
            _expert(scale=0.01, constant_magnitude=True)
            if e in pruned_experts
            else _expert(scale=1.0)
            for e in range(8)
        ]
        for _ in range(2)
    ]
    src = tmp_path / "src"
    src.mkdir()
    _write_checkpoint(src, experts_by_layer, layout)

    pruner = _pruner(src, layout, sparsity=0.25)
    retained = [e for e in range(8) if e not in pruned_experts]
    assert all(r == retained for r in pruner.retained_experts.values())

    dst = tmp_path / "dst"
    convert_checkpoint(model_stub=str(src), save_directory=dst, converter=pruner)

    config = json.loads((dst / "config.json").read_text())
    assert config["num_experts"] == len(retained)
    with safe_open(str(dst / "model.safetensors"), framework="pt") as f:
        for layer in range(2):
            prefix = f"model.layers.{layer}.mlp.experts"
            if layout == "2d":
                down = [
                    f.get_tensor(f"{prefix}.{i}.down_proj.weight")
                    for i in range(len(retained))
                ]
            else:
                down = list(f.get_tensor(f"{prefix}.down_proj"))
            for new, old in enumerate(retained):
                original = experts_by_layer[layer][old]["down_proj"]
                assert torch.equal(down[new], original)


def test_rejects_quantized_weights(tmp_path):
    torch.manual_seed(0)
    experts_by_layer = [[_expert(scale=1.0) for _ in range(8)]]
    _write_checkpoint(tmp_path, experts_by_layer, "2d", dtype=torch.int8)

    with pytest.raises(ValueError, match="every matched expert tensor"):
        _pruner(tmp_path, "2d", sparsity=0.25)


def test_rejects_packed_projection_next_to_float_ones(tmp_path):
    """A packed down_proj next to floating-point gate/up weights is rejected,
    rather than left out of the score."""
    torch.manual_seed(0)
    _write_checkpoint(tmp_path, [[_expert(scale=1.0) for _ in range(8)]])
    path = str(tmp_path / "model.safetensors")
    tensors = load_file(path)
    for name in [n for n in tensors if n.endswith("down_proj.weight")]:
        packed = name.removesuffix(".weight") + ".weight_packed"
        tensors[packed] = tensors.pop(name).to(torch.int32)
    save_file(tensors, path)

    with pytest.raises(ValueError, match="weight_packed has dtype I32"):
        _pruner(tmp_path, "2d", sparsity=0.25)


def test_all_zero_expert_is_retained(tmp_path):
    """An all-zero expert's 0/0 score is 0, as in the reference implementation,
    which is above every valid (negated) score, so the expert is retained."""
    torch.manual_seed(0)
    experts = [_expert(scale=1.0) for _ in range(8)]
    experts[3] = {proj: torch.zeros_like(w) for proj, w in experts[3].items()}
    _write_checkpoint(tmp_path, [experts])

    pruner = _pruner(tmp_path, "2d", sparsity=0.25)
    router = "model.layers.0.mlp.gate.weight"
    assert _scores(tmp_path, pruner)[router][3].item() == 0.0
    assert 3 in pruner.retained_experts[router]


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_rejects_non_finite_weights(tmp_path, value):
    torch.manual_seed(0)
    experts = [_expert(scale=1.0) for _ in range(8)]
    experts[3]["down_proj"][0, 0] = value
    _write_checkpoint(tmp_path, [experts])

    with pytest.raises(ValueError, match=r"experts \[3\] have non-finite weights"):
        _pruner(tmp_path, "2d", sparsity=0.25)
