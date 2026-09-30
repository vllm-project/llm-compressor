# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit and smoke tests for ``ExpertPruner``."""

import json
import re

import pytest
import torch
from compressed_tensors.entrypoints.convert import convert_checkpoint
from compressed_tensors.quantization import QuantizationConfig
from compressed_tensors.utils.moe import extract_expert_index
from compressed_tensors.utils.safetensors_load import (
    get_checkpoint_files,
    get_weight_map,
)
from safetensors import safe_open
from safetensors.torch import save_file

from llmcompressor.entrypoints.converter.expert_pruner import (
    ExpertPruner,
    _compute_retained_experts,
    _compute_retained_experts_nonuniform,
    _magnitude_scores,
    _natural_sort_key,
    _report_scores,
)

# per-expert (2D) tensors, e.g. ``mlp.experts.3.gate_proj.weight``. The default
# ``expert_pattern`` matches stacked (3D) ``gate_up_proj`` / ``down_proj``
EXPERT_PATTERN_2D = r"mlp\.experts\.\d+\.(gate|up|down)_proj"
ROUTER_PATTERN = r"mlp\.gate\.weight$"

# ---------------------------------------------------------------------------
# Synthetic checkpoint / report helpers
# ---------------------------------------------------------------------------


def _write_checkpoint(
    dirpath,
    n_layers=4,
    n_experts=8,
    hidden=8,
    layout="2d",
    num_experts_per_tok=2,
    num_experts_key="num_experts",
    text_config=False,
):
    """Write a tiny synthetic MoE checkpoint (config.json + model.safetensors).

    Every weight of expert ``e`` is filled with ``e`` so tests can check which
    original expert ended up at each position after pruning."""
    tensors = {}
    for layer in range(n_layers):
        prefix = f"model.layers.{layer}.mlp"
        tensors[f"{prefix}.gate.weight"] = torch.randn(n_experts, hidden)
        if layout == "2d":
            for expert in range(n_experts):
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    tensors[f"{prefix}.experts.{expert}.{proj}.weight"] = torch.full(
                        (hidden, hidden), float(expert)
                    )
        else:  # 3d stacked
            expert_ids = torch.arange(n_experts, dtype=torch.float32)
            tensors[f"{prefix}.experts.gate_up_proj"] = expert_ids.view(
                -1, 1, 1
            ).expand(n_experts, 2 * hidden, hidden)
            tensors[f"{prefix}.experts.down_proj"] = expert_ids.view(-1, 1, 1).expand(
                n_experts, hidden, hidden
            )
    # a non-MoE tensor that must pass through untouched
    tensors["lm_head.weight"] = torch.randn(hidden, hidden)
    tensors = {name: tensor.contiguous() for name, tensor in tensors.items()}
    save_file(tensors, str(dirpath / "model.safetensors"))

    experts_cfg = {num_experts_key: n_experts}
    if num_experts_per_tok is not None:
        experts_cfg["num_experts_per_tok"] = num_experts_per_tok
    config = {"text_config": experts_cfg} if text_config else experts_cfg
    (dirpath / "config.json").write_text(json.dumps(config))


def _write_report(path, saliency, count=None):
    count = saliency if count is None else count
    path.write_text(
        json.dumps({"saliency": saliency, "count": count, "topk_weights": saliency})
    )


def _descending_report(path, n_layers, n_experts):
    """Report where expert ``e`` has saliency ``n_experts - e`` (so the highest
    expert indices are the least salient) in every layer."""
    saliency = [
        [float(n_experts - e) for e in range(n_experts)] for _ in range(n_layers)
    ]
    _write_report(path, saliency)
    return saliency


def _load_all(safetensors_path):
    out = {}
    with safe_open(str(safetensors_path), framework="pt") as f:
        for key in f.keys():
            out[key] = f.get_tensor(key)
    return out


def _pruner(path, layout="2d", **kwargs):
    if layout == "2d":
        kwargs.setdefault("expert_pattern", EXPERT_PATTERN_2D)
    return ExpertPruner.from_pretrained(path, **kwargs)


def _checkpoint_meta(path):
    model_files = get_checkpoint_files(path)
    weight_map = get_weight_map(model_files)
    routers = sorted(
        (n for n in weight_map if re.search(ROUTER_PATTERN, n)),
        key=_natural_sort_key,
    )
    return routers, weight_map, model_files


# ---------------------------------------------------------------------------
# Pure-helper unit tests
# ---------------------------------------------------------------------------


def test_natural_sort_key_orders_layers_numerically():
    names = [f"model.layers.{i}.mlp.gate.weight" for i in range(12)]
    scrambled = sorted(names)  # lexicographic: 0,1,10,11,2,...
    assert scrambled != names
    assert sorted(names, key=_natural_sort_key) == names


def test_compute_retained_experts():
    scores = {
        "l0": torch.tensor([4.0, 3.0, 2.0, 1.0]),
        "l1": torch.tensor([1.0, 2.0, 3.0, 4.0]),
    }
    retained = _compute_retained_experts(scores, sparsity=0.5)
    # drop 2 lowest per layer, keep 2 highest, returned sorted
    assert retained == {"l0": [0, 1], "l1": [2, 3]}


def test_compute_retained_experts_keeps_at_least_one():
    scores = {"l0": torch.tensor([1.0, 3.0, 2.0])}
    assert _compute_retained_experts(scores, sparsity=1.0) == {"l0": [1]}


def test_compute_retained_experts_nonuniform_global_budget():
    # layer "low" is globally least salient, so the budget is spent there first
    scores = {
        "low": torch.tensor([0.4, 0.1, 0.3, 0.2]),
        "high": torch.tensor([10.0, 11.0, 0.25, 13.0]),
    }
    retained = _compute_retained_experts_nonuniform(scores, sparsity=0.375)
    # budget = round(0.375 * 8) = 3: low[1], low[3], high[2]
    assert retained == {"low": [0, 2], "high": [0, 1, 3]}


def test_compute_retained_experts_nonuniform_respects_floor():
    scores = {
        "low": torch.tensor([0.1, 0.2, 0.3, 0.4]),
        "high": torch.tensor([10.0, 11.0, 12.0, 13.0]),
    }
    retained = _compute_retained_experts_nonuniform(scores, sparsity=0.5, floor=3)
    # budget = 4, but each layer can lose at most one expert
    assert retained == {"low": [1, 2, 3], "high": [1, 2, 3]}


def test_magnitude_scores_are_l1_of_router_rows(tmp_path):
    _write_checkpoint(tmp_path, n_layers=2, n_experts=4)
    routers, weight_map, model_files = _checkpoint_meta(tmp_path)
    scores = _magnitude_scores(routers, weight_map, model_files)

    tensors = _load_all(tmp_path / "model.safetensors")
    assert list(scores) == routers
    for router in routers:
        expected = tensors[router].abs().sum(dim=-1).to(torch.float64)
        assert torch.allclose(scores[router], expected)


@pytest.mark.parametrize("metric", ["saliency", "count"])
def test_report_scores_reads_metric(tmp_path, metric):
    _write_checkpoint(tmp_path, n_layers=2, n_experts=3)
    saliency = [[2.0, 1.0, 0.0], [4.0, 2.0, 2.0]]
    count = [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]
    report = tmp_path / "r.json"
    _write_report(report, saliency, count)

    routers, weight_map, model_files = _checkpoint_meta(tmp_path)
    scores = _report_scores(routers, weight_map, model_files, report, metric)
    expected = saliency if metric == "saliency" else count
    assert [scores[r].tolist() for r in routers] == expected


def test_report_scores_layerwise_saliency_divides_by_layer_max(tmp_path):
    _write_checkpoint(tmp_path, n_layers=3, n_experts=3)
    report = tmp_path / "r.json"
    # the all-zero layer must not divide by zero; its scores are left untouched
    _write_report(report, [[2.0, 1.0, 0.0], [4.0, 2.0, 2.0], [0.0, 0.0, 0.0]])

    routers, weight_map, model_files = _checkpoint_meta(tmp_path)
    scores = _report_scores(
        routers, weight_map, model_files, report, "layerwise_saliency"
    )
    assert [scores[r].tolist() for r in routers] == [
        [1.0, 0.5, 0.0],
        [1.0, 0.5, 0.5],
        [0.0, 0.0, 0.0],
    ]


def test_report_scores_missing_metric_raises(tmp_path):
    _write_checkpoint(tmp_path, n_layers=1, n_experts=3)
    report = tmp_path / "r.json"
    report.write_text(json.dumps({"saliency": [[1.0, 2.0, 3.0]]}))

    routers, weight_map, model_files = _checkpoint_meta(tmp_path)
    with pytest.raises(ValueError, match="does not contain 'count'"):
        _report_scores(routers, weight_map, model_files, report, "count")


# ---------------------------------------------------------------------------
# from_pretrained validation
# ---------------------------------------------------------------------------


def test_init_directs_to_from_pretrained():
    with pytest.raises(ValueError, match="from_pretrained"):
        ExpertPruner()


def test_from_pretrained_rejects_bad_metric(tmp_path):
    _write_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="metric must be one of"):
        _pruner(tmp_path, sparsity=0.25, metric="bogus")


@pytest.mark.parametrize("metric", ["saliency", "layerwise_saliency", "count"])
def test_from_pretrained_report_metric_requires_report(tmp_path, metric):
    _write_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="requires a saliency_report_path"):
        _pruner(tmp_path, sparsity=0.25, metric=metric)


@pytest.mark.parametrize("sparsity", [1.5, -0.1])
def test_from_pretrained_rejects_bad_sparsity(tmp_path, sparsity):
    _write_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="sparsity must be"):
        _pruner(tmp_path, sparsity=sparsity, metric="magnitude")


def test_from_pretrained_rejects_layer_count_mismatch(tmp_path):
    _write_checkpoint(tmp_path, n_layers=4, n_experts=8)
    report = tmp_path / "r.json"
    _descending_report(report, 3, 8)  # 3 layers vs 4 routers
    with pytest.raises(ValueError, match="router weight"):
        _pruner(tmp_path, sparsity=0.25, metric="saliency", saliency_report_path=report)


def test_from_pretrained_rejects_expert_count_mismatch(tmp_path):
    _write_checkpoint(tmp_path, n_layers=4, n_experts=8)
    report = tmp_path / "r.json"
    _descending_report(report, 4, 6)  # 6 experts in report vs 8 in checkpoint
    with pytest.raises(ValueError, match="experts"):
        _pruner(tmp_path, sparsity=0.25, metric="saliency", saliency_report_path=report)


def test_from_pretrained_rejects_pruning_below_routing_floor(tmp_path):
    # num_experts_per_tok=6, 8 experts, sparsity 0.5 would keep 4, but the
    # router still needs to select 6 experts per token
    _write_checkpoint(tmp_path, n_experts=8, num_experts_per_tok=6)
    report = tmp_path / "r.json"
    _descending_report(report, 4, 8)
    with pytest.raises(ValueError, match="routing floor"):
        _pruner(tmp_path, sparsity=0.5, metric="saliency", saliency_report_path=report)


def test_from_pretrained_without_num_experts_per_tok(tmp_path):
    # without num_experts_per_tok there is no routing floor to check against
    _write_checkpoint(tmp_path, n_experts=8, num_experts_per_tok=None)
    report = tmp_path / "r.json"
    _descending_report(report, 4, 8)
    pruner = _pruner(
        tmp_path, sparsity=0.75, metric="saliency", saliency_report_path=report
    )
    assert {len(v) for v in pruner.retained_experts.values()} == {2}


def test_from_pretrained_no_routers_or_experts_raises(tmp_path):
    _write_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="router_pattern"):
        _pruner(tmp_path, sparsity=0.25, metric="magnitude", router_pattern="nope$")
    with pytest.raises(ValueError, match="expert_pattern"):
        _pruner(tmp_path, sparsity=0.25, metric="magnitude", expert_pattern="nope")


# ---------------------------------------------------------------------------
# Selection + config, 2D and 3D layouts
# ---------------------------------------------------------------------------


def test_natural_ordering_aligns_report_to_layers(tmp_path):
    # 12 layers; reverse the ranking of layer 10 only. If the report were aligned
    # to routers lexicographically (0,1,10,11,2,...), report[10] would land on
    # layer 8 instead
    _write_checkpoint(tmp_path, n_layers=12, n_experts=8)
    saliency = [[float(8 - e) for e in range(8)] for _ in range(12)]
    saliency[10] = [float(e) for e in range(8)]
    report = tmp_path / "r.json"
    _write_report(report, saliency)

    pruner = _pruner(
        tmp_path, sparsity=0.5, metric="saliency", saliency_report_path=report
    )
    for layer in range(12):
        retained = pruner.retained_experts[f"model.layers.{layer}.mlp.gate.weight"]
        expected = [4, 5, 6, 7] if layer == 10 else [0, 1, 2, 3]
        assert retained == expected


def test_metrics_select_different_experts(tmp_path):
    _write_checkpoint(tmp_path, n_layers=1, n_experts=4)
    report = tmp_path / "r.json"
    _write_report(report, saliency=[[4.0, 3.0, 2.0, 1.0]], count=[[1, 2, 3, 4]])

    kwargs = dict(sparsity=0.5, saliency_report_path=report)
    by_saliency = _pruner(tmp_path, metric="saliency", **kwargs)
    by_count = _pruner(tmp_path, metric="count", **kwargs)
    router = "model.layers.0.mlp.gate.weight"
    assert by_saliency.retained_experts[router] == [0, 1]
    assert by_count.retained_experts[router] == [2, 3]


def test_magnitude_metric_matches_router_weights(tmp_path):
    _write_checkpoint(tmp_path, n_layers=2, n_experts=8)
    pruner = _pruner(tmp_path, sparsity=0.25, metric="magnitude")

    tensors = _load_all(tmp_path / "model.safetensors")
    for router, retained in pruner.retained_experts.items():
        scores = tensors[router].abs().sum(dim=-1)
        assert retained == sorted(torch.topk(scores, 6).indices.tolist())


@pytest.mark.parametrize("layout", ["2d", "3d"])
@pytest.mark.parametrize("metric", ["saliency", "layerwise_saliency", "count"])
def test_process_and_config(tmp_path, layout, metric):
    _write_checkpoint(tmp_path, n_layers=3, n_experts=8, layout=layout)
    report = tmp_path / "r.json"
    _descending_report(report, 3, 8)

    pruner = _pruner(
        tmp_path,
        layout=layout,
        sparsity=0.25,
        metric=metric,
        saliency_report_path=report,
    )
    assert pruner.is_3d is (layout == "3d")
    # 8 experts, drop round(0.25*8)=2 -> keep 6, every layer the same
    # descending saliency -> highest-index experts pruned -> keep 0..5
    assert all(v == list(range(6)) for v in pruner.retained_experts.values())

    result = pruner.validate(_load_all(tmp_path / "model.safetensors"))

    # non-MoE tensor passes through
    assert "lm_head.weight" in result
    # router weights sliced to 6 rows
    for name, tensor in result.items():
        if name.endswith("mlp.gate.weight"):
            assert tensor.shape[0] == 6
    if layout == "3d":
        for name, tensor in result.items():
            if ".experts." in name:
                assert tensor.shape[0] == 6
                assert tensor[:, 0, 0].tolist() == list(range(6))
    else:
        # exactly 6 experts * 3 projs per layer, indices renumbered to 0..5
        for layer in range(3):
            names = [
                n for n in result if n.startswith(f"model.layers.{layer}.mlp.experts.")
            ]
            assert len(names) == 6 * 3
            assert {extract_expert_index(n) for n in names} == set(range(6))

    config = json.loads((tmp_path / "config.json").read_text())
    updated = pruner.update_model_config(dict(config))
    assert updated["num_experts"] == 6


def test_process_renumbers_retained_experts(tmp_path):
    # keep experts 1, 4, 6, 7 -> they must become experts 0, 1, 2, 3
    _write_checkpoint(tmp_path, n_layers=1, n_experts=8)
    report = tmp_path / "r.json"
    _write_report(report, [[0.0, 5.0, 0.1, 0.2, 6.0, 0.3, 7.0, 8.0]])

    pruner = _pruner(
        tmp_path, sparsity=0.5, metric="saliency", saliency_report_path=report
    )
    router = "model.layers.0.mlp.gate.weight"
    assert pruner.retained_experts[router] == [1, 4, 6, 7]

    tensors = _load_all(tmp_path / "model.safetensors")
    result = pruner.validate(tensors)
    for new_idx, old_idx in enumerate([1, 4, 6, 7]):
        name = f"model.layers.0.mlp.experts.{new_idx}.up_proj.weight"
        assert result[name][0, 0].item() == old_idx
    assert torch.equal(result[router], tensors[router][[1, 4, 6, 7]])


def test_config_key_in_text_config(tmp_path):
    _write_checkpoint(
        tmp_path, n_experts=8, num_experts_key="num_local_experts", text_config=True
    )
    report = tmp_path / "r.json"
    _descending_report(report, 4, 8)
    pruner = _pruner(
        tmp_path, sparsity=0.25, metric="saliency", saliency_report_path=report
    )
    assert pruner.num_experts_config_key == "num_local_experts"
    updated = pruner.update_model_config(
        {"text_config": {"num_local_experts": 8, "num_experts_per_tok": 2}}
    )
    assert updated["text_config"]["num_local_experts"] == 6


@pytest.mark.parametrize("layout", ["2d", "3d"])
def test_convert_checkpoint(tmp_path, layout):
    src = tmp_path / "src"
    src.mkdir()
    _write_checkpoint(src, n_layers=2, n_experts=8, layout=layout)
    report = tmp_path / "r.json"
    _descending_report(report, 2, 8)

    pruner = _pruner(
        src,
        layout=layout,
        sparsity=0.25,
        metric="saliency",
        saliency_report_path=report,
    )
    save_dir = tmp_path / "pruned"
    convert_checkpoint(model_stub=src, save_directory=save_dir, converter=pruner)

    config = json.loads((save_dir / "config.json").read_text())
    assert config["num_experts"] == 6
    out = _load_all(save_dir / "model.safetensors")
    for layer in range(2):
        assert out[f"model.layers.{layer}.mlp.gate.weight"].shape[0] == 6
    if layout == "2d":
        assert max(extract_expert_index(n) for n in out if ".experts." in n) == 5


# ---------------------------------------------------------------------------
# Non-uniform pruning
# ---------------------------------------------------------------------------


def _skewed_report(path, n_layers, n_experts, low_layer):
    """Descending saliency in every layer, with ``low_layer`` scaled down so its
    experts are globally the least salient."""
    saliency = [
        [float(n_experts - e) for e in range(n_experts)] for _ in range(n_layers)
    ]
    saliency[low_layer] = [0.01 * value for value in saliency[low_layer]]
    _write_report(path, saliency)
    return saliency


def test_nonuniform_prunes_globally_lowest_experts(tmp_path):
    # 12 layers; layer 10 is by far the least salient, so it is pruned down to
    # the routing floor. With lexicographic alignment it would land on layer 8
    _write_checkpoint(tmp_path, n_layers=12, n_experts=8, num_experts_per_tok=2)
    report = tmp_path / "r.json"
    _skewed_report(report, 12, 8, low_layer=10)

    pruner = _pruner(
        tmp_path,
        sparsity=0.25,
        metric="saliency",
        uniform=False,
        saliency_report_path=report,
    )
    counts = [
        len(pruner.retained_experts[f"model.layers.{layer}.mlp.gate.weight"])
        for layer in range(12)
    ]
    # budget = round(0.25 * 96) = 24 pruned in total
    assert sum(counts) == 96 - 24
    assert counts[10] == 2
    assert pruner.retained_experts["model.layers.10.mlp.gate.weight"] == [0, 1]


def test_nonuniform_layerwise_saliency_normalizes_layers(tmp_path):
    # after normalizing by each layer's max, the scaled-down layer is no longer
    # singled out, so every layer is pruned equally
    _write_checkpoint(tmp_path, n_layers=4, n_experts=8)
    report = tmp_path / "r.json"
    _skewed_report(report, 4, 8, low_layer=1)

    kwargs = dict(sparsity=0.25, uniform=False, saliency_report_path=report)
    raw = _pruner(tmp_path, metric="saliency", **kwargs)
    normalized = _pruner(tmp_path, metric="layerwise_saliency", **kwargs)
    assert len(set(map(len, raw.retained_experts.values()))) > 1
    assert {len(v) for v in normalized.retained_experts.values()} == {6}


def _nonuniform_pruner(path, layout="2d", n_layers=3):
    report = path / "r.json"
    _skewed_report(report, n_layers, 8, low_layer=1)
    return _pruner(
        path,
        layout=layout,
        sparsity=0.25,
        metric="saliency",
        uniform=False,
        saliency_report_path=report,
    )


def test_nonuniform_update_config_creates_layer_overrides(tmp_path):
    _write_checkpoint(tmp_path, n_layers=3, n_experts=8)
    pruner = _nonuniform_pruner(tmp_path)
    # budget = round(0.25 * 24) = 6, all spent on the least salient layer 1
    assert [len(v) for v in pruner.retained_experts.values()] == [8, 2, 8]

    config = pruner.update_config(None)
    assert config.config_groups == {}
    assert config.format == "dense"
    assert config.layer_overrides == {"num_experts": [8, 2, 8]}

    # num_experts in the model config is left untouched
    updated = pruner.update_model_config({"num_experts": 8})
    assert updated["num_experts"] == 8


def test_nonuniform_update_config_extends_existing_config(tmp_path):
    _write_checkpoint(tmp_path, n_layers=3, n_experts=8)
    pruner = _nonuniform_pruner(tmp_path)

    existing = QuantizationConfig(
        config_groups={"FP8": ["Linear"]}, format="float-quantized"
    )
    config = pruner.update_config(existing)
    # existing config is preserved and extended, not replaced or mutated
    assert config.format == "float-quantized"
    assert "FP8" in config.config_groups
    assert config.layer_overrides == {"num_experts": [8, 2, 8]}
    assert existing.layer_overrides == {}


def test_uniform_update_config_passes_through(tmp_path):
    _write_checkpoint(tmp_path, n_layers=3, n_experts=8)
    report = tmp_path / "r.json"
    _descending_report(report, 3, 8)
    pruner = _pruner(
        tmp_path, sparsity=0.25, metric="saliency", saliency_report_path=report
    )
    assert pruner.update_config(None) is None


@pytest.mark.parametrize("layout", ["2d", "3d"])
def test_nonuniform_convert_checkpoint(tmp_path, layout):
    src = tmp_path / "src"
    src.mkdir()
    _write_checkpoint(src, n_layers=3, n_experts=8, layout=layout)
    pruner = _nonuniform_pruner(src, layout=layout)
    expected_counts = [8, 2, 8]

    save_dir = tmp_path / "pruned"
    convert_checkpoint(model_stub=src, save_directory=save_dir, converter=pruner)

    config = json.loads((save_dir / "config.json").read_text())
    assert config["num_experts"] == 8
    assert config["quantization_config"]["layer_overrides"] == {
        "num_experts": expected_counts
    }

    out = _load_all(save_dir / "model.safetensors")
    for layer, count in enumerate(expected_counts):
        prefix = f"model.layers.{layer}.mlp"
        assert out[f"{prefix}.gate.weight"].shape[0] == count
        expert_names = [n for n in out if n.startswith(f"{prefix}.experts.")]
        if layout == "3d":
            assert all(out[n].shape[0] == count for n in expert_names)
        else:
            assert {extract_expert_index(n) for n in expert_names} == set(range(count))


# ---------------------------------------------------------------------------
# Smoke tests against real (small) MoE checkpoints
# ---------------------------------------------------------------------------

# (stub, expert_pattern, expected is_3d) — Qwen3-1.6B has 2D per-expert weights,
# Qwen3-VL has 3D stacked/fused expert weights and keeps its expert count under
# text_config
SMOKE_MODELS = [
    ("inference-optimization/Qwen3-1.6B-A0.9B", EXPERT_PATTERN_2D, False),
    (
        "inference-optimization/Qwen3-VL-1.0B-A0.4B-Instruct",
        r"mlp\.experts\.(gate_up_proj|down_proj)",
        True,
    ),
]


def _build_matching_report(model_stub, report_path):
    """Build a correctly-shaped descending-saliency report for a real checkpoint
    by reading its router layout, so no real calibration run is needed."""
    routers, weight_map, model_files = _checkpoint_meta(model_stub)
    num_experts = {}
    for router in routers:
        with safe_open(model_files[weight_map[router]], framework="pt") as f:
            num_experts[router] = f.get_slice(router).get_shape()[0]
    saliency = [
        [float(num_experts[r] - e) for e in range(num_experts[r])] for r in routers
    ]
    _write_report(report_path, saliency)
    return routers, num_experts


@pytest.mark.smoke
@pytest.mark.integration
@pytest.mark.parametrize("model_stub, expert_pattern, expected_is_3d", SMOKE_MODELS)
def test_smoke_from_pretrained(model_stub, expert_pattern, expected_is_3d, tmp_path):
    report_path = tmp_path / "report.json"
    routers, num_experts = _build_matching_report(model_stub, report_path)
    assert routers, "no routers found in checkpoint"

    pruner = ExpertPruner.from_pretrained(
        model_stub,
        sparsity=0.25,
        metric="saliency",
        expert_pattern=expert_pattern,
        saliency_report_path=report_path,
    )
    assert pruner.is_3d is expected_is_3d
    assert set(pruner.retained_experts) == set(routers)
    for router in routers:
        n = num_experts[router]
        assert pruner.retained_experts[router] == list(range(n - round(0.25 * n)))


@pytest.mark.smoke
@pytest.mark.integration
@pytest.mark.parametrize("model_stub, expert_pattern, expected_is_3d", SMOKE_MODELS)
def test_smoke_convert_checkpoint(model_stub, expert_pattern, expected_is_3d, tmp_path):
    report_path = tmp_path / "report.json"
    _, num_experts = _build_matching_report(model_stub, report_path)
    n_experts = next(iter(num_experts.values()))
    expected_kept = n_experts - round(0.25 * n_experts)

    pruner = ExpertPruner.from_pretrained(
        model_stub,
        sparsity=0.25,
        metric="saliency",
        expert_pattern=expert_pattern,
        saliency_report_path=report_path,
    )
    key = pruner.num_experts_config_key

    save_dir = tmp_path / "pruned"
    convert_checkpoint(model_stub=model_stub, save_directory=save_dir, converter=pruner)

    config = json.loads((save_dir / "config.json").read_text())
    updated = config.get(key)
    if updated is None:
        updated = config.get("text_config", {}).get(key)
    assert updated == expected_kept

    # every router weight in the output is sliced to the retained expert count
    out_routers, out_map, out_files = _checkpoint_meta(save_dir)
    assert out_routers
    for name in out_routers:
        with safe_open(out_files[out_map[name]], framework="pt") as f:
            assert f.get_slice(name).get_shape()[0] == expected_kept

    # experts are correspondingly reduced
    expert_names = [n for n in out_map if re.search(expert_pattern, n)]
    assert expert_names
    if expected_is_3d:
        for name in expert_names:
            with safe_open(out_files[out_map[name]], framework="pt") as f:
                assert f.get_slice(name).get_shape()[0] == expected_kept
    else:
        indices = {extract_expert_index(n) for n in expert_names}
        assert indices == set(range(expected_kept))
