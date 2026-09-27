# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit and smoke tests for ``REAPExpertPruner``."""

import json

import pytest
import torch
from compressed_tensors.base import QUANTIZATION_CONFIG_NAME
from compressed_tensors.entrypoints.convert import convert_checkpoint
from compressed_tensors.utils.safetensors_load import (
    get_checkpoint_files,
    get_weight_map,
)
from safetensors import safe_open
from safetensors.torch import save_file

from llmcompressor.entrypoints.converter.reap import (
    NUM_EXPERTS_PER_LAYER_KEY,
    REAPExpertPruner,
    _compute_retained_experts,
    _is_router_weight,
    _looks_like_expert,
    _metric_values,
    _natural_sort_key,
    _num_experts_by_router,
)

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
    """Write a tiny synthetic MoE checkpoint (config.json + model.safetensors)."""
    tensors = {}
    for layer in range(n_layers):
        prefix = f"model.layers.{layer}.mlp"
        tensors[f"{prefix}.gate.weight"] = torch.randn(n_experts, hidden)
        if layout == "2d":
            for expert in range(n_experts):
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    tensors[f"{prefix}.experts.{expert}.{proj}.weight"] = torch.randn(
                        hidden, hidden
                    )
        else:  # 3d stacked
            for proj in ("gate_proj", "up_proj", "down_proj"):
                tensors[f"{prefix}.experts.{proj}.weight"] = torch.randn(
                    n_experts, hidden, hidden
                )
    # a non-MoE tensor that must pass through untouched
    tensors["lm_head.weight"] = torch.randn(hidden, hidden)
    save_file(tensors, str(dirpath / "model.safetensors"))

    experts_cfg = {
        num_experts_key: n_experts,
        "num_experts_per_tok": num_experts_per_tok,
    }
    config = {"text_config": experts_cfg} if text_config else experts_cfg
    (dirpath / "config.json").write_text(json.dumps(config))


def _write_report(path, saliency):
    path.write_text(
        json.dumps({"saliency": saliency, "count": saliency, "topk_weights": saliency})
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


# ---------------------------------------------------------------------------
# Pure-helper unit tests
# ---------------------------------------------------------------------------


def test_natural_sort_key_orders_layers_numerically():
    names = [f"model.layers.{i}.mlp.gate.weight" for i in range(12)]
    scrambled = sorted(names)  # lexicographic: 0,1,10,11,2,...
    assert scrambled != names
    assert sorted(names, key=_natural_sort_key) == names


def test_is_router_weight():
    assert _is_router_weight("model.layers.0.mlp.gate.weight")
    assert _is_router_weight("model.layers.0.mlp.router.weight")
    # gate_proj is an expert projection, not a router
    assert not _is_router_weight("model.layers.0.mlp.experts.3.gate_proj.weight")
    # router bias is not the weight matrix
    assert not _is_router_weight("model.layers.0.mlp.gate.e_score_correction_bias")


def test_looks_like_expert():
    assert _looks_like_expert("model.layers.0.mlp.experts.gate_up_proj")
    assert _looks_like_expert("model.layers.0.mlp.experts.3.down_proj.weight")
    assert not _looks_like_expert("model.layers.0.mlp.gate.weight")


def test_metric_values_saliency_norm_divides_by_layer_max():
    report = {"saliency": [[2.0, 1.0, 0.0], [4.0, 2.0, 2.0]], "count": [[1, 2, 3]] * 2}
    norm = _metric_values(report, "saliency_norm", "r.json")
    assert norm == [[1.0, 0.5, 0.0], [1.0, 0.5, 0.5]]


def test_metric_values_saliency_norm_handles_all_zero_layer():
    report = {"saliency": [[0.0, 0.0, 0.0]]}
    # must not divide by zero; leave the (all-zero) scores untouched
    assert _metric_values(report, "saliency_norm", "r.json") == [[0.0, 0.0, 0.0]]


def test_metric_values_missing_metric_raises():
    with pytest.raises(ValueError, match="does not contain metric"):
        _metric_values({"saliency": [[1.0]]}, "count", "r.json")
    with pytest.raises(ValueError, match="required to compute"):
        _metric_values({"count": [[1.0]]}, "saliency_norm", "r.json")


def test_compute_retained_uniform():
    scores = {
        "l0": torch.tensor([4.0, 3.0, 2.0, 1.0]),
        "l1": torch.tensor([1.0, 2.0, 3.0, 4.0]),
    }
    num = {"l0": 4, "l1": 4}
    retained = _compute_retained_experts(
        scores, num, sparsity=0.5, uniform=True, floor=1
    )
    # drop 2 lowest per layer, keep 2 highest, returned sorted
    assert retained == {"l0": [0, 1], "l1": [2, 3]}


def test_compute_retained_nonuniform_global_budget_and_floor():
    # layer "low" is globally least salient; budget should be spent there first
    scores = {
        "low": torch.tensor([0.1, 0.2, 0.3, 0.4]),
        "high": torch.tensor([10.0, 11.0, 12.0, 13.0]),
    }
    num = {"low": 4, "high": 4}
    retained = _compute_retained_experts(
        scores, num, sparsity=0.5, uniform=False, floor=2
    )
    # global budget = round(0.5 * 8) = 4, but floor=2 caps drops at 2 per layer
    assert len(retained["low"]) == 2  # dropped down to the floor
    assert retained["low"] == [2, 3]  # kept the two most salient
    assert len(retained["high"]) == 2


# ---------------------------------------------------------------------------
# from_pretrained validation
# ---------------------------------------------------------------------------


def test_from_pretrained_rejects_bad_metric(tmp_path):
    _write_checkpoint(tmp_path)
    report = tmp_path / "r.json"
    _descending_report(report, 4, 8)
    with pytest.raises(ValueError, match="metric must be one of"):
        REAPExpertPruner.from_pretrained(tmp_path, report, metric="bogus")


@pytest.mark.parametrize("sparsity", [1.0, 1.5, -0.1])
def test_from_pretrained_rejects_bad_sparsity(tmp_path, sparsity):
    _write_checkpoint(tmp_path)
    report = tmp_path / "r.json"
    _descending_report(report, 4, 8)
    with pytest.raises(ValueError, match="sparsity must be"):
        REAPExpertPruner.from_pretrained(tmp_path, report, sparsity=sparsity)


def test_from_pretrained_rejects_layer_count_mismatch(tmp_path):
    _write_checkpoint(tmp_path, n_layers=4, n_experts=8)
    report = tmp_path / "r.json"
    _descending_report(report, 3, 8)  # 3 layers vs 4 routers
    with pytest.raises(ValueError, match="router weight"):
        REAPExpertPruner.from_pretrained(tmp_path, report, sparsity=0.25)


def test_from_pretrained_rejects_expert_count_mismatch(tmp_path):
    _write_checkpoint(tmp_path, n_layers=4, n_experts=8)
    report = tmp_path / "r.json"
    _descending_report(report, 4, 6)  # 6 experts in report vs 8 in checkpoint
    with pytest.raises(ValueError, match="experts"):
        REAPExpertPruner.from_pretrained(tmp_path, report, sparsity=0.25)


def test_from_pretrained_requires_num_experts_per_tok(tmp_path):
    # a valid MoE always defines num_experts_per_tok; without it we cannot
    # guarantee the routing floor, so pruning must refuse rather than assume 1
    _write_checkpoint(tmp_path, n_experts=8)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    del config["num_experts_per_tok"]
    config_path.write_text(json.dumps(config))
    report = tmp_path / "r.json"
    _descending_report(report, 4, 8)
    with pytest.raises(ValueError, match="num_experts_per_tok"):
        REAPExpertPruner.from_pretrained(tmp_path, report, sparsity=0.25)


def test_uniform_clamps_to_routing_floor(tmp_path):
    # num_experts_per_tok=6, 8 experts, sparsity 0.5 would drop to 4, but the
    # router still needs 6, so each layer is clamped to num_experts_per_tok
    _write_checkpoint(tmp_path, n_experts=8, num_experts_per_tok=6)
    report = tmp_path / "r.json"
    _descending_report(report, 4, 8)
    pruner = REAPExpertPruner.from_pretrained(
        tmp_path, report, uniform=True, sparsity=0.5
    )
    assert {len(v) for v in pruner.retained_experts.values()} == {6}


# ---------------------------------------------------------------------------
# Selection + config, 2D and 3D layouts
# ---------------------------------------------------------------------------


def test_natural_ordering_aligns_report_to_layers(tmp_path):
    # 12 layers; make layer 7 by far the least salient so it should be pruned most
    _write_checkpoint(tmp_path, n_layers=12, n_experts=8, num_experts_per_tok=2)
    saliency = [[float(8 - e) for e in range(8)] for _ in range(12)]
    saliency[7] = [0.01 * (8 - e) for e in range(8)]  # layer 7 tiny saliency
    report = tmp_path / "r.json"
    _write_report(report, saliency)

    pruner = REAPExpertPruner.from_pretrained(
        tmp_path, report, metric="saliency", uniform=False, sparsity=0.25
    )
    counts = {
        int(name.split(".")[2]): len(v) for name, v in pruner.retained_experts.items()
    }
    # if report/router alignment were lexicographic, layer 7 would not be the min
    assert counts[7] == min(counts.values())
    assert counts[7] == 2  # pruned down to the routing floor


@pytest.mark.parametrize("layout", ["2d", "3d"])
def test_uniform_process_and_config(tmp_path, layout):
    _write_checkpoint(tmp_path, n_layers=3, n_experts=8, layout=layout)
    report = tmp_path / "r.json"
    _descending_report(report, 3, 8)

    pruner = REAPExpertPruner.from_pretrained(
        tmp_path, report, metric="saliency_norm", uniform=True, sparsity=0.25
    )
    assert pruner.is_3d is (layout == "3d")
    # 8 experts, drop round(0.25*8)=2 -> keep 6, every layer the same
    assert {len(v) for v in pruner.retained_experts.values()} == {6}
    # descending saliency -> highest-index experts pruned -> keep 0..5
    assert all(v == list(range(6)) for v in pruner.retained_experts.values())

    result = pruner.process(_load_all(tmp_path / "model.safetensors"))

    # non-MoE tensor passes through
    assert "lm_head.weight" in result
    # router weights sliced to 6 rows
    for name, tensor in result.items():
        if name.endswith("mlp.gate.weight"):
            assert tensor.shape[0] == 6
    if layout == "3d":
        for name, tensor in result.items():
            if "experts" in name:
                assert tensor.shape[0] == 6
    else:
        # exactly 6 experts * 3 projs per layer, indices renumbered to 0..5
        for layer in range(3):
            idxs = sorted(
                int(n.split(".experts.")[1].split(".")[0])
                for n in result
                if n.startswith(f"model.layers.{layer}.mlp.experts.")
            )
            assert set(idxs) == set(range(6))

    # uniform updates num_experts; no quantization_config created
    config = json.loads((tmp_path / "config.json").read_text())
    updated = pruner.update_model_config(dict(config))
    assert updated["num_experts"] == 6
    assert QUANTIZATION_CONFIG_NAME not in updated


@pytest.mark.parametrize("layout", ["2d", "3d"])
def test_nonuniform_process_and_config(tmp_path, layout):
    _write_checkpoint(tmp_path, n_layers=3, n_experts=8, layout=layout)
    report = tmp_path / "r.json"
    _descending_report(report, 3, 8)

    pruner = REAPExpertPruner.from_pretrained(
        tmp_path, report, metric="saliency_norm", uniform=False, sparsity=0.25
    )
    total_retained = sum(len(v) for v in pruner.retained_experts.values())
    # global budget = round(0.25 * 24) = 6 dropped
    assert total_retained == 24 - 6

    # validate() should succeed on the processed output
    pruner.validate(_load_all(tmp_path / "model.safetensors"))

    # non-uniform records per-layer counts in the compressed-tensors config and
    # leaves num_experts untouched
    config = json.loads((tmp_path / "config.json").read_text())
    updated = pruner.update_model_config(dict(config))
    assert updated["num_experts"] == 8  # unchanged
    qconfig = updated[QUANTIZATION_CONFIG_NAME]
    per_layer = qconfig[NUM_EXPERTS_PER_LAYER_KEY]
    assert set(per_layer) == {f"model.layers.{i}.mlp" for i in range(3)}
    assert sum(per_layer.values()) == total_retained


def test_nonuniform_extends_existing_quant_config(tmp_path):
    _write_checkpoint(tmp_path, n_layers=2, n_experts=8)
    report = tmp_path / "r.json"
    _descending_report(report, 2, 8)
    pruner = REAPExpertPruner.from_pretrained(
        tmp_path, report, uniform=False, sparsity=0.25
    )
    config = {"num_experts": 8, QUANTIZATION_CONFIG_NAME: {"format": "float-quantized"}}
    updated = pruner.update_model_config(config)
    # existing quant config preserved and extended, not replaced
    assert updated[QUANTIZATION_CONFIG_NAME]["format"] == "float-quantized"
    assert NUM_EXPERTS_PER_LAYER_KEY in updated[QUANTIZATION_CONFIG_NAME]


def test_uniform_config_key_in_text_config(tmp_path):
    _write_checkpoint(
        tmp_path, n_experts=8, num_experts_key="num_local_experts", text_config=True
    )
    report = tmp_path / "r.json"
    _descending_report(report, 4, 8)
    pruner = REAPExpertPruner.from_pretrained(
        tmp_path, report, uniform=True, sparsity=0.25
    )
    assert pruner.num_experts_config_key == "num_local_experts"
    updated = pruner.update_model_config(
        {"text_config": {"num_local_experts": 8, "num_experts_per_tok": 2}}
    )
    assert updated["text_config"]["num_local_experts"] == 6


# ---------------------------------------------------------------------------
# Smoke tests against real (small) MoE checkpoints
# ---------------------------------------------------------------------------

# (stub, expected is_3d) — Qwen3-1.6B has 2D per-expert weights, Qwen3-VL has 3D
# stacked/fused expert weights and keeps its expert count under text_config
SMOKE_MODELS = [
    ("inference-optimization/Qwen3-1.6B-A0.9B", False),
    ("inference-optimization/Qwen3-VL-1.0B-A0.4B-Instruct", True),
]


def _build_matching_report(model_stub, report_path):
    """Build a correctly-shaped descending-saliency report for a real checkpoint
    by reading its router layout, so no real calibration run is needed."""
    model_files = get_checkpoint_files(model_stub)
    weight_map = get_weight_map(model_files)
    routers = sorted(
        (n for n in weight_map if _is_router_weight(n)), key=_natural_sort_key
    )
    num_experts = _num_experts_by_router(routers, weight_map, model_files)
    saliency = [
        [float(num_experts[r] - e) for e in range(num_experts[r])] for r in routers
    ]
    _write_report(report_path, saliency)
    return routers, num_experts


@pytest.mark.smoke
@pytest.mark.integration
@pytest.mark.parametrize("model_stub, expected_is_3d", SMOKE_MODELS)
def test_smoke_from_pretrained(model_stub, expected_is_3d, tmp_path):
    report_path = tmp_path / "report.json"
    routers, num_experts = _build_matching_report(model_stub, report_path)
    assert routers, "no routers found in checkpoint"

    pruner = REAPExpertPruner.from_pretrained(
        model_stub, report_path, metric="saliency_norm", uniform=True, sparsity=0.25
    )
    assert pruner.is_3d is expected_is_3d
    assert set(pruner.retained_experts) == set(routers)
    for router in routers:
        expected = num_experts[router] - round(0.25 * num_experts[router])
        assert len(pruner.retained_experts[router]) == expected


@pytest.mark.smoke
@pytest.mark.integration
@pytest.mark.parametrize("model_stub, expected_is_3d", SMOKE_MODELS)
def test_smoke_convert_checkpoint_uniform(model_stub, expected_is_3d, tmp_path):
    report_path = tmp_path / "report.json"
    routers, num_experts = _build_matching_report(model_stub, report_path)
    n_experts = next(iter(num_experts.values()))
    expected_kept = n_experts - round(0.25 * n_experts)

    pruner = REAPExpertPruner.from_pretrained(
        model_stub, report_path, metric="saliency_norm", uniform=True, sparsity=0.25
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
    out_files = get_checkpoint_files(save_dir)
    out_map = get_weight_map(out_files)
    out_routers = [n for n in out_map if _is_router_weight(n)]
    assert out_routers
    for name in out_routers:
        with safe_open(out_files[out_map[name]], framework="pt") as f:
            assert f.get_slice(name).get_shape()[0] == expected_kept

    # experts are correspondingly reduced
    if expected_is_3d:
        stacked = [n for n in out_map if _looks_like_expert(n) and n not in out_routers]
        assert stacked
        for name in stacked:
            with safe_open(out_files[out_map[name]], framework="pt") as f:
                assert f.get_slice(name).get_shape()[0] == expected_kept
    else:
        from compressed_tensors.utils.moe import extract_expert_index

        indices = {
            extract_expert_index(n)
            for n in out_map
            if extract_expert_index(n) is not None
        }
        assert max(indices) == expected_kept - 1
