# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from tests.quarot.tiny_glm import TinyGLM
from tools import glm52_l4 as l4
from tools.glm52_l4_capture import capture_inputs


@pytest.fixture
def modelslim_source():
    root = Path(
        os.environ.get(
            "MODELSLIM_SOURCE",
            Path(__file__).resolve().parents[4] / "reference/msmodelslim",
        )
    )
    if not root.is_dir():
        pytest.skip("External ModelSlim source required for handoff integration")
    return root


@pytest.fixture
def fixture(tmp_path):
    with torch.random.fork_rng():
        torch.manual_seed(42)
        model = TinyGLM().eval()
    root = tmp_path / "checkpoint"
    root.mkdir()
    (root / "config.json").write_text(json.dumps(asdict(model.config)))
    save_file(
        {name: value.contiguous() for name, value in model.state_dict().items()},
        str(root / "model.safetensors"),
    )
    return model, root


def capture(model, root, path, layer=0, **kwargs):
    with capture_inputs(
        model,
        model_dir=root,
        layer=layer,
        output=path,
        input_kind="synthetic",
        input_description="seeded tiny token fixture",
        max_tokens=3,
        **kwargs,
    ):
        model(torch.tensor([[1, 3, 5, 7], [2, 4, 6, 8]]))


def test_capture_identity_cleanup_and_bound(fixture, tmp_path):
    model, root = fixture
    tokens = torch.tensor([[1, 3, 5, 7], [2, 4, 6, 8]])
    before = model(tokens)[0]
    path = tmp_path / "cache.safetensors"
    capture(model, root, path)
    torch.testing.assert_close(model(tokens)[0], before, rtol=0, atol=0)
    assert not any(
        module._forward_hooks or module._forward_pre_hooks for module in model.modules()
    )
    with safe_open(str(path), framework="pt") as f:
        meta = json.loads(f.metadata()["glm52_l4"])
        assert all(value == 3 for value in meta["tokens_saved"].values())
        assert all(f.get_slice(key).get_shape()[0] == 3 for key in f.keys())
    with pytest.raises(ValueError, match="must be new"):
        capture(model, root, path)


def test_capture_failure_cleans_hooks(fixture, tmp_path):
    model, root = fixture
    path = tmp_path / "failed.safetensors"
    with pytest.raises(RuntimeError, match="inference failed"):
        with capture_inputs(
            model,
            model_dir=root,
            layer=0,
            output=path,
            input_kind="synthetic",
            input_description="failure",
        ):
            raise RuntimeError("inference failed")
    assert not path.exists()
    assert not any(
        module._forward_hooks or module._forward_pre_hooks for module in model.modules()
    )


def test_only_selected_payloads_and_cache_provenance(fixture, tmp_path, monkeypatch):
    model, root = fixture
    config, plan = l4.checkpoint_plan(root, 0)
    requested = []
    original = l4.safe_open

    class Reader:
        def __init__(self, *args, **kwargs):
            self.context = original(*args, **kwargs)

        def __enter__(self):
            self.reader = self.context.__enter__()
            return self

        def __exit__(self, *args):
            return self.context.__exit__(*args)

        def get_tensor(self, key):
            requested.append(key)
            assert key.startswith("model.layers.0.")
            assert ".mlp." not in key
            return self.reader.get_tensor(key)

    monkeypatch.setattr(l4, "safe_open", Reader)
    _, fingerprints = l4.load_selected(root, plan)
    assert set(requested) == {
        entry["key"] for entry in plan["selected_weights"].values()
    }
    monkeypatch.setattr(l4, "safe_open", original)
    path = tmp_path / "cache.safetensors"
    capture(model, root, path)
    values, _ = l4.load_cache(path, plan, fingerprints, config, 2)
    assert all(value.shape[0] == 2 for value in values.values())
    fingerprints["input_layernorm"] = "wrong"
    with pytest.raises(ValueError, match="provenance"):
        l4.load_cache(path, plan, fingerprints, config, 2)


@pytest.mark.parametrize(
    "layer,dtype", [(0, torch.float32), (1, torch.float32), (0, torch.bfloat16)]
)
def test_real_source_numerical_handoff(
    fixture, tmp_path, layer, dtype, modelslim_source
):
    model, root = fixture
    model.to(dtype)
    save_file(model.state_dict(), str(root / "model.safetensors"))
    reference = modelslim_source
    path = tmp_path / "cache" / "inputs.safetensors"
    capture(model, root, path, layer)
    output = tmp_path / f"report-{layer}.json"
    code = l4.main(
        [
            "--model-dir",
            str(root),
            "--layer",
            str(layer),
            "--modelslim-source",
            str(reference),
            "--activation-cache",
            str(path),
            "--output",
            str(output),
            "--threads",
            "1",
        ]
    )
    report = json.loads(output.read_text())
    assert report.get("numerical_checks_passed"), report
    assert not report["full_layer_l4_passed"]
    assert not report["scoped_l4_passed"]  # A captured tiny input is still synthetic.
    assert code == 3


def test_dry_run_and_budget_never_load_payload(fixture, tmp_path, monkeypatch):
    _, root = fixture

    def forbidden(*args, **kwargs):
        pytest.fail("Dry run must not read payloads")

    monkeypatch.setattr(l4, "safe_open", forbidden)
    output = tmp_path / "dry.json"
    assert (
        l4.main(
            [
                "--model-dir",
                str(root),
                "--layer",
                "0",
                "--dry-run",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text())["status"] == "headers_only_no_payload_read"
    with pytest.raises(ValueError, match="exceed"):
        l4.checkpoint_plan(root, 0, max_weight_gib=1e-9)


def test_mismatches_cannot_pass():
    mismatch = l4.compare(torch.ones(2), torch.zeros(2))
    assert not l4.all_passed({"checks": {"injected": mismatch}})
    assert not l4.compare(torch.tensor([float("nan")]), torch.ones(1))["passed"]


def test_sharded_traversal_blocked(fixture, tmp_path):
    model, root = fixture
    weight_map = {key: "model.safetensors" for key in model.state_dict()}
    weight_map["model.layers.0.input_layernorm.weight"] = "../outside.safetensors"
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    (tmp_path / "outside.safetensors").write_bytes(b"never read")
    with pytest.raises(ValueError, match="out-of-directory"):
        l4.checkpoint_plan(root, 0)


def test_sharded_selected_reads(fixture):
    model, root = fixture
    state = model.state_dict()
    selected = {
        key: value for key, value in state.items() if key.startswith("model.layers.0.")
    }
    save_file(selected, str(root / "selected.safetensors"))
    # Unselected shard is deliberately absent: no other layer payload is needed.
    weight_map = {
        key: "selected.safetensors" if key in selected else "absent.safetensors"
        for key in state
    }
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    _, plan = l4.checkpoint_plan(root, 0)
    weights, _ = l4.load_selected(root, plan)
    assert len(weights) == 11


def test_capture_missing_hooks_and_changed_weights(fixture, tmp_path):
    model, root = fixture
    for change in (False, True):
        output = tmp_path / f"missing-{change}.safetensors"
        with pytest.raises(
            ValueError, match="weights changed" if change else "Missing executed hooks"
        ):
            with capture_inputs(
                model,
                model_dir=root,
                layer=0,
                output=output,
                input_kind="synthetic",
                input_description="negative fixture",
            ):
                if change:
                    model(torch.tensor([[1, 2]]))
                    with torch.no_grad():
                        model.model.layers[0].self_attn.q_a_proj.weight.add_(1)
        assert not output.exists()
        assert not any(
            module._forward_hooks or module._forward_pre_hooks
            for module in model.modules()
        )


def test_runner_injected_transform_failure_is_reported(
    fixture, tmp_path, monkeypatch, modelslim_source
):
    _, root = fixture
    reference = modelslim_source
    original = l4.rotate_axis
    monkeypatch.setattr(
        l4, "rotate_axis", lambda *args, **kwargs: original(*args, **kwargs) + 0.01
    )
    output = tmp_path / "failure.json"
    code = l4.main(
        [
            "--model-dir",
            str(root),
            "--layer",
            "0",
            "--modelslim-source",
            str(reference),
            "--synthetic-activations",
            "--output",
            str(output),
            "--threads",
            "1",
            "--tokens",
            "3",
        ]
    )
    report = json.loads(output.read_text())
    assert code == 1
    assert report["status"] == "numerical_failure"
    assert not report["numerical_checks_passed"]


def test_baseline_and_output_protection(fixture, tmp_path):
    _, root = fixture
    with pytest.raises(ValueError, match="Output must be new"):
        l4.safe_output(root / "new.json", [root])
    config_path = root / "config.json"
    config = json.loads(config_path.read_text())
    config["quarot_config"] = {"status": "applied"}
    config_path.write_text(json.dumps(config))
    output = tmp_path / "rejected.json"
    assert (
        l4.main(
            [
                "--model-dir",
                str(root),
                "--layer",
                "0",
                "--dry-run",
                "--output",
                str(output),
            ]
        )
        == 2
    )
    report = json.loads(output.read_text())
    assert report["status"] == "error" and not report["scoped_l4_passed"]


def test_cache_wrong_stage_is_rejected(fixture, tmp_path):
    model, root = fixture
    cache = tmp_path / "cache.safetensors"
    capture(model, root, cache)
    with safe_open(str(cache), framework="pt") as f:
        values = {key: f.get_tensor(key) for key in f.keys()}
        metadata = json.loads(f.metadata()["glm52_l4"])
    metadata["stage"] = "after_quarot"
    wrong = tmp_path / "wrong-stage.safetensors"
    save_file(values, str(wrong), metadata={"glm52_l4": json.dumps(metadata)})
    config, plan = l4.checkpoint_plan(root, 0)
    _, fingerprints = l4.load_selected(root, plan)
    with pytest.raises(ValueError, match="provenance"):
        l4.load_cache(wrong, plan, fingerprints, config, 3)
