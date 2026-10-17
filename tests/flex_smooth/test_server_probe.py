# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Exercise the standalone server handoff on generated local safetensors only."""

import json
import struct
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
from safetensors.torch import save_file

from tests.quarot.tiny_glm import TinyConfig, TinyGLM
from tools import glm52_precision_probe as probe


def checkpoint(root, sharded=False):
    root.mkdir()
    config = TinyConfig()
    (root / "config.json").write_text(json.dumps(asdict(config)))
    state = TinyGLM(config).state_dict()
    if not sharded:
        save_file(state, root / "model.safetensors")
    else:
        mapping = {}
        for index in range(2):
            shard = f"model-{index}.safetensors"
            subset = {
                name: value
                for i, (name, value) in enumerate(state.items())
                if i % 2 == index
            }
            save_file(subset, root / shard)
            mapping.update({name: shard for name in subset})
        (root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": mapping})
        )


@pytest.mark.parametrize("sharded", [False, True])
def test_header_only_probe(tmp_path, monkeypatch, sharded):
    root = tmp_path / "fixture"
    checkpoint(root, sharded)
    original_open = Path.open
    checked = []

    class HeaderOnlyFile:
        def __init__(self, stream):
            self.stream = stream
            self.limit = 8

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def read(self, count):
            # Reading even one byte of the tensor payload is a test failure.
            assert 0 <= count <= self.limit - self.stream.tell()
            result = self.stream.read(count)
            if self.stream.tell() == 8:
                self.limit += struct.unpack("<Q", result)[0]
            return result

    def guarded_open(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        if path.suffix == ".safetensors":
            checked.append(path)
            return HeaderOnlyFile(stream)
        return stream

    monkeypatch.setattr(Path, "open", guarded_open)
    report = probe.inspect_checkpoint(root)
    assert report["selected_shape_checks_passed"], report["errors"]
    assert report["selected_layers"] == [0, 1]
    assert checked
    assert not report["numerical_l4_passed"]
    assert not report["weight_payloads_read"]
    assert any("experts.0." in key for key in report["selected_tensors"])
    assert not any("experts.1." in key for key in report["selected_tensors"])


def test_incompatible_shape_and_quantized_baseline(tmp_path):
    root = tmp_path / "fixture"
    checkpoint(root)
    config = json.loads((root / "config.json").read_text())
    config.update(
        v_head_dim=16, quantization_config={"quant_method": "compressed-tensors"}
    )
    (root / "config.json").write_text(json.dumps(config))
    report = probe.inspect_checkpoint(root, [0])
    assert not report["selected_shape_checks_passed"]
    assert any("Shape mismatch" in error for error in report["errors"])
    assert any("declares quantization" in error for error in report["errors"])
    assert any("block_size" in error for error in report["errors"])


def test_index_cannot_read_outside_checkpoint(tmp_path):
    root = tmp_path / "fixture"
    checkpoint(root, sharded=True)
    index = root / "model.safetensors.index.json"
    contents = json.loads(index.read_text())
    contents["weight_map"]["model.norm.weight"] = "../outside.safetensors"
    (tmp_path / "outside.safetensors").write_bytes(b"not a checkpoint")
    index.write_text(json.dumps(contents))
    with pytest.raises(ValueError, match="out-of-directory"):
        probe.inspect_checkpoint(root)


def test_standalone_cli_without_site_packages(tmp_path):
    root = tmp_path / "fixture"
    checkpoint(root)
    output = tmp_path / "report.json"
    command = [
        sys.executable,
        "-S",
        str(Path(probe.__file__)),
        "--model-dir",
        str(root),
    ]
    result = subprocess.run(
        command + ["--output", str(output)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())["selected_shape_checks_passed"]
    rejected = subprocess.run(
        command + ["--output", str(root / "config.json")],
        capture_output=True,
        text=True,
    )
    assert rejected.returncode == 2
    assert json.loads((root / "config.json").read_text())["hidden_size"] == 64
