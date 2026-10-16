# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import json
from pathlib import Path

import pytest
import torch
from compressed_tensors.entrypoints.convert import AutoAWQConverter, convert_checkpoint
from safetensors.torch import load_file, save_file


@pytest.fixture
def example(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[2] / "examples/convert_checkpoint")
    )
    return importlib.import_module("autoawq_example")


@pytest.fixture
def checkpoint(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2",
                "quantization_config": {
                    "quant_method": "awq",
                    "version": "gemm",
                    "bits": 4,
                    "group_size": 2,
                    "zero_point": True,
                },
            }
        )
    )
    # Literal words and expected weights are independent of the CT packing code.
    save_file(
        {
            "model.proj.qweight": torch.tensor(
                [[1966171168], [-38146904], [38146903], [-1966171169]],
                dtype=torch.int32,
            ).repeat(8, 1),
            "model.proj.bias": torch.arange(8, dtype=torch.float16),
        },
        source / "model-00001-of-00002.safetensors",
    )
    save_file(
        {
            "model.proj.qzeros": torch.tensor(
                [[-76293807], [839962095]], dtype=torch.int32
            ).repeat(8, 1),
            "model.proj.scales": torch.tensor(
                [[1, 2, 3, 4, 5, 6, 7, 8], [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]],
                dtype=torch.float16,
            ).repeat(8, 1),
            "model.norm.weight": torch.ones(4),
        },
        source / "model-00002-of-00002.safetensors",
    )
    (source / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.proj.qweight": "model-00001-of-00002.safetensors",
                    "model.proj.bias": "model-00001-of-00002.safetensors",
                    "model.proj.qzeros": "model-00002-of-00002.safetensors",
                    "model.proj.scales": "model-00002-of-00002.safetensors",
                    "model.norm.weight": "model-00002-of-00002.safetensors",
                }
            }
        )
    )
    convert_checkpoint(
        source,
        output,
        AutoAWQConverter.from_pretrained(str(source)),
        device="cpu",
        max_workers=1,
    )
    return source, output


def test_source_reference_matches_known_gemm_weights(example, checkpoint):
    source, _ = checkpoint
    files = example.tensor_files(source)
    q = example.unpack_awq_gemm(example.load_tensor(files, "model.proj.qweight"))
    z = example.unpack_awq_gemm(example.load_tensor(files, "model.proj.qzeros"))
    scales = example.load_tensor(files, "model.proj.scales").float()
    groups = torch.arange(q.shape[0]) // 2
    actual = (q - z[groups]).float() * scales[groups]
    expected = torch.tensor(
        [
            [-1, -4, -9, -16, -25, -36, -49, -64],
            [7, 12, 15, 16, 15, 12, 7, 0],
            [-4, 6, -13.5, 6, -25, 0, -38.5, -12],
            [0, 14, -1.5, 22, -5, 24, -10.5, 20],
        ]
    )
    torch.testing.assert_close(actual, expected.repeat(8, 1), rtol=0, atol=0)


def test_verify_converted_checkpoint_with_cross_shard_qparams(example, checkpoint):
    assert example.verify_checkpoint(*checkpoint) == 1


@pytest.mark.parametrize(
    "suffix", ["weight_packed", "weight_zero_point", "weight_scale", "bias"]
)
def test_verification_rejects_corrupted_tensors(example, checkpoint, suffix):
    source, output = checkpoint
    name = f"model.proj.{suffix}"
    path = example.tensor_files(output)[name]
    tensors = load_file(path)
    tensors[name] = tensors[name].clone()
    tensors[name].view(-1)[0] += 1
    save_file(tensors, path)
    with pytest.raises(AssertionError):
        example.verify_checkpoint(source, output)


@pytest.mark.parametrize(
    "field", ["symmetric", "targets", "ignore", "format", "output_activations"]
)
def test_verification_rejects_incorrect_config(example, checkpoint, field):
    source, output = checkpoint
    path = output / "config.json"
    config = json.loads(path.read_text())
    group = next(iter(config["quantization_config"]["config_groups"].values()))
    if field == "symmetric":
        group["weights"][field] = True
    elif field == "targets":
        group[field] = []
    elif field == "ignore":
        config["quantization_config"][field] = ["model.proj"]
    elif field == "format":
        group[field] = "int-quantized"
    else:
        group[field] = {"num_bits": 8, "type": "int"}
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="source W4A16"):
        example.verify_checkpoint(source, output)


def test_verification_rejects_incorrect_index(example, checkpoint):
    source, output = checkpoint
    path = output / "model.safetensors.index.json"
    index = json.loads(path.read_text())
    index["weight_map"]["model.proj.weight_packed"] = "missing.safetensors"
    path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="index"):
        example.verify_checkpoint(source, output)
