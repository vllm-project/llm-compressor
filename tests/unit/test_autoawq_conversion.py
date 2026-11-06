# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import json
from pathlib import Path

import pytest
import torch
from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
from compressed_tensors.entrypoints.convert import AutoAWQConverter, convert_checkpoint
from safetensors.torch import load_file, save_file


@pytest.fixture
def example(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[2] / "examples/convert_checkpoint")
    )
    return importlib.import_module("autoawq_example")


@pytest.fixture(params=["single-shard", "cross-shard"])
def checkpoint(tmp_path, request):
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
    weight_tensors = {
        "model.proj.qweight": torch.tensor(
            [[1966171168], [-38146904], [38146903], [-1966171169]],
            dtype=torch.int32,
        ).repeat(8, 1),
        "model.proj.bias": torch.arange(8, dtype=torch.float16),
    }
    qparam_tensors = {
        "model.proj.qzeros": torch.tensor(
            [[-76293807], [839962095]], dtype=torch.int32
        ).repeat(8, 1),
        "model.proj.scales": torch.tensor(
            [[1, 2, 3, 4, 5, 6, 7, 8], [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]],
            dtype=torch.float16,
        ).repeat(8, 1),
        "model.norm.weight": torch.ones(4),
    }
    if request.param == "single-shard":
        save_file(weight_tensors | qparam_tensors, source / "model.safetensors")
    else:
        shards = {
            "model-00001-of-00002.safetensors": weight_tensors,
            "model-00002-of-00002.safetensors": qparam_tensors,
        }
        for filename, tensors in shards.items():
            save_file(tensors, source / filename)
        (source / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        name: filename
                        for filename, tensors in shards.items()
                        for name in tensors
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


def test_verify_converted_weights(example, checkpoint):
    assert example.verify_weights(*checkpoint) == 1


def test_conversion_preserves_integer_weights_and_qparams(example, checkpoint):
    _, output = checkpoint
    files = example.tensor_files(output)
    assert set(files) == {
        "model.proj.weight_packed",
        "model.proj.weight_shape",
        "model.proj.weight_scale",
        "model.proj.weight_zero_point",
        "model.proj.bias",
        "model.norm.weight",
    }
    shape = example.load_tensor(files, "model.proj.weight_shape")
    torch.testing.assert_close(shape, torch.tensor([8, 32]), rtol=0, atol=0)
    packed = example.load_tensor(files, "model.proj.weight_packed")
    assert packed.dtype == torch.int32
    assert packed.shape == (8, 4)
    weight = unpack_from_int32(packed, 4, torch.Size(shape.tolist()))
    expected_weight = (
        torch.tensor(
            [
                [0, 1, 2, 3, 4, 5, 6, 7],
                [8, 9, 10, 11, 12, 13, 14, 15],
                [7, 6, 5, 4, 3, 2, 1, 0],
                [15, 14, 13, 12, 11, 10, 9, 8],
            ],
            dtype=torch.int8,
        )
        .repeat(8, 1)
        .T
        - 8
    )
    torch.testing.assert_close(weight, expected_weight, rtol=0, atol=0)

    scale = example.load_tensor(files, "model.proj.weight_scale")
    expected_scale = (
        torch.tensor(
            [[1, 2, 3, 4, 5, 6, 7, 8], [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]],
            dtype=torch.float16,
        )
        .repeat(8, 1)
        .T
    )
    torch.testing.assert_close(scale, expected_scale, rtol=0, atol=0)
    packed_zero = example.load_tensor(files, "model.proj.weight_zero_point")
    assert packed_zero.dtype == torch.int32
    assert packed_zero.shape == (1, 16)
    zero = unpack_from_int32(packed_zero, 4, scale.shape, packed_dim=0)
    expected_zero = (
        torch.tensor(
            [[1, 3, 5, 7, 9, 11, 13, 15], [15, 0, 14, 1, 13, 2, 12, 3]],
            dtype=torch.int8,
        )
        .repeat(8, 1)
        .T
        - 8
    )
    torch.testing.assert_close(zero, expected_zero, rtol=0, atol=0)


def test_conversion_preserves_unquantized_tensors(example, checkpoint):
    source, output = checkpoint
    source_files, output_files = (
        example.tensor_files(source),
        example.tensor_files(output),
    )
    for name in ("model.proj.bias", "model.norm.weight"):
        torch.testing.assert_close(
            example.load_tensor(output_files, name),
            example.load_tensor(source_files, name),
            rtol=0,
            atol=0,
        )


def test_conversion_writes_w4a16_config(checkpoint):
    _, output = checkpoint
    config = json.loads((output / "config.json").read_text())
    assert config["model_type"] == "qwen2"
    quantization = config["quantization_config"]
    assert quantization["quant_method"] == "compressed-tensors"
    assert quantization["format"] == "pack-quantized"
    assert quantization["quantization_status"] == "compressed"
    assert quantization["ignore"] == ["lm_head"]
    assert quantization["kv_cache_scheme"] is None
    assert len(quantization["config_groups"]) == 1
    group = next(iter(quantization["config_groups"].values()))
    assert group["format"] == "pack-quantized"
    assert group["targets"] == ["Linear"]
    assert group["input_activations"] is None
    assert group["output_activations"] is None
    weights = group["weights"]
    assert weights["num_bits"] == 4
    assert weights["type"] == "int"
    assert weights["symmetric"] is False
    assert weights["strategy"] == "group"
    assert weights["group_size"] == 2
    assert weights["dynamic"] is False


def test_conversion_writes_complete_index(example, checkpoint):
    source, output = checkpoint
    source_files = example.tensor_files(source)
    output_files = example.tensor_files(output)
    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert index["weight_map"] == {
        name: path.name for name, path in output_files.items()
    }
    assert index["metadata"]["total_size"] == sum(
        example.load_tensor(output_files, name).nbytes for name in output_files
    )
    if len(set(source_files.values())) == 1:
        assert {path.name for path in output_files.values()} == {"model.safetensors"}
    else:
        assert source_files["model.proj.qweight"] != source_files["model.proj.scales"]
        assert source_files["model.proj.qweight"] != source_files["model.proj.qzeros"]


@pytest.mark.parametrize(
    "suffix", ["weight_packed", "weight_zero_point", "weight_scale"]
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
        example.verify_weights(source, output)
