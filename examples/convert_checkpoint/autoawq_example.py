# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Convert and verify an existing asymmetric AutoAWQ GEMM W4A16 checkpoint."""

import argparse
import json
from pathlib import Path

import torch
from compressed_tensors.compressors import PackedQuantizationCompressor
from compressed_tensors.entrypoints.convert import AutoAWQConverter, convert_checkpoint
from compressed_tensors.quantization import QuantizationConfig
from safetensors import safe_open


def tensor_files(directory: Path) -> dict[str, Path]:
    """Index tensor names from safetensors headers without loading the weights."""
    files = {}
    for path in sorted(directory.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as shard:
            for name in shard.keys():
                if name in files:
                    raise ValueError(f"Duplicate tensor: {name}")
                files[name] = path
    return files


def load_tensor(files: dict[str, Path], name: str) -> torch.Tensor:
    with safe_open(files[name], framework="pt", device="cpu") as shard:
        return shard.get_tensor(name)


def unpack_awq_gemm(words: torch.Tensor) -> torch.Tensor:
    """Decode the source producer's packing, independently of the CT converter.

    AutoAWQ v0.2.9 from_linear packs output offsets in this order:
    https://github.com/casper-hansen/AutoAWQ/blob/v0.2.9/awq/modules/linear/gemm.py
    GEMM stores the zero-points directly; there is no zero-point +1 adjustment.
    """
    if words.ndim != 2 or words.dtype != torch.int32:
        raise ValueError("Expected a two-dimensional int32 AutoAWQ GEMM tensor")
    unpacked = torch.empty(words.shape[0], words.shape[1] * 8, dtype=torch.int32)
    for slot, output_offset in enumerate((0, 2, 4, 6, 1, 3, 5, 7)):
        unpacked[:, output_offset::8] = (words >> (4 * slot)) & 15
    return unpacked


def verify_weights(source: Path, output: Path) -> int:
    """Compare every layer's reconstructed weights against the source checkpoint."""
    source_config = json.loads((source / "config.json").read_text())[
        "quantization_config"
    ]
    group_size = source_config["group_size"]
    output_config = json.loads((output / "config.json").read_text())[
        "quantization_config"
    ]
    config = QuantizationConfig.model_validate(output_config)
    scheme = next(iter(config.config_groups.values()))

    source_files, output_files = tensor_files(source), tensor_files(output)
    modules = sorted(
        name.removesuffix(".qweight")
        for name in source_files
        if name.endswith(".qweight")
    )
    if not modules:
        raise ValueError("No AutoAWQ qweight tensors found")
    for module in modules:
        q = unpack_awq_gemm(load_tensor(source_files, f"{module}.qweight"))
        z = unpack_awq_gemm(load_tensor(source_files, f"{module}.qzeros"))
        scales = load_tensor(source_files, f"{module}.scales")

        # Reconstruct the source in float32 using AutoAWQ's (K, N) convention.
        groups = torch.arange(q.shape[0]) // group_size
        reference = (q - z[groups]).float() * scales[groups].float()

        # Reconstruct the converted layer with the CT consumer, in (N, K) order.
        compressed = {
            suffix: load_tensor(output_files, f"{module}.{suffix}")
            for suffix in (
                "weight_packed",
                "weight_shape",
                "weight_scale",
                "weight_zero_point",
            )
        }
        compressed["weight_scale"] = compressed["weight_scale"].float()
        actual = PackedQuantizationCompressor.decompress(compressed, scheme)["weight"]
        torch.testing.assert_close(actual, reference.T, rtol=0, atol=0)
    return len(modules)


def run_inference(output: Path) -> None:
    """Load through Transformers and run a CPU smoke test after dequantization."""
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        CompressedTensorsConfig,
    )

    tokenizer = AutoTokenizer.from_pretrained(output)
    model, loading_info = AutoModelForCausalLM.from_pretrained(
        output,
        dtype=torch.float32,
        quantization_config=CompressedTensorsConfig(dequantize=True),
        output_loading_info=True,
    )
    if any(
        loading_info.get(key)
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ):
        raise ValueError(f"Checkpoint did not load cleanly: {loading_info}")
    model.eval()
    inputs = tokenizer("The capital of France is", return_tensors="pt")
    with torch.inference_mode():
        if not torch.isfinite(model(**inputs).logits).all():
            raise ValueError("Non-finite logits after loading")
        tokens = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    print(tokenizer.decode(tokens[0], skip_special_tokens=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-inference", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a new output directory")
    converter = AutoAWQConverter.from_pretrained(str(args.source))
    convert_checkpoint(
        model_stub=args.source,
        save_directory=args.output,
        converter=converter,
        device="cpu",
        max_workers=1,
    )
    count = verify_weights(args.source, args.output)
    print(f"Verified {count} AWQ layers against the source checkpoint")
    if args.run_inference:
        run_inference(args.output)


if __name__ == "__main__":
    main()
