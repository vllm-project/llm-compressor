"""
Conversion tool for converting AutoAWQ quantized checkpoints to the
compressed-tensors format (``pack_quantized`` compressor).

AutoAWQ stores int4 weights in int32 tensors with an interleaved packing
order ``[0, 2, 4, 6, 1, 3, 5, 7]``, while compressed-tensors uses the
sequential order ``[0, 1, 2, 3, 4, 5, 6, 7]``.  This module handles the
re-packing and metadata generation so the output model can be loaded
directly by vLLM.

Usage (CLI)::

    python -m llmcompressor.conversion.autoawq_to_ct \\
        --model-path /path/to/autoawq-model \\
        --output-path /path/to/output \\
        --num-bits 4 --group-size 128

Usage (Python API)::

    from llmcompressor.conversion.autoawq_to_ct import convert_autoawq_to_ct

    convert_autoawq_to_ct(
        model_path="/path/to/autoawq-model",
        output_path="/path/to/output",
    )
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

logger = logging.getLogger(__name__)

__all__ = ["convert_autoawq_to_ct"]

# AutoAWQ packs 8 int4 values into int32 using the interleaved order
# ``[0, 2, 4, 6, 1, 3, 5, 7]``.  The *inverse* permutation
# ``[0, 4, 1, 5, 2, 6, 3, 7]`` maps bit-positions back to the
# sequential column indices expected by compressed-tensors.
_AWQ_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]
_AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]  # inverse of _AWQ_ORDER

# AutoAWQ tensor suffix → compressed-tensors tensor suffix
_KEY_MAP = {
    ".qweight": ".weight_packed",
    ".scales": ".weight_scale",
    ".qzeros": ".weight_zero_point",
}


# ---------------------------------------------------------------------------
# Weight conversion helpers
# ---------------------------------------------------------------------------


def _unpack_awq_int4(packed: torch.Tensor) -> torch.Tensor:
    """Unpack int4 values from AutoAWQ's **interleaved** int32 packing.

    AutoAWQ's ``gemm_pack`` packs 8 int4 values per int32 using::

        order_map = [0, 2, 4, 6, 1, 3, 5, 7]
        for i in range(8):
            int_weight[:, col] |= weight[:, col*8 + order_map[i]] << (i * 4)

    This function reverses that process and returns values in the natural
    (sequential) column order.

    :param packed: ``(rows, cols // 8)`` int32 tensor.
    :return: ``(rows, cols)`` int32 tensor with values in ``[0, 15]``.
    """
    rows, packed_cols = packed.shape
    cols = packed_cols * 8

    # Step 1: extract the 8 nibbles stored at each bit-position.
    raw = torch.zeros((rows, cols), dtype=torch.int32, device=packed.device)
    for bit_pos in range(8):
        raw[:, bit_pos::8] = (packed >> (bit_pos * 4)) & 0xF

    # Step 2: undo the interleaving.
    # Bit-position ``bit_pos`` holds the original column ``_AWQ_ORDER[bit_pos]``
    # within each group of 8.  We scatter back to sequential order.
    result = torch.zeros_like(raw)
    for seq_idx, bit_pos in enumerate(_AWQ_REVERSE_ORDER):
        result[:, seq_idx::8] = raw[:, bit_pos::8]

    return result


def _pack_ct_int4(values: torch.Tensor) -> torch.Tensor:
    """Pack int4 values into compressed-tensors' **sequential** int32 format.

    compressed-tensors stores 8 int4 values per int32 in natural order:
    ``value[i]`` occupies bits ``4*i … 4*i+3``.

    :param values: ``(rows, cols)`` int32 tensor (each element in ``[0, 15]``).
    :return: ``(rows, cols // 8)`` int32 tensor.
    """
    rows, cols = values.shape
    if cols % 8 != 0:
        raise ValueError(f"columns must be divisible by 8, got {cols}")

    packed = torch.zeros(
        (rows, cols // 8), dtype=torch.int32, device=values.device
    )
    for i in range(8):
        packed |= (values[:, i::8] & 0xF).to(torch.int32) << (i * 4)
    return packed


def _repack_awq_to_ct(packed_awq: torch.Tensor) -> torch.Tensor:
    """One-shot conversion: AWQ-packed int32 → CT-packed int32."""
    return _pack_ct_int4(_unpack_awq_int4(packed_awq))


# ---------------------------------------------------------------------------
# Key renaming helpers
# ---------------------------------------------------------------------------


def _rename_key(key: str, awq_prefixes: set[str]) -> str:
    """Return the compressed-tensors key name for *key*, or *key* unchanged."""
    for prefix in awq_prefixes:
        if not key.startswith(prefix):
            continue
        suffix = key[len(prefix):]
        if suffix in _KEY_MAP:
            return prefix + _KEY_MAP[suffix]
    return key


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------


def convert_autoawq_to_ct(
    model_path: str | Path,
    output_path: str | Path,
    num_bits: int = 4,
    group_size: int = 128,
    symmetric: bool = False,
) -> None:
    """Convert an AutoAWQ checkpoint to the compressed-tensors ``pack_quantized``
    format so that the resulting model can be loaded directly in vLLM.

    :param model_path: directory containing the AutoAWQ model.
    :param output_path: destination directory for the converted model.
    :param num_bits: quantization bit-width (default 4).
    :param group_size: quantization group size (default 128).
    :param symmetric: ``True`` for symmetric quantisation (AutoAWQ default
        is *asymmetric*, i.e. ``False``).
    """
    model_path = Path(model_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Converting AutoAWQ model: %s → %s", model_path, output_path)

    # ----- Load model config -----
    config = AutoConfig.from_pretrained(model_path)
    awq_config = getattr(config, "quantization_config", None)
    if awq_config and isinstance(awq_config, dict):
        num_bits = awq_config.get("bits", num_bits)
        group_size = awq_config.get("group_size", group_size)
        # AutoAWQ uses ``zero_point: True`` to indicate *asymmetric* quant.
        symmetric = not awq_config.get("zero_point", True)
    logger.info(
        "Quantisation params: bits=%d  group_size=%d  symmetric=%s",
        num_bits, group_size, symmetric,
    )

    # ----- Discover safetensors shards -----
    st_files = sorted(model_path.glob("*.safetensors"))
    if not st_files:
        raise FileNotFoundError(
            f"No .safetensors files in {model_path}. "
            "Make sure the model was saved in safetensors format."
        )
    logger.info("Found %d safetensors shard(s)", len(st_files))

    # Collect *all* AWQ quantised layer prefixes across shards so that the
    # index-file rewriting can reference them.
    all_awq_prefixes: set[str] = set()

    # ----- Convert each shard -----
    for st_file in tqdm(st_files, desc="Converting shards"):
        converted: dict[str, torch.Tensor] = {}

        with safe_open(str(st_file), framework="pt", device="cpu") as f:
            keys = list(f.keys())

            # AWQ prefixes in *this* shard
            shard_prefixes: set[str] = set()
            for key in keys:
                if key.endswith(".qweight"):
                    shard_prefixes.add(key.removesuffix(".qweight"))
            all_awq_prefixes |= shard_prefixes

            for key in tqdm(keys, desc=f"  {st_file.name}", leave=False):
                tensor = f.get_tensor(key)

                # Try to match to an AWQ quantised layer
                matched_prefix = None
                for prefix in shard_prefixes:
                    if key.startswith(prefix):
                        matched_prefix = prefix
                        break

                if matched_prefix is None:
                    # Non-quantised parameter – pass through unchanged.
                    converted[key] = tensor
                    continue

                suffix = key[len(matched_prefix):]

                if suffix == ".qweight":
                    converted[f"{matched_prefix}.weight_packed"] = (
                        _repack_awq_to_ct(tensor)
                    )

                elif suffix == ".scales":
                    converted[f"{matched_prefix}.weight_scale"] = tensor

                elif suffix == ".qzeros":
                    # Zero-points are also packed with the AWQ interleave.
                    zp = _unpack_awq_int4(tensor)
                    converted[f"{matched_prefix}.weight_zero_point"] = zp

                elif suffix == ".bias":
                    converted[key] = tensor

                else:
                    converted[key] = tensor

        save_file(converted, str(output_path / st_file.name))

    # ----- Build compressed-tensors quantization_config -----
    strategy = "group" if group_size > 0 else "channel"
    quant_config = {
        "quant_method": "compressed-tensors",
        "format": "pack_quantized",
        "global_compression_ratio": None,
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": num_bits,
                    "type": "int",
                    "symmetric": symmetric,
                    "strategy": strategy,
                    "group_size": group_size if group_size > 0 else None,
                },
                "input_activations": None,
                "output_activations": None,
            }
        },
        "ignore": ["lm_head"],
    }

    # ----- Write config.json -----
    config_dict = config.to_dict()
    config_dict["quantization_config"] = quant_config
    with open(output_path / "config.json", "w") as fp:
        json.dump(config_dict, fp, indent=2)
    logger.info("Wrote config.json with compressed-tensors quantization_config")

    # ----- Tokenizer -----
    try:
        tok = AutoTokenizer.from_pretrained(model_path)
        tok.save_pretrained(output_path)
        logger.info("Saved tokenizer")
    except Exception as exc:
        logger.warning("Could not copy tokenizer: %s", exc)

    # ----- Rewrite safetensors index (multi-shard models) -----
    for idx_file in model_path.glob("*.safetensors.index.json"):
        with open(idx_file) as fp:
            index = json.load(fp)

        new_map: dict[str, str] = {}
        for old_key, shard_name in index.get("weight_map", {}).items():
            new_map[_rename_key(old_key, all_awq_prefixes)] = shard_name
        index["weight_map"] = new_map

        with open(output_path / idx_file.name, "w") as fp:
            json.dump(index, fp, indent=2)
        logger.info("Rewrote %s", idx_file.name)

    # ----- Copy any remaining auxiliary files -----
    _auxiliary_globs = [
        "generation_config.json",
        "special_tokens_map.json",
        "merges.txt",
    ]
    for pattern in _auxiliary_globs:
        for src in model_path.glob(pattern):
            dst = output_path / src.name
            if not dst.exists():
                shutil.copy2(src, dst)

    logger.info("Conversion complete.")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an AutoAWQ quantized model checkpoint to the "
            "compressed-tensors pack_quantized format."
        ),
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the AutoAWQ model directory.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Destination directory for the converted model.",
    )
    parser.add_argument(
        "--num-bits",
        type=int,
        default=4,
        help="Quantization bit-width (default: 4).",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=128,
        help="Quantization group size (default: 128).",
    )
    parser.add_argument(
        "--symmetric",
        action="store_true",
        default=False,
        help="Treat quantisation as symmetric (default: asymmetric).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    convert_autoawq_to_ct(
        model_path=args.model_path,
        output_path=args.output_path,
        num_bits=args.num_bits,
        group_size=args.group_size,
        symmetric=args.symmetric,
    )


if __name__ == "__main__":
    main()
