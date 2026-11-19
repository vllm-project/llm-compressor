#!/usr/bin/env python3
"""Compare each quantized answer's generated tokens with the BF16 answer."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def load_records(path: Path) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"No per-sample records found in {path}")
    result: dict[str, int] = {}
    for record in records:
        sample_id = str(record["sample_id"])
        tokens = record.get("output_tokens")
        if tokens is None:
            raise ValueError(f"Missing output token usage for sample {sample_id} in {path}")
        if sample_id in result:
            raise ValueError(f"Duplicate sample ID {sample_id} in {path}")
        result[sample_id] = int(tokens)
    return result


def summarize(
    baseline: dict[str, int], quantized: dict[str, int], bottom_k: int
) -> dict[str, Any]:
    if set(baseline) != set(quantized):
        raise ValueError("Baseline and quantized evals must contain identical sample IDs")
    if any(tokens <= 0 for tokens in baseline.values()):
        raise ValueError("Baseline output-token counts must all be positive")
    if not 1 <= bottom_k <= len(baseline):
        raise ValueError(f"bottom-k must be between 1 and {len(baseline)}")

    per_sample = sorted(
        (
            quantized[sample_id] / baseline[sample_id] * 100,
            sample_id,
            quantized[sample_id],
            baseline[sample_id],
        )
        for sample_id in baseline
    )
    ordered_ratios = sorted(item[0] for item in per_sample)
    middle = len(ordered_ratios) // 2
    if len(ordered_ratios) % 2:
        median = ordered_ratios[middle]
    else:
        median = (ordered_ratios[middle - 1] + ordered_ratios[middle]) / 2
    p90 = ordered_ratios[math.ceil(0.90 * len(ordered_ratios)) - 1]
    bottom = per_sample[:bottom_k]
    return {
        "samples": len(per_sample),
        "mean_per_sample_percent": sum(ordered_ratios) / len(ordered_ratios),
        "median_per_sample_percent": median,
        "p90_nearest_rank_percent": p90,
        "bottom_k": bottom_k,
        "bottom_k_mean_per_sample_percent": sum(item[0] for item in bottom) / bottom_k,
        "bottom_k_pooled_percent": (
            sum(item[2] for item in bottom) / sum(item[3] for item in bottom) * 100
        ),
        "bottom_k_quantized_output_tokens": sum(item[2] for item in bottom),
        "bottom_k_bf16_output_tokens": sum(item[3] for item in bottom),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize quantized/BF16 answer-token ratios from scorer JSON files."
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument(
        "--condition",
        action="append",
        required=True,
        metavar="LABEL=JSON",
        help="Condition label and Math-Verify JSON sidecar; repeat for each quantization.",
    )
    parser.add_argument("--bottom-k", type=int, default=450)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    baseline = load_records(args.baseline)
    results = {}
    for specification in args.condition:
        if "=" not in specification:
            parser.error(f"Condition must be LABEL=JSON: {specification}")
        label, filename = specification.split("=", maxsplit=1)
        results[label] = summarize(
            baseline, load_records(Path(filename)), args.bottom_k
        )

    print(
        "| Condition | Mean % | Median % | P90 % (nearest rank) | "
        f"Bottom {args.bottom_k} mean % | Bottom {args.bottom_k} pooled % |"
    )
    print("|---|---:|---:|---:|---:|---:|")
    for label, result in results.items():
        print(
            f"| {label} | {result['mean_per_sample_percent']:.2f} | "
            f"{result['median_per_sample_percent']:.2f} | "
            f"{result['p90_nearest_rank_percent']:.2f} | "
            f"{result['bottom_k_mean_per_sample_percent']:.2f} | "
            f"{result['bottom_k_pooled_percent']:.2f} |"
        )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
