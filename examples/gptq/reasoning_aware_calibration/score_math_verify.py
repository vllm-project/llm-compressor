#!/usr/bin/env python3
"""Score an Inspect MATH log with Math-Verify and retain per-sample usage."""

from __future__ import annotations

import argparse
import json
import math
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


def latest_eval(path: Path) -> Path:
    logs = sorted(path.glob("*.eval"), key=lambda item: item.stat().st_mtime)
    if not logs:
        raise FileNotFoundError(f"No .eval files found in {path}")
    return logs[-1]


def score_log(path: Path) -> dict[str, Any]:
    try:
        math_verify_version = version("math-verify")
    except PackageNotFoundError:
        math_verify_version = "unknown"

    from inspect_ai.log import read_eval_log
    from math_verify import parse, verify

    log = read_eval_log(str(path))
    samples = log.samples or []
    if not samples:
        raise ValueError(f"No samples found in Inspect log: {path}")

    records: list[dict[str, Any]] = []
    correct = 0
    parsed_predictions = 0
    parsed_targets = 0
    sample_errors = 0
    scorer_errors = 0

    for sample in samples:
        metadata = sample.metadata or {}
        target = metadata.get("solution")
        completion = sample.output.completion if sample.output else None
        usage = sample.output.usage if sample.output else None
        output_tokens = getattr(usage, "output_tokens", None) if usage else None
        sample_error = sample.error is not None
        sample_errors += int(sample_error)
        errors: list[str] = []
        parsed_target = None
        parsed_prediction = None
        verdict = False

        if isinstance(target, str) and target.strip():
            try:
                parsed_target = parse(target)
            except Exception as exc:
                errors.append(f"target parse: {type(exc).__name__}: {exc}")
        if isinstance(completion, str) and completion.strip():
            try:
                parsed_prediction = parse(completion)
            except Exception as exc:
                errors.append(f"prediction parse: {type(exc).__name__}: {exc}")

        target_parsed = bool(parsed_target)
        prediction_parsed = bool(parsed_prediction)
        parsed_targets += int(target_parsed)
        parsed_predictions += int(prediction_parsed)

        if not sample_error and target_parsed and prediction_parsed:
            try:
                verdict = bool(verify(parsed_target, parsed_prediction))
            except Exception as exc:
                errors.append(f"verify: {type(exc).__name__}: {exc}")

        if errors:
            scorer_errors += 1
        correct += int(verdict)
        records.append(
            {
                "sample_id": str(sample.id),
                "epoch": sample.epoch,
                "correct": verdict,
                "target_parsed": target_parsed,
                "prediction_parsed": prediction_parsed,
                "sample_error": sample_error,
                "scorer_error": "; ".join(errors) if errors else None,
                "output_tokens": int(output_tokens) if output_tokens is not None else None,
            }
        )

    count = len(samples)
    accuracy = correct / count
    stderr_pp = math.sqrt(accuracy * (1 - accuracy) / count) * 100
    return {
        "eval_log": str(path.resolve()),
        "math_verify_version": math_verify_version,
        "method": "parse(full target solution), parse(full completion), verify(target, prediction)",
        "correct": correct,
        "samples": count,
        "accuracy": accuracy,
        "stderr_percentage_points": stderr_pp,
        "parsed_predictions": parsed_predictions,
        "parsed_targets": parsed_targets,
        "sample_errors": sample_errors,
        "scorer_errors": scorer_errors,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-score a saved Inspect MATH log with Math-Verify."
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Inspect .eval file, or a directory containing top-level .eval files.",
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    eval_path = latest_eval(args.path) if args.path.is_dir() else args.path
    if not eval_path.is_file():
        parser.error(f"Inspect log does not exist: {eval_path}")
    result = score_log(eval_path)
    json_path = args.json_out or eval_path.with_suffix(".math_verify.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"Math-Verify {result['math_verify_version']} re-score: {eval_path}")
    print(
        "Accuracy ± standard error: "
        f"{result['accuracy'] * 100:.2f}% ± {result['stderr_percentage_points']:.2f} pp "
        f"({result['correct']}/{result['samples']})"
    )
    print(
        "Parsed predictions / targets; sample errors / scorer errors: "
        f"{result['parsed_predictions']} / {result['parsed_targets']}; "
        f"{result['sample_errors']} / {result['scorer_errors']}"
    )
    print(f"Per-sample JSON: {json_path}")


if __name__ == "__main__":
    main()
