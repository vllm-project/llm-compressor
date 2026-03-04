#!/usr/bin/env python3
"""Extract model evaluation results from log file."""

import re
import sys

log_file = sys.argv[1] if len(sys.argv) > 1 else "/home/HDCharles/logs/20260305-162700_updated_sh-run-sh.log"

# Pattern to match the vllm summary line with pretrained model name
model_pattern = re.compile(r"vllm \(\{'pretrained': '([^']+)'")
# Patterns for flexible and strict match scores
flexible_pattern = re.compile(r"\|gsm8k\|.*\|flexible-extract\|.*\|exact_match\|[^|]*\|([\d.]+)\|[±\s]")
strict_pattern = re.compile(r"\|.*\|.*\|strict-match\s*\|.*\|exact_match\|[^|]*\|([\d.]+)\|[±\s]")

results = []
current_model = None

with open(log_file) as f:
    for line in f:
        model_match = model_pattern.search(line)
        if model_match:
            current_model = model_match.group(1).lstrip("./")

        flex_match = flexible_pattern.search(line)
        if flex_match and current_model:
            flexible_score = flex_match.group(1)
            # Next line should have strict match, store model + flexible for now
            results.append({"model": current_model, "flexible": flexible_score, "strict": None})

        strict_match = strict_pattern.search(line)
        if strict_match and results and results[-1]["strict"] is None:
            results[-1]["strict"] = strict_match.group(1)

# Print results
print(f"{'Model':<60} {'Flexible':>10} {'Strict':>10}")
print("-" * 82)
for r in results:
    print(f"{r['model']:<60} {r['flexible']:>10} {r['strict']:>10}")
