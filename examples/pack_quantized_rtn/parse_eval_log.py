#!/usr/bin/env python3
"""Parse an lm_eval log and print a summary table."""

import re
import sys

def parse_log(path):
    with open(path) as f:
        lines = f.readlines()

    results = {}
    current_model = None
    metrics = set()

    for line in lines:
        m = re.match(r"=== Evaluating: (.+) ===", line)
        if m:
            current_model = m.group(1)
            results[current_model] = {}
            continue

        if current_model and line.startswith("|") and "|" in line[1:]:
            cols = [c.strip() for c in line.split("|")]
            # cols[0] is empty, cols[1] is task/empty, ..., cols[5] is metric, cols[7] is value
            if len(cols) >= 8:
                metric = cols[5]
                value = cols[7]
                if metric and metric not in ("Metric", ""):
                    try:
                        results[current_model][metric] = float(value)
                        metrics.add(metric)
                    except ValueError:
                        pass

    return results, sorted(metrics)


def print_summary(results, metrics):
    col_w = max(len(m) for m in metrics)
    name_w = max(len(n) for n in results)
    header = f"{'Model':<{name_w}}  " + "  ".join(f"{m:>{col_w}}" for m in metrics)
    print(header)
    print("-" * len(header))
    for model, vals in results.items():
        row = f"{model:<{name_w}}  " + "  ".join(
            f"{vals.get(m, float('nan')):>{col_w}.4f}" for m in metrics
        )
        print(row)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else (
        "/home/HDCharles/logs/20260610-151553_sh-home-HDCharles-repos-llm-compressor-examples-pack_quantized_rtn-run_evals-sh.log"
    )
    results, metrics = parse_log(path)
    print_summary(results, metrics)
