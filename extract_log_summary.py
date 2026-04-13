#!/usr/bin/env python3
"""Extract summary data from AWQ DDP regression test log files.

Parses the log output from run_all_tests.sh and produces a comparison table
showing pre-DDP vs post-DDP results across models, schemes, and benchmarks.

Usage:
    python extract_log_summary.py regression_results.log
"""

import re
import sys
from collections import defaultdict


def extract_log_summary(log_path):
    with open(log_path, "r") as f:
        content = f.read()

    # Split into sections by the box-drawing delimiters
    section_pattern = re.compile(
        r"║\s+MODEL:\s+(.+?)\n"
        r"\s*║\s+SCHEME:\s+(.+?)\n"
        r"\s*║\s+CODE STATE:\s+(.+?)\n"
        r".*?"
        r"(?=║\s+MODEL:|╔══.*FINAL SUMMARY|\Z)",
        re.DOTALL,
    )

    sections = section_pattern.findall(content)

    # results[model][scheme][code_state] = {task: {metric: value}}
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    timing = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for model, scheme, code_state in sections:
        model = model.strip()
        scheme = scheme.strip()
        code_state = code_state.strip()

        # Find the body for this section
        pattern = re.escape(f"MODEL: {model}") + r".*?" + re.escape(f"CODE STATE: {code_state}")
        match = re.search(pattern, content, re.DOTALL)
        if not match:
            continue
        # Get everything after the match until the next section
        start = match.end()
        next_section = re.search(r"║\s+MODEL:", content[start:])
        end = start + next_section.start() if next_section else len(content)
        body = content[start:end]

        # Extract timing
        time_match = re.search(
            r"Time:\s*([\d.]+)\s*minutes\s*\(([\d.]+)\s*seconds\)", body
        )
        if time_match:
            timing[model][scheme][code_state]["time_min"] = float(
                time_match.group(1)
            )

        gpu_match = re.search(r"Peak GPU Memory:\s*([\d.]+)\s*GB", body)
        if gpu_match:
            timing[model][scheme][code_state]["gpu_gb"] = float(gpu_match.group(1))

        # Extract GSM8K flexible-extract
        flex_match = re.search(
            r"flexible-extract\|.*?\|.*?\|.*?\|([\d.]+)\|", body
        )
        if flex_match:
            results[model][scheme][code_state]["gsm8k_flex"] = float(
                flex_match.group(1)
            )

        # Extract GSM8K strict-match
        strict_match = re.search(
            r"strict-match\s*\|.*?\|.*?\|.*?\|([\d.]+)\|", body
        )
        if strict_match:
            results[model][scheme][code_state]["gsm8k_strict"] = float(
                strict_match.group(1)
            )

        # Extract wikitext word_perplexity
        ppl_match = re.search(
            r"word_perplexity\s*\|.*?\|.*?\|.*?\|([\d.]+)\|", body
        )
        if ppl_match:
            results[model][scheme][code_state]["wikitext_ppl"] = float(
                ppl_match.group(1)
            )

        # Extract MMLU accuracy
        mmlu_match = re.search(
            r"acc\s*\|.*?\|.*?\|.*?\|([\d.]+)\|", body
        )
        if mmlu_match:
            results[model][scheme][code_state]["mmlu_acc"] = float(
                mmlu_match.group(1)
            )

    # Print comparison table
    print(f"\nLog: {log_path}\n")

    metrics = ["gsm8k_flex", "gsm8k_strict", "wikitext_ppl", "mmlu_acc"]
    metric_labels = ["GSM8K Flex", "GSM8K Strict", "Wiki PPL", "MMLU Acc"]

    header = (
        f"{'Model':<35} {'Scheme':<12} {'State':<10} "
        f"{'Time':>7} {'GPU':>6} "
        + " ".join(f"{m:>12}" for m in metric_labels)
    )
    print(header)
    print("-" * len(header))

    for model in sorted(results.keys()):
        for scheme in sorted(results[model].keys()):
            for code_state in ["pre-ddp", "post-ddp"]:
                r = results[model][scheme].get(code_state, {})
                t = timing[model][scheme].get(code_state, {})

                time_str = (
                    f"{t['time_min']:.1f}m" if "time_min" in t else "N/A"
                )
                gpu_str = (
                    f"{t['gpu_gb']:.1f}G" if "gpu_gb" in t else "N/A"
                )

                vals = []
                for m in metrics:
                    if m in r:
                        vals.append(f"{r[m]:.4f}")
                    else:
                        vals.append("N/A")

                print(
                    f"{model:<35} {scheme:<12} {code_state:<10} "
                    f"{time_str:>7} {gpu_str:>6} "
                    + " ".join(f"{v:>12}" for v in vals)
                )

            # Print delta row
            pre = results[model][scheme].get("pre-ddp", {})
            post = results[model][scheme].get("post-ddp", {})
            deltas = []
            for m in metrics:
                if m in pre and m in post:
                    diff = post[m] - pre[m]
                    sign = "+" if diff >= 0 else ""
                    # For perplexity, lower is better so flip the sign indicator
                    if m == "wikitext_ppl":
                        indicator = " *" if diff > 0 else ""
                    else:
                        indicator = " *" if diff < -0.01 else ""
                    deltas.append(f"{sign}{diff:.4f}{indicator}")
                else:
                    deltas.append("---")

            print(
                f"{'':35} {'':12} {'delta':<10} "
                f"{'':>7} {'':>6} "
                + " ".join(f"{d:>12}" for d in deltas)
            )
            print()


if __name__ == "__main__":
    log_path = sys.argv[1] if len(sys.argv) > 1 else "regression_results.log"
    extract_log_summary(log_path)
