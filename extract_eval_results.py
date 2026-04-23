#!/usr/bin/env python3
"""Extract evaluation results (flexible/strict match values, wikitext ppl, and model names) from log files."""

import argparse
import re
import sys
from datetime import datetime


def extract_results(log_path):
    with open(log_path) as f:
        text = f.read()

    # Find all result blocks: gsm8k scores, wikitext ppl, and model name
    pattern = re.compile(
        r"\|gsm8k\|\s*\d+\|flexible-extract\|\s*\d+\|exact_match\|↑\s*\|(\S+)\|"
        r".*?"
        r"strict-match\s*\|\s*\d+\|exact_match\|↑\s*\|(\S+)\|"
        r".*?"
        r"word_perplexity\|↓\s*\|(\S+)\|"
        r".*?"
        r"Cleanup complete for (.+?)$",
        re.DOTALL | re.MULTILINE,
    )

    results = pattern.findall(text)
    if not results:
        print("No results found in log file.")
        sys.exit(1)

    # Extract quantization runtimes
    reset_pattern = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+) \| reset \| INFO - Compression lifecycle reset")
    finalize_pattern = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+) \| finalize \| INFO - Compression lifecycle finalized")

    reset_times = reset_pattern.findall(text)
    finalize_times = finalize_pattern.findall(text)

    runtimes = []
    for i, (reset_str, finalize_str) in enumerate(zip(reset_times, finalize_times)):
        reset_time = datetime.fromisoformat(reset_str)
        finalize_time = datetime.fromisoformat(finalize_str)
        duration = finalize_time - reset_time
        # Convert to minutes
        duration_minutes = duration.total_seconds() / 60
        runtimes.append(duration_minutes)

    print(f"{'Model':<55} {'Flexible':>10} {'Strict':>10} {'WikiPPL':>10} {'Quant(min)':>12}")
    print("-" * 100)
    for i, (flexible, strict, ppl, model) in enumerate(results):
        runtime_str = f"{runtimes[i]:.2f}" if i < len(runtimes) else "N/A"
        print(f"{model:<55} {flexible:>10} {strict:>10} {ppl:>10} {runtime_str:>12}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract eval results from log files.")
    parser.add_argument("log_file", help="Path to the log file")
    args = parser.parse_args()
    extract_results(args.log_file)
