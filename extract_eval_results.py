#!/usr/bin/env python3
"""Extract evaluation results (flexible/strict match values and model names) from log files."""

import argparse
import re
import sys


def extract_results(log_path):
    with open(log_path) as f:
        text = f.read()

    # Find all result blocks: the table lines followed by "Cleanup complete for <model>"
    pattern = re.compile(
        r"\|gsm8k\|\s*\d+\|flexible-extract\|\s*\d+\|exact_match\|↑\s*\|(\S+)\|"
        r".*?"
        r"strict-match\s*\|\s*\d+\|exact_match\|↑\s*\|(\S+)\|"
        r".*?"
        r"Cleanup complete for (.+?)$",
        re.DOTALL | re.MULTILINE,
    )

    results = pattern.findall(text)
    if not results:
        print("No results found in log file.")
        sys.exit(1)

    print(f"{'Model':<55} {'Flexible':>10} {'Strict':>10}")
    print("-" * 77)
    for flexible, strict, model in results:
        print(f"{model:<55} {flexible:>10} {strict:>10}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract eval results from log files.")
    parser.add_argument("log_file", help="Path to the log file")
    args = parser.parse_args()
    extract_results(args.log_file)
