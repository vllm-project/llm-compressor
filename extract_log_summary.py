#!/usr/bin/env python3
"""Extract summary data from AWQ benchmark log files.

For each section (delimited by "Running: <script>"), extracts:
- Model name (from vllm evaluation output)
- Max time across all ranks (seconds & minutes)
- Max peak GPU memory across all ranks (GB)
- GSM8K flexible-extract exact_match score
- GSM8K strict-match exact_match score
"""

import re
import sys


def extract_log_summary(log_path):
    with open(log_path, "r") as f:
        content = f.read()

    # Split into sections by the "Running:" delimiter
    section_pattern = re.compile(
        r"={40,}\nRunning: (.+?)\n={40,}\n(.*?)(?=\n={40,}\nRunning:|\Z)",
        re.DOTALL,
    )

    sections = section_pattern.findall(content)

    results = []
    for script_name, body in sections:
        # Extract model name from vllm results line: vllm ({'pretrained': '...', ...})
        vllm_results = re.findall(
            r"^vllm \(\{'pretrained':\s*'([^']+)'", body, re.MULTILINE
        )
        if vllm_results:
            model_name = vllm_results[0]
        else:
            # Fallback: get from Initializing vllm model line
            model_match = re.findall(r"'pretrained':\s*'(\./[^']+)'", body)
            model_name = model_match[0] if model_match else "N/A"

        # Extract all times (in seconds) - take max across ranks
        times = re.findall(r"^Time:.*?\(([\d.]+) seconds\)", body, re.MULTILINE)
        max_time = max(float(t) for t in times) if times else None

        # Extract all peak GPU memory values - take max across ranks
        gpus = re.findall(r"^Peak GPU Memory:\s*([\d.]+) GB", body, re.MULTILINE)
        max_gpu = max(float(g) for g in gpus) if gpus else None

        # Extract flexible-extract score
        flex_match = re.search(
            r"flexible-extract\|.*?\|.*?\|↑\s*\|([\d.]+)\|", body
        )
        flexible = float(flex_match.group(1)) if flex_match else None

        # Extract strict-match score
        strict_match = re.search(
            r"strict-match\s*\|.*?\|.*?\|↑\s*\|([\d.]+)\|", body
        )
        strict = float(strict_match.group(1)) if strict_match else None

        results.append({
            "script": script_name.strip(),
            "model": model_name,
            "max_time_s": max_time,
            "max_time_min": round(max_time / 60, 2) if max_time else None,
            "max_gpu_gb": max_gpu,
            "flexible_extract": flexible,
            "strict_match": strict,
        })

    # Print results as a table
    header = (
        f"{'Script':<50} {'Model':<55} {'Time (min)':>10} "
        f"{'GPU (GB)':>9} {'Flex':>7} {'Strict':>7}"
    )
    print(f"Log: {log_path}\n")
    print(header)
    print("-" * len(header))
    for r in results:
        time_str = f"{r['max_time_min']:.2f}" if r["max_time_min"] is not None else "N/A"
        gpu_str = f"{r['max_gpu_gb']:.2f}" if r["max_gpu_gb"] is not None else "N/A"
        flex_str = f"{r['flexible_extract']:.4f}" if r["flexible_extract"] is not None else "N/A"
        strict_str = f"{r['strict_match']:.4f}" if r["strict_match"] is not None else "N/A"
        print(
            f"{r['script']:<50} {r['model']:<55} {time_str:>10} "
            f"{gpu_str:>9} {flex_str:>7} {strict_str:>7}"
        )


if __name__ == "__main__":
    log_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "/home/HDCharles/logs/20260311-164539_awqddp_fin_sh-run-sh.log"
    )
    extract_log_summary(log_path)
