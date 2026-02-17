#!/usr/bin/env python3
"""
Recursively find evaluation result directories and format their metrics into a table.
Searches for directories matching patterns like *-cuda_eval or *-cpu_eval.
"""
import json
from pathlib import Path


def gather_eval_results(root_dir: str = "."):
    """Recursively find eval directories and extract results."""
    root_path = Path(root_dir)
    results = []

    # Find all directories matching *_eval pattern
    for eval_dir in root_path.rglob("*_eval"):
        if not eval_dir.is_dir():
            continue

        # Look for results_*.json files in subdirectories
        for results_file in eval_dir.glob("*/results_*.json"):
            try:
                with open(results_file, 'r') as f:
                    data = json.load(f)

                # Extract timestamp from filename (results_YYYY-MM-DDTHH-MM-SS.xxxxxx.json)
                filename = results_file.name
                timestamp = "N/A"
                if filename.startswith("results_") and len(filename) > 8:
                    # Extract the timestamp part
                    timestamp_part = filename[8:].split('.')[0]  # Remove "results_" prefix and extension
                    # Replace hyphens with colons in time part for readability
                    if len(timestamp_part) >= 19:
                        date_part = timestamp_part[:10]  # YYYY-MM-DD
                        time_part = timestamp_part[11:].replace('-', ':')  # HH:MM:SS
                        timestamp = f"{date_part} {time_part}"

                # Extract relevant metrics
                eval_entry = {
                    "timestamp": timestamp,
                    "model_name": data.get("model_name", "N/A"),
                    "eval_time": data.get("total_evaluation_time_seconds", "N/A"),
                }

                # Extract gsm8k metrics if available
                gsm8k = data.get("results", {}).get("gsm8k", {})
                eval_entry["strict_match"] = gsm8k.get("exact_match,strict-match", "N/A")
                eval_entry["flex_extract"] = gsm8k.get("exact_match,flexible-extract", "N/A")

                results.append(eval_entry)

            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not read {results_file}: {e}")
                continue

    # Sort by timestamp
    results.sort(key=lambda x: x.get("timestamp", ""))

    return results


def format_results_table(results):
    """Format results as a table."""
    if not results:
        print("No evaluation results found.")
        return

    # Define columns with custom widths (name, width)
    columns = [
        ("timestamp", 19),
        ("model_name", 50),
        ("strict_match", 13),
        ("flex_extract", 13),
        ("eval_time", 10),
    ]

    # Print header (centered)
    header = " | ".join(f"{col:^{width}}" for col, width in columns)
    separator = "-" * len(header)
    print(header)
    print(separator)

    # Print rows (left-aligned)
    for entry in results:
        values = []
        for col, width in columns:
            val = entry.get(col, 'N/A')

            # Round float values to 4 decimals for accuracy metrics
            if isinstance(val, float) and col in ["strict_match", "flex_extract"]:
                val = f"{val:.4f}"
            elif isinstance(val, float):
                val = f"{val:.2f}"

            values.append(f"{str(val):<{width}}")

        row = " | ".join(values)
        print(row)


if __name__ == "__main__":
    import sys

    # Allow optional root directory as argument
    root_dir = sys.argv[1] if len(sys.argv) > 1 else "."

    results = gather_eval_results(root_dir)
    format_results_table(results)
