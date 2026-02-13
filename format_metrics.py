#!/usr/bin/env python3
"""
Format metrics.json into a table with selected columns.
"""
import json


def format_metrics_table(json_file: str = "metrics.json"):
    """Read metrics JSON and format as a table."""
    # Read the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Define columns with custom widths (name, width)
    columns = [
        ("model_id", 40),
        ("device_map", 11),
        ("world_size", 11),
        ("max_time", 10),
        ("max_memory", 11),
        ("strict_match", 13),
        ("flex_extract", 13),
    ]

    # Print header (centered)
    header = " | ".join(f"{col:^{width}}" for col, width in columns)
    separator = "-" * len(header)
    print(header)
    print(separator)

    # Print rows (left-aligned)
    for entry in data:
        values = []
        for col, width in columns:
            # Handle nested eval_metrics
            if col == "strict_match":
                val = entry.get("eval_metrics", {}).get("strict_match", {}).get("value", "N/A")
            elif col == "flex_extract":
                val = entry.get("eval_metrics", {}).get("flexible_extract", {}).get("value", "N/A")
            else:
                val = entry.get(col, 'N/A')

            # Round float values to 2 decimals
            if isinstance(val, float):
                val = f"{val:.2f}"

            values.append(f"{str(val):<{width}}")

        row = " | ".join(values)
        print(row)


if __name__ == "__main__":
    format_metrics_table()
