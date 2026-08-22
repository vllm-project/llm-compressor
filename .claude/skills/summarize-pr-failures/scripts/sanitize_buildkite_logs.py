#!/usr/bin/env python3
"""
Sanitize Buildkite logs by removing ANSI codes, timestamps, and other noise.
Usage: python sanitize_buildkite_logs.py <input_dir> <output_dir>
"""

import re
import sys
from pathlib import Path


def remove_ansi_codes(text: str) -> str:
    """Remove ANSI escape codes from text."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


def remove_timestamps(text: str) -> str:
    """Remove common timestamp patterns."""
    # Remove patterns like [2024-01-01 12:34:56]
    text = re.sub(r'\[\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\]', '', text)
    # Remove patterns like 2024-01-01T12:34:56.123Z
    text = re.sub(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z', '', text)
    # Remove patterns like [12:34:56]
    text = re.sub(r'\[\d{2}:\d{2}:\d{2}\]', '', text)
    return text


def remove_progress_indicators(text: str) -> str:
    """Remove progress bars and spinners."""
    # Remove progress bars like [=====>    ] 50%
    text = re.sub(r'\[=*>?\s*\]\s*\d+%?', '', text)
    # Remove lines with only dots or dashes (progress indicators)
    text = re.sub(r'^[.\-]+\s*$', '', text, flags=re.MULTILINE)
    return text


def remove_buildkite_metadata(text: str) -> str:
    """Remove Buildkite-specific metadata and decorative elements."""
    # Remove lines with box drawing characters
    text = re.sub(r'^[│║╔╗╚╝═─┌┐└┘├┤┬┴┼]+\s*$', '', text, flags=re.MULTILINE)
    # Remove lines that are purely decorative borders
    text = re.sub(r'^[=*-]{3,}\s*$', '', text, flags=re.MULTILINE)
    # Remove "~~~ " prefix that buildkite adds
    text = re.sub(r'^~~~ ', '', text, flags=re.MULTILINE)
    return text


def extract_relevant_errors(text: str) -> str:
    """Extract the most relevant error information."""
    lines = text.split('\n')
    cleaned_lines = []

    # Keywords that indicate important error information
    error_keywords = [
        'error:', 'failed', 'failure', 'exception', 'traceback',
        'assert', 'test_', 'passed', 'skipped', 'xfailed',
        '::test_', 'FAILED', 'ERROR', 'ERRORS', 'warnings summary'
    ]

    in_error_section = False
    blank_line_count = 0

    for line in lines:
        line_lower = line.lower()

        # Check if this line is relevant
        is_relevant = any(keyword in line_lower for keyword in error_keywords)

        if is_relevant:
            in_error_section = True
            blank_line_count = 0
            cleaned_lines.append(line)
        elif in_error_section:
            # Keep context around errors
            if line.strip():
                cleaned_lines.append(line)
                blank_line_count = 0
            else:
                blank_line_count += 1
                # Stop after 3 consecutive blank lines
                if blank_line_count > 3:
                    in_error_section = False
                else:
                    cleaned_lines.append(line)
        elif line.startswith('=') or line.startswith('_'):
            # Keep pytest separator lines as they provide structure
            cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)


def sanitize_log(log_content: str, aggressive: bool = False) -> str:
    """
    Sanitize a log file by removing noise and keeping relevant information.

    Args:
        log_content: Raw log content
        aggressive: If True, only keep error-related content

    Returns:
        Sanitized log content
    """
    # Apply all sanitization steps
    sanitized = remove_ansi_codes(log_content)
    sanitized = remove_timestamps(sanitized)
    sanitized = remove_progress_indicators(sanitized)
    sanitized = remove_buildkite_metadata(sanitized)

    if aggressive:
        sanitized = extract_relevant_errors(sanitized)

    # Remove excessive blank lines
    sanitized = re.sub(r'\n{4,}', '\n\n\n', sanitized)

    # Remove leading/trailing whitespace
    sanitized = sanitized.strip()

    return sanitized


def main():
    if len(sys.argv) < 2:
        print("Usage: python sanitize_buildkite_logs.py <input_dir> [output_dir] [--aggressive]")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else input_dir / "sanitized"
    aggressive = "--aggressive" in sys.argv

    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    log_files = list(input_dir.glob("*.log"))

    if not log_files:
        print(f"No .log files found in {input_dir}")
        sys.exit(0)

    print(f"Sanitizing {len(log_files)} log file(s)...")
    print(f"Mode: {'aggressive (errors only)' if aggressive else 'standard'}")

    for log_file in log_files:
        print(f"  - {log_file.name}")

        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        sanitized = sanitize_log(content, aggressive=aggressive)

        output_file = output_dir / log_file.name
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(sanitized)

    print(f"\nSanitized logs saved to: {output_dir}")


if __name__ == "__main__":
    main()
