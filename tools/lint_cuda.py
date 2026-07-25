#!/usr/bin/env python3
"""
Linter to detect torch.cuda usage and suggest torch.accelerator API instead.

This linter scans Python files for direct torch.cuda API calls and recommends
using the torch.accelerator API for better device abstraction and portability.
"""

import ast
import argparse
import sys
import warnings
from pathlib import Path
from typing import List, Tuple, Optional


class TorchCudaLinter(ast.NodeVisitor):
    """AST visitor to detect torch.cuda usage."""

    def __init__(self, filename: str):
        self.filename = filename
        self.issues: List[Tuple[int, int, str, str]] = []
        self.reported_positions = set()  # Track (line, col) to avoid duplicates
        self.torch_cuda_suggestions = {
            "torch.cuda.is_available": "torch.accelerator.is_available",
            "torch.cuda.device_count": "torch.accelerator.device_count",
            "torch.cuda.current_device": "torch.accelerator.current_device_index",
            "torch.cuda.current_stream": "torch.accelerator.current_stream",
            "torch.cuda.set_device": "torch.accelerator.set_device_index",
            "torch.cuda.synchronize": "torch.accelerator.synchronize",
            "torch.cuda.Stream": "torch.Stream",
            "torch.cuda.stream": "torch.Stream",
            "torch.cuda.Event": "torch.Event",
            "torch.cuda.memory_allocated": "torch.accelerator.memory_allocated",
            "torch.cuda.memory_reserved": "torch.accelerator.memory_reserved",
            "torch.cuda.empty_cache": "torch.accelerator.empty_cache",
            "torch.cuda.max_memory_allocated": "torch.accelerator.max_memory_allocated",
            "torch.cuda.reset_peak_memory_stats": "torch.accelerator.reset_peak_memory_stats",
        }

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Visit attribute access nodes (e.g., torch.cuda.is_available)."""
        # Build the full attribute chain
        full_attr = self._get_full_attribute(node)

        # Only report the outermost torch.cuda usage to avoid duplicates
        if full_attr and full_attr.startswith("torch.cuda"):
            position = (node.lineno, node.col_offset)
            if position not in self.reported_positions:
                suggestion = self._get_suggestion(full_attr)
                self.issues.append((
                    node.lineno,
                    node.col_offset,
                    full_attr,
                    suggestion
                ))
                self.reported_positions.add(position)

        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Visit 'from torch.cuda import ...' statements."""
        if node.module and node.module.startswith("torch.cuda"):
            for alias in node.names:
                imported_name = alias.name
                self.issues.append((
                    node.lineno,
                    node.col_offset,
                    f"from {node.module} import {imported_name}",
                    f"Consider using 'from torch.accelerator import {imported_name}' instead"
                ))

        self.generic_visit(node)

    def _get_full_attribute(self, node: ast.AST) -> Optional[str]:
        """Recursively build the full attribute chain (e.g., torch.cuda.is_available)."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            parent = self._get_full_attribute(node.value)
            if parent:
                return f"{parent}.{node.attr}"
        return None

    def _get_suggestion(self, cuda_call: str) -> str:
        """Get the appropriate torch.accelerator suggestion for a torch.cuda call."""
        # Check for partial matches (e.g., torch.cuda.foo.bar)
        for cuda_api, suggestion in self.torch_cuda_suggestions.items():
            if cuda_call.startswith(cuda_api):
                return suggestion

        # Generic suggestion
        return cuda_call.replace("torch.cuda", "torch.get_device_module()")


def apply_fixes(filepath: Path, issues: List[Tuple[int, int, str, str]], verbose: bool = False) -> bool:
    """
    Apply automatic fixes to a file using the suggested replacements.

    Args:
        filepath: Path to the Python file to fix
        issues: List of issues to fix (line, col, usage, suggestion)
        verbose: Whether to print verbose output

    Returns:
        True if fixes were applied, False otherwise
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Group issues by line to handle multiple issues on the same line
        issues_by_line = {}
        for line_num, col_offset, usage, suggestion in issues:
            if line_num not in issues_by_line:
                issues_by_line[line_num] = []
            issues_by_line[line_num].append((col_offset, usage, suggestion))

        modified = False
        changes_made = []  # Track what actually changed for warning message

        # Process lines in reverse order to maintain positions
        for line_num in sorted(issues_by_line.keys(), reverse=True):
            if line_num > len(lines):
                continue

            line_idx = line_num - 1
            original_line = lines[line_idx]
            new_line = original_line

            # Apply each suggested fix on this line
            # Sort by column offset in reverse to maintain positions during replacement
            sorted_issues = sorted(issues_by_line[line_num], key=lambda x: x[0], reverse=True)

            for col_offset, usage, suggestion in sorted_issues:
                # Apply the suggested replacement
                if usage in new_line:
                    new_line = new_line.replace(usage, suggestion, 1)
                    if verbose:
                        print(f"  Fixed line {line_num}: {usage} → {suggestion}")

            if new_line != original_line:
                lines[line_idx] = new_line
                modified = True
                # Track unique changes for the summary
                for _, usage, suggestion in sorted_issues:
                    change = (usage, suggestion)
                    if change not in changes_made:
                        changes_made.append(change)

        if modified:
            with open(filepath, "w", encoding="utf-8") as f:
                f.writelines(lines)

            # Print warning about what changed
            print(f"\n⚠️  Applied fixes to {filepath}:")
            for usage, suggestion in changes_made:
                print(f"   {usage} → {suggestion}")
            print()

            return True
        return False

    except Exception as e:
        if verbose:
            print(f"Error applying fixes to {filepath}: {e}", file=sys.stderr)
        return False


def lint_file(filepath: Path, verbose: bool = False) -> List[Tuple[int, int, str, str]]:
    """
    Lint a single Python file for torch.cuda usage.

    Args:
        filepath: Path to the Python file to lint
        verbose: Whether to print verbose output

    Returns:
        List of issues found (line, col, usage, suggestion)
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Suppress SyntaxWarnings (e.g., invalid escape sequences) from parsed files
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=SyntaxWarning)
            tree = ast.parse(content, filename=str(filepath))

        linter = TorchCudaLinter(str(filepath))
        linter.visit(tree)

        return linter.issues
    except SyntaxError as e:
        if verbose:
            print(f"Syntax error in {filepath}: {e}", file=sys.stderr)
        return []
    except Exception as e:
        if verbose:
            print(f"Error processing {filepath}: {e}", file=sys.stderr)
        return []


def lint_directory(directory: Path, verbose: bool = False) -> dict:
    """
    Recursively lint all Python files in a directory.

    Args:
        directory: Path to directory to lint
        verbose: Whether to print verbose output

    Returns:
        Dictionary mapping filenames to lists of issues
    """
    all_issues = {}

    for py_file in directory.rglob("*.py"):
        issues = lint_file(py_file, verbose=verbose)
        if issues:
            all_issues[str(py_file)] = issues

    return all_issues


def print_issues(all_issues: dict, verbose: bool = False) -> int:
    """
    Print linting issues in a readable format.

    Args:
        all_issues: Dictionary mapping filenames to lists of issues
        verbose: Whether to print in verbose mode

    Returns:
        Number of total issues found
    """
    total_issues = 0

    for filename, issues in sorted(all_issues.items()):
        print(f"\n{filename}")
        for line, col, usage, suggestion in issues:
            total_issues += 1
            print(f"  Line {line}, Col {col}: {usage}")
            print(f"    → Suggestion: Use '{suggestion}' instead")
            if verbose:
                print(f"    → Reason: torch.accelerator provides better device abstraction")

    return total_issues


def main():
    parser = argparse.ArgumentParser(
        description="Lint Python files for torch.cuda usage and suggest torch.accelerator alternatives"
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Files or directories to lint"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--fail-on-issues",
        action="store_true",
        help="Exit with non-zero status if issues are found"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Automatically apply fixes by replacing torch.cuda with torch.accelerator"
    )

    args = parser.parse_args()

    all_issues = {}

    for path in args.paths:
        if not path.exists():
            print(f"Error: Path '{path}' does not exist", file=sys.stderr)
            continue

        if path.is_file():
            if path.suffix == ".py":
                issues = lint_file(path, verbose=args.verbose)
                if issues:
                    all_issues[str(path)] = issues
        elif path.is_dir():
            dir_issues = lint_directory(path, verbose=args.verbose)
            all_issues.update(dir_issues)

    if all_issues:
        if args.fix:
            # Fix mode: apply automatic fixes
            fixed_files = 0
            total_issues = 0

            print(f"{'='*70}")
            print("Fixing `torch.cuda` to `torch.accelerator`...")
            print(f"{'='*70}\n")

            for filename, issues in sorted(all_issues.items()):
                total_issues += len(issues)
                if apply_fixes(Path(filename), issues, verbose=args.verbose):
                    fixed_files += 1

            print(f"{'='*70}")
            print(f"✓ Fixed {total_issues} issue(s) across {fixed_files} file(s)")
            print(
                "  Please check that the applied changes "
                "do not break device-specific functionality"
            )
            print(f"{'='*70}")

        else:
            # Check mode: just report issues
            total_issues = print_issues(all_issues, verbose=args.verbose)
            print()
            print(f"{'='*70}")
            print("\nℹ️  torch.accelerator provides better device abstraction and portability")
            print("   compared to direct torch.cuda calls. It supports multiple backends")
            print("   (CUDA, XPU, MPS, etc.) with a unified API.")
            print()
            print(f"{'='*70}")

            if args.fail_on_issues:
                sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
