#!/usr/bin/env python3
"""
Test suite for the torch.cuda linter.

This module contains tests to verify the linter correctly detects
torch.cuda usage and suggests torch.accelerator alternatives.
"""

import ast
import sys
import tempfile
from pathlib import Path

from tools.lint_cuda import TorchCudaLinter, apply_fixes, lint_file


def test_detect_torch_cuda_is_available():
    """Test detection of torch.cuda.is_available()."""
    code = """
import torch

if torch.cuda.is_available():
    print("CUDA available")
"""
    tree = ast.parse(code)
    linter = TorchCudaLinter("test.py")
    linter.visit(tree)

    assert len(linter.issues) == 1
    line, col, usage, suggestion = linter.issues[0]
    assert usage == "torch.cuda.is_available"
    assert suggestion == "torch.accelerator.is_available"
    print("✓ test_detect_torch_cuda_is_available passed")


def test_detect_torch_cuda_stream():
    """Test detection of torch.cuda.Stream."""
    code = """
import torch

stream = torch.cuda.Stream()
"""
    tree = ast.parse(code)
    linter = TorchCudaLinter("test.py")
    linter.visit(tree)

    assert len(linter.issues) == 1
    line, col, usage, suggestion = linter.issues[0]
    assert usage == "torch.cuda.Stream"
    assert suggestion == "torch.Stream"
    print("✓ test_detect_torch_cuda_stream passed")


def test_detect_torch_cuda_event():
    """Test detection of torch.cuda.Event."""
    code = """
import torch

event = torch.cuda.Event()
"""
    tree = ast.parse(code)
    linter = TorchCudaLinter("test.py")
    linter.visit(tree)

    assert len(linter.issues) == 1
    line, col, usage, suggestion = linter.issues[0]
    assert usage == "torch.cuda.Event"
    assert suggestion == "torch.Event"
    print("✓ test_detect_torch_cuda_event passed")


def test_detect_torch_cuda_context_manager():
    """Test detection of torch.cuda.stream context manager."""
    code = """
import torch

with torch.cuda.stream(stream):
    x = torch.randn(10)
"""
    tree = ast.parse(code)
    linter = TorchCudaLinter("test.py")
    linter.visit(tree)

    assert len(linter.issues) == 1
    line, col, usage, suggestion = linter.issues[0]
    assert usage == "torch.cuda.stream"
    # Generic replacement since torch.cuda.stream is not in the specific suggestions
    assert suggestion == "torch.Stream"
    print("✓ test_detect_torch_cuda_context_manager passed")


def test_detect_memory_functions():
    """Test detection of memory-related CUDA functions."""
    code = """
import torch

allocated = torch.cuda.memory_allocated()
reserved = torch.cuda.memory_reserved()
torch.cuda.empty_cache()
"""
    tree = ast.parse(code)
    linter = TorchCudaLinter("test.py")
    linter.visit(tree)

    assert len(linter.issues) == 3
    usages = [usage for _, _, usage, _ in linter.issues]
    assert "torch.cuda.memory_allocated" in usages
    assert "torch.cuda.memory_reserved" in usages
    assert "torch.cuda.empty_cache" in usages
    print("✓ test_detect_memory_functions passed")


def test_detect_from_import():
    """Test detection of 'from torch.cuda import ...' statements."""
    code = """
from torch.cuda import Stream, Event
"""
    tree = ast.parse(code)
    linter = TorchCudaLinter("test.py")
    linter.visit(tree)

    assert len(linter.issues) == 2
    print("✓ test_detect_from_import passed")


def test_ignore_non_cuda_torch():
    """Test that non-CUDA torch calls are not flagged."""
    code = """
import torch

x = torch.randn(10)
y = torch.tensor([1, 2, 3])
device = torch.device("cpu")
"""
    tree = ast.parse(code)
    linter = TorchCudaLinter("test.py")
    linter.visit(tree)

    assert len(linter.issues) == 0
    print("✓ test_ignore_non_cuda_torch passed")


def test_complex_usage():
    """Test detection in complex code with multiple CUDA calls."""
    code = """
import torch

def setup_cuda():
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        torch.cuda.set_device(0)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            torch.cuda.synchronize()
"""
    tree = ast.parse(code)
    linter = TorchCudaLinter("test.py")
    linter.visit(tree)

    assert len(linter.issues) == 6
    print("✓ test_complex_usage passed")


def test_lint_file_integration():
    """Test the lint_file function with a temporary file."""
    code = """
import torch

if torch.cuda.is_available():
    stream = torch.cuda.Stream()
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        temp_path = Path(f.name)

    try:
        issues = lint_file(temp_path)
        assert len(issues) == 2
        print("✓ test_lint_file_integration passed")
    finally:
        temp_path.unlink()


def test_no_duplicate_reports():
    """Test that the same position is not reported multiple times."""
    code = """
import torch

if torch.cuda.is_available():
    pass
"""
    tree = ast.parse(code)
    linter = TorchCudaLinter("test.py")
    linter.visit(tree)

    # Should only report once, not for both torch.cuda and torch.cuda.is_available
    # at overlapping positions
    positions = [(line, col) for line, col, _, _ in linter.issues]
    assert len(positions) == len(set(positions)), "Duplicate positions found"
    print("✓ test_no_duplicate_reports passed")


def test_apply_fixes():
    """Test that apply_fixes correctly replaces torch.cuda with torch.accelerator."""
    code = """import torch

def check_device():
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        current = torch.cuda.current_device()
        torch.cuda.synchronize()
        return True
    return False
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        temp_path = Path(f.name)

    try:
        # First lint to get issues
        issues = lint_file(temp_path)
        assert len(issues) > 0, "Should find issues"

        # Apply fixes
        result = apply_fixes(temp_path, issues)
        assert result is True, "Fixes should be applied"

        # Read the fixed content
        with open(temp_path, "r") as f:
            fixed_code = f.read()

        # Verify torch.cuda is replaced with torch.accelerator
        assert "torch.cuda" not in fixed_code, "torch.cuda should be replaced"
        assert "torch.accelerator" in fixed_code, "torch.accelerator should be present"

        # Verify no more issues after fix
        issues_after = lint_file(temp_path)
        assert len(issues_after) == 0, "Should have no issues after fix"

        print("✓ test_apply_fixes passed")
    finally:
        temp_path.unlink()


def run_all_tests():
    """Run all test functions."""
    print("Running torch.cuda linter tests...\n")

    tests = [
        test_detect_torch_cuda_is_available,
        test_detect_torch_cuda_stream,
        test_detect_torch_cuda_event,
        test_detect_torch_cuda_context_manager,
        test_detect_memory_functions,
        test_detect_from_import,
        test_ignore_non_cuda_torch,
        test_complex_usage,
        test_lint_file_integration,
        test_no_duplicate_reports,
        test_apply_fixes,
    ]

    failed = []
    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed.append(test.__name__)
        except Exception as e:
            print(f"✗ {test.__name__} errored: {e}")
            failed.append(test.__name__)

    print(f"\n{'='*60}")
    print(f"Tests passed: {len(tests) - len(failed)}/{len(tests)}")
    if failed:
        print(f"Failed tests: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("All tests passed! ✓")
        sys.exit(0)


if __name__ == "__main__":
    run_all_tests()
