# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Structured execution against the pre-optimization dense mathematical path."""

import json
import os
from pathlib import Path

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from llmcompressor.modifiers.transform.quarot.rotation import (
    make_hadamard_rotation,
    rotate_axis,
)


def dense_reference(size, block, shifted, dtype, seed=1234):
    """Freeze the old construction independently of the structured factory."""
    hadamard = torch.ones(1, 1, dtype=dtype)
    base = torch.tensor([[1, 1], [1, -1]], dtype=dtype)
    while hadamard.shape[0] < block:
        hadamard = torch.kron(hadamard, base)
    generator = torch.Generator().manual_seed(seed)
    signs = torch.randint(2, (block,), generator=generator) * 2 - 1
    signed = (
        signs.to(dtype).unsqueeze(1)
        * hadamard
        / torch.tensor(block, dtype=dtype).sqrt()
    )
    rotation = torch.kron(torch.eye(size // block, dtype=dtype), signed)
    if shifted:
        permutation = torch.eye(size, dtype=dtype).roll(16, dims=1)
        rotation = rotation @ permutation @ rotation
    return rotation


@pytest.fixture(scope="module")
def comparisons():
    records = []
    yield records
    if directory := os.environ.get("QUAROT_REPORT_DIR"):
        Path(directory, "structured-rotation.json").write_text(
            json.dumps(records, indent=2), encoding="utf-8"
        )


@pytest.mark.parametrize("size,block", [(32, 8), (96, 32), (128, 64)])
@pytest.mark.parametrize("shifted", [False, True])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("padding", [0, 7])
@pytest.mark.parametrize(
    "storage,precision",
    [
        (torch.float64, torch.float64),
        (torch.float32, torch.float32),
        (torch.bfloat16, torch.float32),
        (torch.float16, torch.float32),
    ],
)
def test_dense_equivalence(
    size, block, shifted, axis, padding, storage, precision, comparisons
):
    stride, offset = size + padding, 3 if padding else 0
    shape = (2 * stride, 11) if axis == 0 else (11, 2 * stride)
    value = torch.randn(shape, generator=torch.Generator().manual_seed(71)).to(storage)
    before = value.clone()
    rotation = make_hadamard_rotation(
        size, block_size=block, shifted=shifted, dtype=precision
    )
    dense = dense_reference(size, block, shifted, precision)
    torch.testing.assert_close(rotation.to_dense(), dense, atol=0, rtol=0)

    # Independent segment loop: x @ Q, or Q.T @ W for the output axis.
    expected_compute = value.to(precision).clone()
    for start in range(0, shape[axis], stride):
        selected = [slice(None), slice(None)]
        selected[axis] = slice(start + offset, start + offset + size)
        selected = tuple(selected)
        segment = value[selected].to(precision)
        transformed = dense.T @ segment if axis == 0 else segment @ dense
        expected_compute[selected] = transformed
    expected = expected_compute.to(storage)
    actual_compute = rotate_axis(
        value.to(precision),
        rotation,
        axis=axis,
        stride=stride,
        offset=offset,
        precision=precision,
    )
    compute_l2 = (
        (actual_compute.double() - expected_compute.double()).norm()
        / expected_compute.double().norm()
    ).item()
    assert compute_l2 <= (1e-5 if precision == torch.float32 else 2e-12)
    actual = rotate_axis(
        value, rotation, axis=axis, stride=stride, offset=offset, precision=precision
    )
    # Existing source-differential tolerances; BF16/FP16 allow one storage ULP.
    atol, rtol = {
        torch.float64: (2e-12, 2e-12),
        torch.float32: (2e-6, 2e-5),
        torch.bfloat16: (2e-6, 0.008),
        torch.float16: (2e-6, 0.001),
    }[storage]
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    delta = actual.double() - expected.double()
    relative_l2 = (delta.norm() / expected.double().norm()).item()
    # Low-precision storage can land on opposite sides of a midpoint. Keep the
    # strict pre-storage check above separate from a quarter-epsilon L2 bound.
    storage_limit = {torch.float64: 2e-12, torch.float32: 1e-5}.get(
        storage, torch.finfo(storage).eps / 4
    )
    assert relative_l2 <= storage_limit
    rows = actual.movedim(axis, -1).reshape(11, 2, stride)
    original = value.movedim(axis, -1).reshape(11, 2, stride)
    assert torch.equal(rows[..., :offset], original[..., :offset])
    assert torch.equal(rows[..., offset + size :], original[..., offset + size :])
    assert torch.equal(value, before)
    assert (actual.dtype, actual.device, actual.shape) == (
        value.dtype,
        value.device,
        value.shape,
    )
    comparisons.append(
        dict(
            size=size,
            block=block,
            shifted=shifted,
            axis=axis,
            stride=stride,
            offset=offset,
            storage=str(storage),
            precision=str(precision),
            max_abs=delta.abs().max().item(),
            relative_l2=relative_l2,
            compute_relative_l2=compute_l2,
        )
    )


@pytest.mark.parametrize("shifted", [False, True])
def test_large_rotation_never_materializes_dense_or_batched_gemm(shifted):
    """Guard the actual 6144/32 structure without timing-dependent assertions."""

    class BlockGemmOnly(TorchDispatchMode):
        calls = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func in (torch.ops.aten.bmm.default, torch.ops.aten.baddbmm.default):
                raise AssertionError("Rotation must not broadcast a batched matrix")
            if func is torch.ops.aten.mm.default:
                assert args[0].ndim == 2 and args[1].shape == (32, 32)
                self.calls += 1
            result = func(*args, **(kwargs or {}))
            if isinstance(result, torch.Tensor):
                assert result.numel() <= 4 * 6144, "Unexpected dense allocation"
            return result

    value = torch.randn(4, 6144)
    with BlockGemmOnly() as mode:
        rotation = make_hadamard_rotation(6144, block_size=32, shifted=shifted)
        actual = rotate_axis(value, rotation, axis=1, precision=torch.float32)
    assert rotation.block.numel() == 32 * 32
    assert mode.calls == (2 if shifted else 1)
    assert actual.shape == value.shape
