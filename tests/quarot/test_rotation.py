# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from llmcompressor.modifiers.transform.quarot.rotation import (
    make_hadamard_rotation,
    rotate_axis,
)


@pytest.mark.parametrize(
    "size,block,shifted",
    [(64, 32, False), (64, 32, True), (32, None, True), (64, None, False)],
)
def test_orthogonality_and_rng_isolation(size, block, shifted):
    before = torch.random.get_rng_state().clone()
    q = make_hadamard_rotation(
        size, block_size=block, shifted=shifted, dtype=torch.float64
    )
    torch.testing.assert_close(
        q.to_dense().T @ q.to_dense(),
        torch.eye(size, dtype=q.block.dtype),
        atol=1e-12,
        rtol=1e-12,
    )
    assert torch.equal(before, torch.random.get_rng_state())
    assert torch.equal(
        q.to_dense(),
        make_hadamard_rotation(
            size, block_size=block, shifted=shifted, dtype=q.block.dtype
        ).to_dense(),
    )


@pytest.mark.parametrize(
    "dtype,atol,rtol",
    [
        (torch.float64, 1e-12, 1e-12),
        (torch.float32, 3e-6, 3e-6),
        (torch.bfloat16, 0.08, 0.03),
    ],
)
def test_linear_pair_with_bias(dtype, atol, rtol):
    gen = torch.Generator().manual_seed(7)
    x = torch.randn(4, 32, generator=gen).to(dtype)
    w = torch.randn(16, 32, generator=gen).to(dtype)
    bias = torch.randn(16, generator=gen).to(dtype)
    q = make_hadamard_rotation(32, dtype=torch.float64)
    rotated = rotate_axis(x, q, axis=-1) @ rotate_axis(w, q, axis=1).T + bias
    torch.testing.assert_close(rotated, x @ w.T + bias, atol=atol, rtol=rtol)
    output_q = make_hadamard_rotation(16, dtype=torch.float64)
    out = x @ rotate_axis(w, output_q, axis=0).T + rotate_axis(bias, output_q, axis=0)
    torch.testing.assert_close(
        out, rotate_axis(x @ w.T + bias, output_q, axis=-1), atol=atol, rtol=rtol
    )


@pytest.mark.parametrize("axis", [0, 1])
def test_repeated_segment_matches_explicit_matrix(axis):
    gen = torch.Generator().manual_seed(11)
    value = torch.randn(40, 40, generator=gen, dtype=torch.float64)
    before = value.clone()
    q = make_hadamard_rotation(8, dtype=torch.float64)
    block = torch.block_diag(torch.eye(2, dtype=q.block.dtype), q.to_dense())
    full = torch.block_diag(*[block] * 4)
    expected = full.T @ value if axis == 0 else value @ full
    actual = rotate_axis(value, q, axis=axis, stride=10, offset=2)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    assert torch.equal(value, before)
    # Exact, not tolerance-based, protection of unrotated K/RoPE segments.
    assert torch.equal(
        actual.movedim(axis, -1).reshape(40, 4, 10)[..., :2],
        value.movedim(axis, -1).reshape(40, 4, 10)[..., :2],
    )


def test_noncontiguous_roundtrip():
    value = torch.arange(256, dtype=torch.float64).view(16, 16).T
    q = make_hadamard_rotation(8, dtype=torch.float64)
    result = rotate_axis(rotate_axis(value, q, axis=0), q.to_dense().T, axis=0)
    torch.testing.assert_close(result, value, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize(
    "shape,axis,stride,offset",
    [
        ((32, 192), 0, 32, 0),
        ((32, 192), 1, 48, 16),
        ((3, 32, 192), 1, 32, 0),
        ((32,), 0, 32, 0),
    ],
)
def test_rotation_uses_unbatched_gemm(shape, axis, stride, offset):
    """A transposed O projection must not broadcast Q over every input channel.

    Guard dispatch on small tensors instead of reproducing the server's OOM at
    [6144, 16384]. Also exercise repeated segments, higher ranks and a bias.
    """
    generator = torch.Generator().manual_seed(91)
    value = torch.randn(shape, dtype=torch.float64, generator=generator)
    original = value.clone()
    rotation = make_hadamard_rotation(32, dtype=torch.float64)
    block = torch.eye(stride, dtype=torch.float64)
    block[offset : offset + 32, offset : offset + 32] = rotation.to_dense()
    full = torch.block_diag(*[block] * (shape[axis] // stride))
    expected = torch.einsum("...j,jk->...k", value.movedim(axis, -1), full)
    expected = expected.movedim(-1, axis)

    class UnbatchedGemmOnly(TorchDispatchMode):
        def __init__(self):
            self.calls = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func in (torch.ops.aten.bmm.default, torch.ops.aten.baddbmm.default):
                raise AssertionError("Rotation must not batch/broadcast the matrix")
            if func is torch.ops.aten.mm.default:
                assert args[0].ndim == args[1].ndim == 2
                self.calls += 1
            return func(*args, **(kwargs or {}))

    with UnbatchedGemmOnly() as mode:
        result = rotate_axis(value, rotation, axis=axis, stride=stride, offset=offset)
    assert mode.calls == 1
    torch.testing.assert_close(result, expected, atol=1e-12, rtol=1e-12)
    assert torch.equal(value, original)


@pytest.mark.parametrize(
    "size,block", [(0, 32), (33, 32), (32, 0), (30, None), (32, 3)]
)
def test_invalid_construction(size, block):
    with pytest.raises(ValueError):
        make_hadamard_rotation(size, block_size=block)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"axis": 2},
        {"axis": 0, "stride": 7},
        {"axis": 1, "offset": -1},
        {"axis": 1, "offset": 1},
        {"axis": 0, "stride": 0},
    ],
)
def test_invalid_segment(kwargs):
    with pytest.raises(ValueError):
        rotate_axis(torch.ones(16, 16), torch.eye(8), **kwargs)
