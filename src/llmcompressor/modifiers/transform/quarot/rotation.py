# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Pure tensor operations for offline, repeated-segment orthogonal transforms.

Weights follow PyTorch's [output, input] convention. Rotating axis 0 applies
Q.T @ W; rotating axis 1 applies W @ Q. The same routine handles output biases.
Parameter mutation, norm fusion and offloading belong to the Modifier.
"""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class HadamardRotation:
    """Repeated signed block B, or B P B with a 16-column cyclic permutation.

    Only the small signed block is stored. ``to_dense`` is an explicit diagnostic
    conversion; the weight executor never materializes the full rotation.
    """

    size: int
    block: Tensor
    shifted: bool = False

    @property
    def block_size(self) -> int:
        return self.block.shape[0]

    def apply(self, value: Tensor) -> Tensor:
        """Right-multiply rows in the caller's arithmetic dtype and device."""
        block = self.block.to(device=value.device, dtype=value.dtype)
        rotated = value.reshape(-1, self.block_size) @ block
        if self.shifted:
            # x @ eye(size).roll(16, dims=1) == x.roll(16, dims=-1).
            # Roll across the entire rotation space, including block boundaries.
            rotated = rotated.reshape(-1, self.size).roll(16, dims=-1)
            rotated = rotated.reshape(-1, self.block_size) @ block
        return rotated.reshape(value.shape)

    def to_dense(self) -> Tensor:
        """Materialize the original matrix for explicit reference comparisons."""
        identity = torch.eye(
            self.size // self.block_size,
            dtype=self.block.dtype,
            device=self.block.device,
        )
        rotation = torch.kron(identity, self.block)
        if self.shifted:
            permutation = torch.eye(
                self.size, dtype=self.block.dtype, device=self.block.device
            ).roll(16, dims=1)
            rotation = rotation @ permutation @ rotation
        return rotation


def make_hadamard_rotation(
    size: int,
    *,
    block_size: int | None = None,
    shifted: bool = False,
    seed: int = 1234,
    dtype: torch.dtype = torch.float32,
) -> HadamardRotation:
    """Construct a seeded structured rotation without changing global RNG state.

    Initial scope is power-of-two blocks. For shifted rotations, construct B P B,
    where B repeats a signed Hadamard and P rolls identity columns by 16. A full
    non-power-of-two Hadamard basis requires a separate implementation.
    """
    block = (32 if shifted else size) if block_size is None else block_size
    if size <= 0 or block <= 0 or block & (block - 1) or size % block:
        raise ValueError("size must be positive and divisible by a power-of-two block")
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("Construct rotations in float32 or float64")

    # Sylvester construction, independently expressed as a Kronecker product.
    hadamard = torch.ones(1, 1, dtype=dtype, device="cpu")
    base = torch.tensor([[1, 1], [1, -1]], dtype=dtype, device="cpu")
    while hadamard.shape[0] < block:
        hadamard = torch.kron(hadamard, base)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    signs = torch.randint(2, (block,), generator=generator, device="cpu") * 2 - 1
    # FP32 normalization also reproduces the reference's scalar sqrt rounding.
    normalizer = torch.tensor(block, dtype=dtype, device="cpu").sqrt()
    signed = signs.to(dtype).unsqueeze(1) * hadamard / normalizer
    return HadamardRotation(size=size, block=signed, shifted=shifted)


def rotate_axis(
    value: Tensor,
    rotation: HadamardRotation | Tensor,
    *,
    axis: int,
    stride: int | None = None,
    offset: int = 0,
    precision: torch.dtype = torch.float64,
) -> Tensor:
    """Apply Q to a contiguous segment within every stride along an axis.

    With W shaped [h*(K+V), C], axis=0, stride=K+V, offset=K and Q of
    dimension V applies only the V-output rotation in every head. Unselected
    elements are copied exactly; a large block-diagonal matrix is never formed.
    The input tensor is unchanged. The output preserves its dtype and device.
    """
    if isinstance(rotation, HadamardRotation):
        size, coefficients = rotation.size, rotation.block
    else:
        if rotation.ndim != 2 or rotation.shape[0] != rotation.shape[1]:
            raise ValueError("rotation must be a square matrix")
        size, coefficients = rotation.shape[0], rotation
    if not value.is_floating_point() or not coefficients.is_floating_point():
        raise ValueError("value and rotation must be floating point")
    if precision not in (torch.float32, torch.float64):
        raise ValueError("Compute rotations in float32 or float64")
    if not -value.ndim <= axis < value.ndim:
        raise ValueError("axis is outside the tensor dimensions")
    stride = size if stride is None else stride
    if size <= 0 or stride <= 0 or offset < 0 or offset + size > stride:
        raise ValueError("rotation segment must fit within a positive stride")
    if value.shape[axis] == 0 or value.shape[axis] % stride:
        raise ValueError("stride must divide the selected tensor dimension")

    rows = value.movedim(axis, -1)
    grouped = rows.reshape(*rows.shape[:-1], rows.shape[-1] // stride, stride)
    result = grouped.clone()
    segment = grouped[..., offset : offset + size].to(precision)
    if isinstance(rotation, HadamardRotation):
        rotated = rotation.apply(segment)
    else:
        # Dense compatibility path also uses unbatched GEMM, avoiding broadcast
        # allocations when the selected axis is noncontiguous.
        matrix = rotation.to(device=value.device, dtype=precision)
        rotated = segment.reshape(-1, size) @ matrix
    result[..., offset : offset + size] = rotated.reshape(segment.shape).to(value.dtype)
    return result.reshape(rows.shape).movedim(-1, axis)
