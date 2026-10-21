# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Test-only CPU plan executor. Does not implement the Modifier lifecycle."""

import torch

from llmcompressor.modeling import fuse_norm_linears
from llmcompressor.modifiers.transform.quarot.rotation import (
    make_hadamard_rotation,
    rotate_axis,
)


def matrices_for(plan, block_size=32, dtype=torch.float32):
    return {
        space.name: make_hadamard_rotation(
            space.size, block_size=block_size, shifted=space.shifted, dtype=dtype
        )
        for space in plan.spaces
    }


@torch.no_grad()
def fuse_plan(model, plan, precision):
    for item in plan.fusions:
        fuse_norm_linears(
            model.get_submodule(item.norm),
            [model.get_submodule(name) for name in item.consumers],
            precision=precision,
        )


@torch.no_grad()
def execute(model, operations, matrices, precision):
    # Test-only on-device tensors; production will use offload parameter updates.
    for op in operations:
        module = model.get_submodule(op.target)
        kwargs = dict(
            axis=op.axis, stride=op.stride, offset=op.offset, precision=precision
        )
        module.weight.copy_(rotate_axis(module.weight, matrices[op.space], **kwargs))
        if op.axis == 0 and getattr(module, "bias", None) is not None:
            module.bias.copy_(rotate_axis(module.bias, matrices[op.space], **kwargs))
