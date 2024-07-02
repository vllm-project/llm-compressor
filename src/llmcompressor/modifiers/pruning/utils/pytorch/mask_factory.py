import re
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import Tensor
from torch.nn.parameter import Parameter

__all__ = [
    "PruningMaskCreatorArgs",
    "MaskCreatorType",
    "CreateMaskCreatorType",
    "PruningMaskFactory",
    "unstructured_pruning",
    "channel_pruning",
    "filter_pruning",
    "block_pruning",
]


@dataclass
class PruningMaskCreatorArgs:
    parameter: Parameter
    sparsity: float
    scores: Tensor
    prev_mask: Optional[Tensor] = None


MaskCreatorType = Callable[[PruningMaskCreatorArgs], Tensor]
CreateMaskCreatorType = Callable[[str], MaskCreatorType]


class PruningMaskFactory:
    registry = {}

    @staticmethod
    def register(name: str, func: CreateMaskCreatorType):
        PruningMaskFactory.registry[name] = func

    @staticmethod
    def register_decorator(name: str):
        def inner(func: CreateMaskCreatorType):
            PruningMaskFactory.registry[name] = func
            return func

        return inner

    @staticmethod
    def create_mask_creator(mask_structure: str, **kwargs) -> MaskCreatorType:
        for pattern, creator in PruningMaskFactory.registry.items():
            if pattern == mask_structure:
                return creator(mask_structure=mask_structure, **kwargs)

            try:
                if re.match(pattern, mask_structure):
                    return creator(mask_structure=mask_structure, **kwargs)
            except Exception:
                pass

        raise ValueError(f"Invalid mask_structure: {mask_structure}")


@PruningMaskFactory.register_decorator("unstructured")
def unstructured_pruning(mask_structure: str):
    if mask_structure != "unstructured":
        raise ValueError(f"Invalid mask_structure: {mask_structure}")

    def _create_mask(args: PruningMaskCreatorArgs) -> Tensor:
        prune_elements = int(args.sparsity * args.scores.numel())
        mask = (
            args.prev_mask
            if args.prev_mask is not None
            else torch.ones_like(args.parameter.data, dtype=torch.bool)
        )

        if prune_elements > 0:
            threshold, _ = torch.topk(
                args.scores.view(-1), prune_elements, largest=False
            )
            mask = (args.scores > threshold[-1]).to(dtype=torch.bool)
        else:
            mask = torch.ones_like(mask, dtype=torch.bool)

        return mask

    return _create_mask


@PruningMaskFactory.register_decorator("channel")
def channel_pruning(mask_structure: str, aggregate: str = "sum"):
    if mask_structure != "channel":
        raise ValueError(f"Invalid mask_structure: {mask_structure}")

    def _aggregate(tensor, method="sum"):
        return getattr(tensor, method)(dim=(1, 2, 3))

    def _create_mask(args: PruningMaskCreatorArgs) -> Tensor:
        prune_channels = int(args.sparsity * args.scores.size(0))
        aggregated_scores = _aggregate(args.scores, aggregate)
        _, top_indices = torch.topk(aggregated_scores, prune_channels, largest=False)
        mask = torch.ones_like(args.scores, dtype=torch.bool)
        mask[top_indices, :, :, :] = 0
        return mask

    return _create_mask


@PruningMaskFactory.register_decorator("filter")
def filter_pruning(mask_structure: str, aggregate: str = "sum"):
    if mask_structure != "filter":
        raise ValueError(f"Invalid mask_structure: {mask_structure}")

    def _aggregate(tensor, method="sum"):
        return getattr(tensor, method)(dim=(0, 2, 3))

    def _create_mask(args: PruningMaskCreatorArgs) -> Tensor:
        prune_filters = int(args.sparsity * args.scores.size(1))
        aggregated_scores = _aggregate(args.scores, aggregate)
        _, top_indices = torch.topk(aggregated_scores, prune_filters, largest=False)
        mask = torch.ones_like(args.scores, dtype=torch.bool)
        mask[:, top_indices, :, :] = 0
        return mask

    return _create_mask


@PruningMaskFactory.register_decorator("^block_.*")
def block_pruning(mask_structure: str, aggregate: str = "sum"):
    pattern = re.compile(r"^block_(.*)")
    match = pattern.search(mask_structure)

    if not match:
        raise ValueError(f"invalid block mask type {mask_structure}")

    block_dims = list(map(int, match.group(1).split(",")))

    def _aggregate_block(block, method="sum"):
        return getattr(block, method)()

    def _create_mask(args: PruningMaskCreatorArgs) -> Tensor:
        block_view = args.scores
        for dim, size in enumerate(block_dims):
            block_view = block_view.unfold(dimension=dim, size=size, step=size)
        block_sums = _aggregate_block(block_view, aggregate)
        prune_blocks = int(args.sparsity * block_sums.numel())
        threshold, _ = torch.topk(block_sums.view(-1), prune_blocks, largest=False)
        mask = (block_sums > threshold[-1]).float().unsqueeze(-1)
        for size in block_dims:
            mask = mask.repeat_interleave(size, dim=-1)
        return mask.to(dtype=torch.bool)

    return _create_mask
