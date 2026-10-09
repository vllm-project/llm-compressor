"""
StreamingPipeline: meta-device streaming calibration pipeline.

Architecture overview (three-layer pipeline):

    Disk bg:    [read sg_N+1 ──────────]
    H2D stream:                   [DMA sg_N+1 ──]
    GPU:        [compute sg_N          ]  [wait≈0ms]  [compute sg_N+1 ────]

Each subgraph loop:
    prefetch(next, async) → apply(allowed_modules) → calibrate → compress
    → stream_write → decompress(leave_decompressed=False) → release → meta

Dependencies not yet merged:
    - PR #3111 (@Isotr0py): base meta-device architecture (CheckpointMap, SubgraphPrefetcher)
    - PR #3208 (@Roderick-Wu): load_quantizable_moe() for MoE support
    - PR #2995 + CT PR #811 (@kylesayrs): allowed_modules + leave_decompressed primitives
    - Issue #3131: stream_write (incremental safetensors shard writing)

This file is a discussion placeholder. See Issue #3132 for benchmark results.
"""

from typing import TYPE_CHECKING

import torch
from torch.utils.data.dataloader import DataLoader

from llmcompressor.pipelines.registry import CalibrationPipeline

if TYPE_CHECKING:
    from llmcompressor.args.dataset_arguments import DatasetArguments


class StreamingPipeline(CalibrationPipeline):
    """
    Meta-device streaming pipeline with async subgraph prefetch.

    Not yet implemented — opened as a discussion Draft PR.
    Benchmark results and architecture description: Issue #3132.
    """

    @staticmethod
    def __call__(
        model: torch.nn.Module,
        dataloader: DataLoader,
        dataset_args: "DatasetArguments",
    ):
        raise NotImplementedError(
            "StreamingPipeline is not yet implemented. "
            "This file is a discussion placeholder for Issue #3132."
        )
