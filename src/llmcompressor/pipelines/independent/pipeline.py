from typing import TYPE_CHECKING

import torch
from loguru import logger
from torch.utils.data.dataloader import DataLoader

from llmcompressor.core import get_compressor

if TYPE_CHECKING:
    from llmcompressor.args import PostTrainArguments

__all__ = ["run_pipeline"]


def run_pipeline(
    model: torch.nn.Module,
    dataloader: DataLoader,
    args: "PostTrainArguments",
):
    # avoid circular import
    from llmcompressor.pipelines.registry import get_pipeline_fn

    compressor = get_compressor()
    modifiers = compressor.modifiers

    for index, modifier in enumerate(modifiers):
        mod_type = str(type(modifier).__name__)
        compressor.modifiers = [modifier]

        pipeline, pipeline_fn = get_pipeline_fn(user=None, modifiers=[modifier])
        logger.info(f"Inferred `{pipeline}` calibration pipeline for `{mod_type}`")

        pipeline_fn(model, dataloader, args)

    compressor.modifiers = modifiers
