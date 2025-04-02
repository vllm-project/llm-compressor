from typing import TYPE_CHECKING

import torch
import torch.utils.data.dataloader
from llmcompressor.core import get_compressor

if TYPE_CHECKING:
    from llmcompressor.args import PostTrainArguments

__all__ = ["run_pipeline"]


def run_pipeline(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, args: "PostTrainArguments"):
    """
    A pipeline for skipping calibration
    """
    compressor = get_compressor()
    compressor.initialize()
    compressor.calibration_epoch_end()
