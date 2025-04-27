from typing import TYPE_CHECKING

import torch
from torch.utils.data.dataloader import DataLoader

from llmcompressor.core.session_functions import LifecycleCallbacks, active_session

if TYPE_CHECKING:
    from llmcompressor.args.dataset_arguments import DatasetArguments

__all__ = ["run_pipeline"]


def run_pipeline(
    model: torch.nn.Module,
    dataloader: DataLoader,
    dataset_args: "DatasetArguments",
):
    """
    A pipeline for data-free calibration
    """
    session = active_session()
    session.initialize()
    LifecycleCallbacks.calibration_epoch_end()
