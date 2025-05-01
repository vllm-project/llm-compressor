from typing import TYPE_CHECKING

import torch
from torch.utils.data.dataloader import DataLoader

from llmcompressor.core.session_functions import LifecycleCallbacks

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

    :param model: model being calibrated
    :param dataloader: loads data for calibration
    :param dataset_args: dataset arguments relevant to pipelines
    """
    LifecycleCallbacks.calibration_epoch_start()
    LifecycleCallbacks.calibration_epoch_end()
