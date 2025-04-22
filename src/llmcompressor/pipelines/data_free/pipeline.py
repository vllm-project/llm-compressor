import torch
from torch.utils.data.dataloader import DataLoader

from llmcompressor.core.session_functions import LifecycleCallbacks, active_session

__all__ = ["run_pipeline"]


def run_pipeline(
    model: torch.nn.Module,
    dataloader: DataLoader,
):
    """
    A pipeline for data-free calibration
    """
    session = active_session()
    session.initialize()
    LifecycleCallbacks.calibration_epoch_end()
