import torch
import torch.utils.data.dataloader

__all__ = ["run_pipeline"]


def run_pipeline(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader):
    """
    A dummy pipeline for skipping calibration
    """
    pass
