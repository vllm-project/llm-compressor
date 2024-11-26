import torch

from llmcompressor.utils.helpers import calibration_forward_context


def run_pipeline(
    model: torch.nn.Module
):
    with calibration_forward_context(model):
        pass