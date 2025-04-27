from typing import TYPE_CHECKING

import torch
import tqdm
from compressed_tensors.utils import get_execution_device
from torch.utils.data.dataloader import DataLoader

from llmcompressor.core import LifecycleCallbacks, active_session
from llmcompressor.modifiers.utils.pytorch_helpers import apply_pad_mask_to_batch
from llmcompressor.pytorch.utils.helpers import tensors_to_device
from llmcompressor.utils.helpers import calibration_forward_context

if TYPE_CHECKING:
    from llmcompressor.args.dataset_arguments import DatasetArguments

__all__ = ["run_pipeline"]


def run_pipeline(
    model: torch.nn.Module,
    dataloader: DataLoader,
    dataset_args: "DatasetArguments",
):
    """
    Run a basic data pipeline.

    Batches are fetched from the data loader and are used to perform forward passes
    through the model. This pipeline is typically used for basic model calibration
    and, unlike the sequential pipelines, does not propagate compression error when
    used to calibrate model compression

    :param model: model being calibrated
    :param dataloader: loads data for calibration
    :param modifiers: list of modifiers, only included to match PipelineFn signature
    """
    session = active_session()
    model_device = get_execution_device(model)

    session.initialize()

    with calibration_forward_context(model):
        for batch in tqdm.tqdm(dataloader, desc="Calibrating"):
            batch = apply_pad_mask_to_batch(batch)
            batch = tensors_to_device(batch, model_device)
            model(**batch)

    LifecycleCallbacks.calibration_epoch_end()
