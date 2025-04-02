from typing import TYPE_CHECKING

import torch
import torch.utils.data.dataloader
import tqdm
from loguru import logger
from transformers import PreTrainedModel

from llmcompressor.core import get_compressor
from llmcompressor.pytorch.utils.helpers import tensors_to_device
from llmcompressor.utils import apply_pad_mask_to_batch
from llmcompressor.utils.helpers import calibration_forward_context

if TYPE_CHECKING:
    from llmcompressor.args import PostTrainArguments

__all__ = ["run_pipeline"]


def run_pipeline(
    model: PreTrainedModel,
    dataloader: torch.utils.data.DataLoader,
    args: "PostTrainArguments",
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
    compressor = get_compressor()

    if args.oneshot_device is not None:
        logger.warning(
            "Basic pipeline does not utilize `oneshot_device` argument, instead use "
            "`from_pretrained(device_map=...)` to determine onloading behavior"
        )

    compressor.initialize()
    with calibration_forward_context(model):
        for batch in tqdm.tqdm(dataloader, desc="Calibrating"):
            batch = apply_pad_mask_to_batch(batch)
            batch = tensors_to_device(batch, model.device)
            model(**batch)

    compressor.calibration_epoch_end()
