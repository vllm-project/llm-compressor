import torch
import torch.utils.data.dataloader
import tqdm
from compressed_tensors.utils import get_execution_device

from llmcompressor.core import LifecycleCallbacks, active_session
from llmcompressor.modifiers.utils.pytorch_helpers import apply_pad_mask_to_batch
from llmcompressor.pytorch.utils.helpers import tensors_to_device
from llmcompressor.utils.helpers import calibration_forward_context
from llmcompressor.core.llmcompressor.globals import get_compressor

__all__ = ["run_pipeline"]


def run_pipeline(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
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
    try:
        compressor = get_compressor()
    except:
        session = active_session()
        compressor = None
    
    model_device = get_execution_device(model)

    if compressor is not None:
        compressor.initialize()
    else:
        session.initialize()

    with calibration_forward_context(model):
        for batch in tqdm.tqdm(dataloader, desc="Calibrating"):
            batch = apply_pad_mask_to_batch(batch)
            batch = tensors_to_device(batch, model_device)
            model(**batch)

    if compressor is not None:
        compressor.calibration_epoch_end()
    else:
        LifecycleCallbacks.calibration_epoch_end()
