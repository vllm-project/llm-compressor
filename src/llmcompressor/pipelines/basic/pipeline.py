import torch
import torch.utils.data.dataloader
import tqdm

from llmcompressor.modifiers.utils.pytorch_helpers import apply_pad_mask_to_batch
from llmcompressor.pytorch.utils.helpers import tensors_to_device
from llmcompressor.utils.helpers import calibration_forward_context

__all__ = ["run_pipeline"]


def run_pipeline(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
):
    # TODO: revisit
    device_map = getattr(model, "hf_device_map", None)
    if device_map is not None:
        model_device = next(iter(device_map.values()))
    else:
        model_device = model.device

    with calibration_forward_context(model):
        for batch in tqdm.tqdm(dataloader, desc="Calibrating"):
            batch = apply_pad_mask_to_batch(batch)
            batch = tensors_to_device(batch, model_device)
            model(**batch)
