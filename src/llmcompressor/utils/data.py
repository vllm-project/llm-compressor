from typing import Dict

import torch

__all__ = [
    "apply_pad_mask_to_batch",
]


def apply_pad_mask_to_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Apply a mask to the input ids of a batch. This is used to zero out
    padding tokens so they do not contribute to the hessian calculation in the
    GPTQ and SparseGPT algorithms

    Assumes that `attention_mask` only contains zeros and ones

    :param batch: batch to apply padding to if it exists
    :return: batch with padding zeroed out in the input_ids
    """
    if "attention_mask" in batch:
        for key in ("input_ids", "decoder_input_ids"):
            if key in batch:
                batch[key] = batch[key] * batch["attention_mask"]

    return batch
