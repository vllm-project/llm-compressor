# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from itertools import cycle
from typing import Callable, Dict, Optional

import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

from llmcompressor.pytorch.utils import tensors_module_forward, tensors_to_device

__all__ = ["apply_pad_mask_to_batch", "run_calibration_forward"]


def apply_pad_mask_to_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Apply a mask to the input ids of a batch. This is used to zero out
    padding tokens so they do not contribute to the hessian calculation in the
    SparseGPT algorithm

    :param batch: batch to apply padding to if it exists
    :return: batch with padding zeroed out in the input_ids
    """
    batch["input_ids"] = batch["input_ids"] * batch["attention_mask"]
    return batch


def run_calibration_forward(
    model: Module,
    calibration_dataloader: DataLoader,
    num_calibration_steps: Optional[int] = None,
    calibration_function: Optional[Callable] = None,
    device: Optional[str] = None,
    mask_padding: bool = False,
):
    """
    Helper function used by one-shot modifiers, runs calibration data through a model to
    update modifier statistics and trigger hooks

    :param model: PyTorch model to run
    :param calibration_dataloader: data to use for calibration
    :param num_calibration_steps: number of items in calibration_dataloader to process,
    None or a negative number to process all available data
    :param calibration_function: option to pass a custom forward function for model
    :param device: option to move the model to a specific device before calibration
    :param mask_padding: whether to zero out padding tokens during calibration
    """
    model.eval()

    forward_fn: Callable = (
        calibration_function if calibration_function else tensors_module_forward
    )

    # move model to optional specified device if it is not already there
    model_device = next(model.parameters()).device
    if device is not None and model_device != device:
        model.to(device)
        model_device = next(model.parameters()).device
    _dataloader = (
        calibration_dataloader
        if num_calibration_steps is None
        else cycle(calibration_dataloader)
    )

    # run through the calibration data
    for batch_idx, batch in enumerate(tqdm(_dataloader)):
        if num_calibration_steps and batch_idx >= num_calibration_steps:
            break
        if mask_padding:
            batch = apply_pad_mask_to_batch(batch)
        batch = tensors_to_device(batch, model_device)
        with torch.no_grad():
            forward_fn(batch, module=model)
