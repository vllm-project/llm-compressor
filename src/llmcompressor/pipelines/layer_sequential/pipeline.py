from contextlib import nullcontext
from typing import List

import torch
import torch.utils.data.dataloader
import tqdm

from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.pipelines.layer_sequential.helpers import (
    capture_first_layer_intermediates,
    match_modules,
    to_next_layer_kwargs,
)
from llmcompressor.utils.helpers import calibration_forward_context

__all__ = ["run_pipeline"]


def run_pipeline(
    model: torch.nn.Module,
    sequential_targets: List[str],  # FUTURE: replace with recipe inference
    dataloader: torch.utils.data.DataLoader,
    propagate_error: bool,
):
    """ """
    # find layers
    layers = match_modules(model, sequential_targets)

    # FUTURE: apply recipe to model
    # initialize(recipe, model)

    with calibration_forward_context(model):
        intermediates = capture_first_layer_intermediates(model, layers, dataloader)

        num_layers = len(layers)
        for layer_index, layer in enumerate(layers):
            # prepare tqdm description texts
            calib_desc = f"({layer_index + 1}/{num_layers}): Calibrating"
            prop_desc = f"({layer_index + 1}/{num_layers}): Propagate"

            if propagate_error:
                # do an preliminary pass to trigger modifier hooks
                for batch_index in tqdm.tqdm(range(len(dataloader)), desc=calib_desc):
                    inputs = intermediates.fetch(batch_index)
                    layer(**inputs)

            # if using propagate_error, then this pass does not trigger modifier hooks
            # and is only used for capturing intermediates
            # otherwise, this pass triggers modifier hooks and captures intermediates
            with HooksMixin.disable_hooks() if propagate_error else nullcontext():
                desc = prop_desc if propagate_error else calib_desc
                for batch_index in tqdm.tqdm(range(len(dataloader)), desc=desc):
                    inputs = intermediates.fetch(batch_index)
                    output = layer(**inputs)
                    output = to_next_layer_kwargs(output, layers[layer_index + 1])

                    if layer_index < num_layers - 1:
                        intermediates.delete(batch_index)
                        intermediates.update(batch_index, output)
