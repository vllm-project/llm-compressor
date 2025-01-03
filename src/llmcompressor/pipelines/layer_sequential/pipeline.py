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
    sequential_targets: List[str],
    dataloader: torch.utils.data.DataLoader,
):
    """
    Run a layer-wise sequential data pipeline.
    1. Layers are identified according to `sequential_targets`
    2. A hook is attached to the first layer. This hook raises an exception which is
        then caught and used to capture the input arguments to the first layer
    3. The inputs to the first layer are used to calibrate the first layer, and the
        output of the previous layer is used as inputs to calibrate the next layer

    This pipeline requires that the model have distinct layers defined in its
    architecture and that the outputs of the previous layer are exactly the inputs
    to the next layer. This is violated by encoder-decoder architectures among others.

    If your model architecture violates these assumptions, consider using the sequential
    pipeline (see llmcompressor.pipelines.sequential)
    """
    # find layers
    layers = match_modules(model, sequential_targets)

    with calibration_forward_context(model):
        # prepare intermediates cache
        intermediates = capture_first_layer_intermediates(model, layers, dataloader)

        num_layers = len(layers)
        for layer_index, layer in enumerate(layers):
            # prepare tqdm description texts
            calib_desc = f"({layer_index + 1}/{num_layers}): Calibrating"
            prop_desc = f"({layer_index + 1}/{num_layers}): Propagating"

            # do an preliminary pass to trigger modifier hooks
            for batch_index in tqdm.tqdm(range(len(dataloader)), desc=calib_desc):
                inputs = intermediates.fetch(batch_index)
                layer(**inputs)

            # this pass does not trigger modifier hooks
            # and is only used for capturing outputs from the newly compressed modules
            with HooksMixin.disable_hooks():
                for batch_index in tqdm.tqdm(range(len(dataloader)), desc=prop_desc):
                    inputs = intermediates.fetch(batch_index)
                    output = layer(**inputs)

                    if layer_index < num_layers - 1:
                        output = to_next_layer_kwargs(output, layers[layer_index + 1])
                        intermediates.delete(batch_index)
                        intermediates.update(batch_index, output)
