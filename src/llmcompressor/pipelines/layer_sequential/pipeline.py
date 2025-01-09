from typing import TYPE_CHECKING, List, Optional

import torch
import torch.utils.data.dataloader
import tqdm

from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.pipelines.cache import IntermediatesCache
from llmcompressor.pipelines.layer_sequential.helpers import (
    capture_first_layer_intermediates,
    match_modules,
    to_next_layer_kwargs,
)
from llmcompressor.utils.helpers import calibration_forward_context

if TYPE_CHECKING:
    from llmcompressor.modifiers import Modifier

__all__ = ["run_pipeline"]


def run_pipeline(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    sequential_targets: List[str],
    callback_modifier: Optional[Modifier] = None,
):
    """
    Run a layer-wise sequential data pipeline according to the following steps:

    1. Layers are identified according to `sequential_targets`
    2. A hook is attached to the first layer. This hook raises an exception which is
        then caught and used to capture the input arguments to the first layer
    3. The inputs to the first layer are used to calibrate the first layer, and the
        output of the previous layer is used as inputs to calibrate the next layer

    This pipeline requires that the model have distinct layers defined in its
    architecture and that the outputs of the previous layer are exactly the inputs
    to the next layer. This is violated by encoder-decoder architectures among others.

    If your model architecture violates these assumptions, consider using the sequential
    pipeline (see llmcompressor.pipelines.sequential). Architectures which are known to
    fail these assumptions include GPT-J and most vision language models

    :param model: model being calibrated
    :param dataloader: loads data for calibration
    :param sequential_targets: patterns which match to the layer modules of the model
    :param callback_modifier: Temporary HACK which should be replaced by event callback
    """
    # find layers
    layers = match_modules(model, sequential_targets)

    with calibration_forward_context(model):
        # prepare intermediates cache
        intermediates: IntermediatesCache = capture_first_layer_intermediates(
            model, layers[0], dataloader
        )

        num_layers = len(layers)
        for layer_index, layer in enumerate(layers):
            # prepare tqdm description texts
            calib_desc = f"({layer_index + 1}/{num_layers}): Calibrating"
            prop_desc = f"({layer_index + 1}/{num_layers}): Propagating"

            # do an preliminary pass to trigger modifier hooks
            for batch_index in tqdm.tqdm(range(len(dataloader)), desc=calib_desc):
                inputs = intermediates.fetch(batch_index)
                layer(**inputs)

            # TODO: replace with a lifecycle event
            if callback_modifier:
                callback_modifier.on_sequential_batch_end()

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
