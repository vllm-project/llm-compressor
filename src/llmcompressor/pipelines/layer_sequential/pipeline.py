from typing import TYPE_CHECKING

import torch
import tqdm
from torch.utils.data.dataloader import DataLoader

from llmcompressor.core import get_compressor
from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.pipelines.cache import IntermediatesCache
from llmcompressor.pipelines.layer_sequential.helpers import (
    capture_first_layer_intermediates,
    match_modules,
    maybe_inject_pos_embeddings,
    to_next_layer_kwargs,
)
from llmcompressor.pipelines.sequential.helpers import (
    get_targets_from_modifiers,
    infer_oneshot_device,
)
from llmcompressor.utils.helpers import align_modules, calibration_forward_context

if TYPE_CHECKING:
    from llmcompressor.args import PostTrainArguments

__all__ = ["run_pipeline"]


def run_pipeline(
    model: torch.nn.Module,
    dataloader: DataLoader,
    args: "PostTrainArguments",
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
    compressor = get_compressor()

    # if the model is dispatched, use the dispatch to determine onloading, return None
    # otherwise, infer a oneshot device (either user passed or the first available gpu)
    oneshot_device = infer_oneshot_device(model, args.oneshot_device)

    # find layers
    sequential_targets, _ = get_targets_from_modifiers(compressor.modifiers, model)
    layers = match_modules(model, sequential_targets)

    compressor.initialize()
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

            with align_modules(layer.modules(), oneshot_device):
                # do an preliminary pass to trigger modifier hooks
                for batch_index in tqdm.tqdm(range(len(dataloader)), desc=calib_desc):
                    inputs = intermediates.fetch(batch_index)
                    layer(**inputs)

                # trigger compression
                compressor.sequential_epoch_end()

                # this pass does not trigger modifier hooks
                # and is only used for capturing outputs from newly compressed modules
                with HooksMixin.disable_hooks():
                    for batch_index in tqdm.tqdm(
                        range(len(dataloader)), desc=prop_desc
                    ):
                        inputs = intermediates.fetch(batch_index)
                        output = layer(**inputs)

                        if layer_index < num_layers - 1:
                            next_layer = layers[layer_index + 1]
                            output = to_next_layer_kwargs(output, next_layer)
                            output = maybe_inject_pos_embeddings(
                                output, next_layer, inputs
                            )

                            intermediates.delete(batch_index)
                            intermediates.update(batch_index, output)

        # redudant, finish any remaining compression
        compressor.calibration_epoch_end()
