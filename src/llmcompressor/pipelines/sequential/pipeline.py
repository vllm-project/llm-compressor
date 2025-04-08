from typing import TYPE_CHECKING, List, Optional

import torch
from tqdm import tqdm
from transformers import PreTrainedModel

from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.pipelines.cache import IntermediatesCache
from llmcompressor.pipelines.sequential.helpers import (
    infer_oneshot_device,
    trace_subgraphs,
)
from llmcompressor.utils.helpers import (
    DisableQuantization,
    align_modules,
    calibration_forward_context,
)

if TYPE_CHECKING:
    from llmcompressor.modifiers import Modifier

__all__ = ["run_pipeline"]


def run_pipeline(
    model: PreTrainedModel,
    dataloader: torch.utils.data.DataLoader,
    sequential_targets: List[str],
    ignore: List[str],
    oneshot_device: Optional[torch.device],
    callback_modifier: Optional["Modifier"] = None,
):
    """
    Run a sequential data pipeline according to the following steps:

    1. The model is partitioned into subgraphs according to `sequential_targets`
    2. Data passes through each subgraph sequentially. Data is passed through each
        subgraph twice, once to trigger calibration hooks, then a second time in order
        to capture activations after quantization has occurred through the hooks.
    3. The intermediate activations between each subgraph are cached and offloaded to
        the cpu between each batch in order to save memory

    This pipeline requires that the model be traceable with respect to data from the
    data loader. This may be an issue for vision language models with vision datasets,
    due to specialized input processing in the model.

    In the event that tracing fails, a torch.fx.proxy.TraceError will be raised. A model
    can be made traceable by wrapping the untraceable functions (see
    llmcompressor.transformers.tracing)

    :param model: model being calibrated
    :param dataloader: loads data for calibration
    :param sequential_targets: patterns which match to the layer modules of the model
    :param ignore: patterns which match to modules which should be ignored by tracing
    :param oneshot_device: device to onload layers ontop, uses device_map if None
    :param callback_modifier: Temporary HACK which should be replaced by event callback
    """
    # if the model is dispatched, use the dispatch to determine onloading, return None
    # otherwise, infer a oneshot device (either user passed or the first available gpu)
    oneshot_device = infer_oneshot_device(model, oneshot_device)

    # trace subgraphs
    sample_input = next(iter(dataloader))
    subgraphs = trace_subgraphs(model, sample_input, sequential_targets, ignore)

    # prepare intermediates cache
    model_device = oneshot_device or model.device
    intermediates = IntermediatesCache.from_dataloader(dataloader, model_device)

    with calibration_forward_context(model), DisableQuantization(model):
        num_subgraphs = len(subgraphs)
        for subgraph_index, subgraph in enumerate(subgraphs):
            # prepare tqdm description texts
            calib_desc = f"({subgraph_index + 1}/{num_subgraphs}): Calibrating"
            prop_desc = f"({subgraph_index + 1}/{num_subgraphs}): Propagating"

            # compile subgraph forward function
            forward_function = subgraph.compile_forward()

            with align_modules(subgraph.modules, oneshot_device):
                # do an preliminary pass to trigger modifier hooks
                for batch_index in tqdm(range(len(dataloader)), desc=calib_desc):
                    inputs = intermediates.fetch(batch_index, subgraph.input_names)
                    forward_function(model, **inputs)

                # TODO: replace with a lifecycle event
                if callback_modifier:
                    callback_modifier.on_sequential_batch_end()

                # this pass does not trigger modifier hooks
                # and is only used for capturing outputs from newly compressed modules
                with HooksMixin.disable_hooks():
                    for batch_index in tqdm(range(len(dataloader)), desc=prop_desc):
                        inputs = intermediates.fetch(batch_index, subgraph.input_names)
                        output = forward_function(model, **inputs)

                        if subgraph_index < num_subgraphs - 1:
                            intermediates.update(batch_index, output)
                            intermediates.delete(batch_index, subgraph.consumed_names)
