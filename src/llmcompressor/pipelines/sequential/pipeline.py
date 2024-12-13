from contextlib import nullcontext
from typing import List

import torch
import torch.utils.data.dataloader
import tqdm
from compressed_tensors.utils import get_execution_device

from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.pipelines.sequential.cache import IntermediatesCache
from llmcompressor.pipelines.sequential.helpers import trace_subgraphs
from llmcompressor.utils.helpers import calibration_forward_context

__all__ = ["run_pipeline"]


def run_pipeline(
    model: torch.nn.Module,
    sequential_targets: List[str],  # FUTURE: replace with recipe inference
    ignore: List[str],
    dataloader: torch.utils.data.DataLoader,
    propagate_error: bool,
):
    """
    Run a sequential data pipeline. First, the model is partitioned into subgraphs
    according to `sequential_targets`. Then, data passes through each subgraph
    sequentially. If `propagate_error` is enabled, then data is passed through each
    subgraph twice, once to trigger calibration hooks, then a second time in order to
    capture activations after quantization has occurred through the hooks.

    In order to reduce memory requirements
    1. Data is passed through each subgraph with batch size 1
    2. Intermediate activations between each subgraph are offloaded onto the CPU

    This pipeline requires that the model be tracable with respect to data from the
    data loader. This may be an issue for vision language models with vision datasets,
    due to specialized input processing in the model. In the event that tracing fails,
    a torch.fx.proxy.TraceError will be raised.
    """
    # trace subgraphs
    sample_input = next(iter(dataloader))
    subgraphs = trace_subgraphs(model, sample_input, sequential_targets, ignore)

    # FUTURE: apply recipe to model
    # initialize(recipe, model)

    with calibration_forward_context(model):
        # prepare intermediates cache
        model_device = get_execution_device(model)
        intermediates = IntermediatesCache.from_dataloader(dataloader, model_device)

        num_subgraphs = len(subgraphs)
        for subgraph_index, subgraph in enumerate(subgraphs):
            # prepare tqdm description texts
            calib_desc = f"({subgraph_index + 1}/{num_subgraphs}): Calibrating"
            prop_desc = f"({subgraph_index + 1}/{num_subgraphs}): Propagate"

            # compile subgraph forward function
            forward_function = subgraph.compile_forward()

            if propagate_error:
                # do an preliminary pass to trigger modifier hooks
                for batch_index in tqdm.tqdm(range(len(dataloader)), desc=calib_desc):
                    inputs = intermediates.fetch(batch_index, subgraph.input_names)
                    forward_function(model, **inputs)

            # if using propagate_error, then this pass does not trigger modifier hooks
            # and is only used for capturing intermediates
            # otherwise, this pass triggers modifier hooks and captures intermediates
            with HooksMixin.disable_hooks() if propagate_error else nullcontext():
                desc = prop_desc if propagate_error else calib_desc
                for batch_index in tqdm.tqdm(range(len(dataloader)), desc=desc):
                    inputs = intermediates.fetch(batch_index, subgraph.input_names)
                    output = forward_function(model, **inputs)

                    if subgraph_index < len(subgraphs) - 1:
                        intermediates.update(batch_index, output)
                        intermediates.delete(batch_index, subgraph.consumed_names)
