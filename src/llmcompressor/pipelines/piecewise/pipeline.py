from contextlib import nullcontext

import torch
import torch.utils.data.dataloader
import tqdm

from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.modifiers.utils.pytorch_helpers import apply_pad_mask_to_batch
from llmcompressor.pipelines.piecewise.helpers import (
    get_compression_targets,
    trace_subgraphs,
)
from llmcompressor.pytorch.utils.helpers import tensors_to_device
from llmcompressor.recipe.recipe import Recipe
from llmcompressor.utils.helpers import calibration_forward_context

__all__ = ["run_pipeline"]


def run_pipeline(
    model: torch.nn.Module,
    recipe: Recipe,
    dataloader: torch.utils.data.DataLoader,
    propagate_error: bool,
):
    # trace subgraphs
    sample_input = next(iter(dataloader))
    targets = get_compression_targets(model, recipe)
    subgraphs = trace_subgraphs(model, sample_input, targets)

    # FUTURE: apply recipe to model
    # initialize(recipe, model)

    model_device = next(model.parameters()).device

    with calibration_forward_context(model):
        # prepare intermediates cache
        batch_intermediates = [
            apply_pad_mask_to_batch(batch) for batch in iter(dataloader)
        ]
        batch_outputs = [None for _ in range(len(dataloader))]

        num_subgraphs = len(subgraphs)
        for index, subgraph in enumerate(subgraphs):
            # prepare tqdm description texts
            uncomp_desc = f"({index + 1}/{num_subgraphs}): Calibrating"
            comp_desc = f"({index + 1}/{num_subgraphs}): Propagate"

            # compile subgraph forward function
            code = subgraph.graph.python_code("self")
            exec(code.src, code.globals)
            forward_function = code.globals.get("forward")

            if propagate_error:
                # do an preliminary pass to trigger modifier hooks
                for batch_index in tqdm.tqdm(range(len(dataloader)), desc=uncomp_desc):
                    intermediates = batch_intermediates[batch_index]
                    inputs = {
                        input_name: intermediates[input_name]
                        for input_name in subgraph.input_names
                    }
                    inputs = tensors_to_device(inputs, model_device)
                    forward_function(model, **inputs)

            # if using propagate_error, then this pass does not trigger modifier hooks
            # and is only used for capturing intermediates
            # otherwise, this pass triggers modifier hooks and captures intermediates
            with HooksMixin.disable_hooks() if propagate_error else nullcontext():
                desc = comp_desc if propagate_error else uncomp_desc
                for batch_index in tqdm.tqdm(range(len(dataloader)), desc=desc):
                    intermediates = batch_intermediates[batch_index]

                    inputs = {
                        input_name: intermediates[input_name]
                        for input_name in subgraph.input_names
                    }
                    inputs = tensors_to_device(inputs, model_device)
                    subgraph_output = forward_function(model, **inputs)
                    subgraph_output = tensors_to_device(subgraph_output, "cpu")

                    for consumed_name in subgraph.consumed_names:
                        del intermediates[consumed_name]

                    if index < len(subgraphs) - 1:
                        intermediates.update(subgraph_output)
                    else:
                        batch_outputs[batch_index] = subgraph_output
