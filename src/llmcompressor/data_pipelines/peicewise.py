import contextlib
import torch

from datasets import Dataset

from llmcompressor.core.session_functions import initialize
from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.modifiers.utils.pytorch_helpers import EarlyStopException
from llmcompressor.recipe.recipe import Recipe
from llmcompressor.utils.helpers import calibration_forward_context, trace_subgraphs, get_targets, get_model_device, tensors_to_device, create_dataloader


def run_pipeline(
    model: torch.nn.Module,
    recipe: Recipe,
    dataset: Dataset,
    propagate_error: bool,
):
    # trace subgraphs
    targets = get_targets(recipe)
    sample_input_names = next(iter(dataset)).keys()
    subgraphs = trace_subgraphs(model, sample_input_names, targets)

    # apply recipe to model
    initialize(recipe, model)

    # create dataloader
    model_device = get_model_device(model)
    dataloader = create_dataloader(dataset, batch_size=..., mask_padding=True, model_device=model_device)

    with calibration_forward_context(model):
        # prepare intermediates cache
        batch_intermediates = list(iter(dataloader))
        batch_outputs = [None for _ in range(len(dataloader))]

        for subgraph_index, subgraph in enumerate(subgraphs):
            # compile subgraph forward function
            code = subgraph["code"]
            exec(code.src, code.globals)
            forward_function = code.globals.get("forward")

            if propagate_error:
                # do an preliminary pass to trigger modifier hooks
                for batch_index in range(len(dataloader)):
                    intermediates = batch_intermediates[batch_index]
                    inputs = {
                        input_name: intermediates[input_name]
                        for input_name in subgraph["input_names"]
                    }
                    inputs = tensors_to_device(inputs, model_device)
                    try:
                        forward_function(model, **inputs)
                    except EarlyStopException:
                        pass

            # if using propagate_error, then this pass does not trigger modifier hooks
            # and is only used for capturing intermediates
            # otherwise, this pass triggers modifier hooks and captures intermediates
            with HooksMixin.disable_hooks() if propagate_error else contextlib.nullcontext():
                for batch_index in range(len(dataloader)):
                    intermediates = batch_intermediates[batch_index]

                    inputs = {
                        input_name: intermediates[input_name]
                        for input_name in subgraph["input_names"]
                    }
                    inputs = tensors_to_device(inputs, model_device)
                    subgraph_output = forward_function(model, **inputs)
                    subgraph_output = tensors_to_device(subgraph_output, "cpu")

                    for consumed_name in subgraph["consumed_names"]:
                        del intermediates[consumed_name]

                    if subgraph_index < len(subgraphs) - 1:
                        intermediates.update(subgraph_output)
                    else:
                        batch_outputs[batch_index] = subgraph_output
