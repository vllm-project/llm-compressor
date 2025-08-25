import contextlib
from typing import TYPE_CHECKING

import torch
from compressed_tensors.utils import disable_offloading, get_execution_device
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from llmcompressor.core import LifecycleCallbacks, active_session
from llmcompressor.modeling.prepare import moe_calibration_context
from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.pipelines.cache import IntermediatesCache
from llmcompressor.pipelines.registry import CalibrationPipeline
from llmcompressor.pipelines.sequential.helpers import (
    dispatch_for_sequential,
    get_sequential_targets,
    trace_subgraphs,
)
from llmcompressor.utils.helpers import DisableQuantization, calibration_forward_context

if TYPE_CHECKING:
    from llmcompressor.args.dataset_arguments import DatasetArguments

__all__ = ["SequentialPipeline"]


@CalibrationPipeline.register("sequential")
class SequentialPipeline(CalibrationPipeline):
    @staticmethod
    def __call__(
        model: torch.nn.Module,
        dataloader: DataLoader,
        dataset_args: "DatasetArguments",
    ):
        """
        Run a sequential data pipeline according to the following steps:

        1. The model is partitioned into subgraphs according to `sequential_targets`
        2. Data passes through each subgraph sequentially. Data is passed through each
            subgraph twice, once to trigger calibration hooks, then a second time in
            order to capture activations after quantization has occurred through hooks.
        3. The intermediate activations between each subgraph are cached and offloaded
            to the cpu between each batch in order to save memory

        This pipeline requires that the model be traceable with respect to data from the
        data loader. This may be an issue for vision models with vision datasets, due
        to specialized input processing in the model.

        In the event that tracing fails, a torch.fx.proxy.TraceError will be raised. A
        model can be made traceable by wrapping the untraceable functions (see
        llmcompressor.transformers.tracing)

        :param model: model being calibrated
        :param dataloader: loads data for calibration
        :param dataset_args: dataset arguments relevant to pipelines
        """
        session = active_session()

        # prepare model for sequential onloading
        dispatch_for_sequential(model)
        model_device = get_execution_device(model)

        # prepare to trace subgraphs
        modifiers = session.lifecycle.recipe.modifiers
        sequential_targets = get_sequential_targets(modifiers, model, dataset_args)

        ignore = dataset_args.tracing_ignore

        # trace subgraphs
        sample_input = next(iter(dataloader))
        subgraphs = trace_subgraphs(model, sample_input, sequential_targets, ignore)
        num_subgraphs = len(subgraphs)

        LifecycleCallbacks.calibration_epoch_start()

        # TODO: remove this to enable quantization aware calibration for GPTQ and AWQ
        disable_qac = any(
            type(mod).__name__ in ["GPTQModifier", "AWQModifier"]
            for mod in session.lifecycle.recipe.modifiers
        )

        with contextlib.ExitStack() as stack:
            stack.enter_context(calibration_forward_context(model))
            # Optionally disable quantization
            if not dataset_args.quantization_aware_calibration or disable_qac:
                stack.enter_context(DisableQuantization(model))

            if dataset_args.calibrate_moe_context:
                moe_calibration_context(model, stack)

            # prepare intermediates cache
            activations = IntermediatesCache.from_dataloader(dataloader, model_device)

            for subgraph_index, subgraph in enumerate(subgraphs):
                # prepare tqdm description texts
                calib_desc = f"({subgraph_index + 1}/{num_subgraphs}): Calibrating"
                prop_desc = f"({subgraph_index + 1}/{num_subgraphs}): Propagating"

                # reduce memory movement by keeping modules onloaded
                with disable_offloading():
                    # do a preliminary pass to trigger modifier hooks
                    for batch_idx in tqdm(range(len(dataloader)), desc=calib_desc):
                        inputs = activations.fetch(batch_idx, subgraph.input_names)
                        subgraph.forward(model, **inputs)

                    LifecycleCallbacks.sequential_epoch_end()

                    # this pass does not trigger modifier hooks
                    # and is only used for capturing outputs of newly compressed modules
                    with HooksMixin.disable_hooks():
                        for batch_idx in tqdm(range(len(dataloader)), desc=prop_desc):
                            inputs = activations.fetch(batch_idx, subgraph.input_names)
                            output = subgraph.forward(model, **inputs)

                            if subgraph_index < num_subgraphs - 1:
                                activations.update(batch_idx, output)
                                activations.delete(batch_idx, subgraph.consumed_names)

            # redundant, finish any remaining compression
            LifecycleCallbacks.calibration_epoch_end()
