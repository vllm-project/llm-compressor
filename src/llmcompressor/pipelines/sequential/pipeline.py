import contextlib
import time
from typing import TYPE_CHECKING

import torch
from compressed_tensors.utils import disable_offloading, get_execution_device
from loguru import logger
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from llmcompressor.core import LifecycleCallbacks, active_session
from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.pipelines.cache import IntermediatesCache
from llmcompressor.pipelines.registry import CalibrationPipeline
from llmcompressor.pipelines.sequential.helpers import (
    dispatch_for_sequential,
    get_sequential_targets,
    trace_subgraphs,
)
from llmcompressor.utils.helpers import (
    DISABLE_QAC_MODIFIERS,
    DisableQuantization,
    calibration_forward_context,
)

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

        # TODO: remove this to enable quantization aware calibration
        # for GPTQ, AWQ and AutoRound.
        disable_qac = any(
            type(mod).__name__ in DISABLE_QAC_MODIFIERS
            for mod in session.lifecycle.recipe.modifiers
        )

        # Initialize timing stats
        timing_stats = {
            "calibration_forward": 0.0,
            "propagation_forward": 0.0,
            "cache_preparation": 0.0,
        }

        with contextlib.ExitStack() as stack:
            stack.enter_context(calibration_forward_context(model))
            # Optionally disable quantization
            if not dataset_args.quantization_aware_calibration or disable_qac:
                stack.enter_context(DisableQuantization(model))

            # prepare intermediates cache
            cache_start = time.time()
            cache_offload = dataset_args.offload_sequential_activations
            offload_device = torch.device(dataset_args.sequential_offload_device)
            activations = IntermediatesCache.from_dataloader(
                dataloader, model_device, offload_device=offload_device
            )
            timing_stats["cache_preparation"] = time.time() - cache_start

            for subgraph_index, subgraph in enumerate(subgraphs):
                # prepare tqdm description texts
                calib_desc = f"({subgraph_index + 1}/{num_subgraphs}): Calibrating"
                prop_desc = f"({subgraph_index + 1}/{num_subgraphs}): Propagating"

                # reduce memory movement by keeping modules onloaded
                with disable_offloading():
                    # do a preliminary pass to trigger modifier hooks
                    calib_start = time.time()
                    for batch_idx in tqdm(range(len(dataloader)), desc=calib_desc):
                        inputs = activations.fetch(batch_idx, subgraph.input_names)
                        subgraph.forward(model, **inputs)
                    timing_stats["calibration_forward"] += time.time() - calib_start

                    LifecycleCallbacks.sequential_epoch_end(subgraph)

                    # this pass does not trigger modifier hooks
                    # and is only used for capturing outputs of newly compressed modules
                    with HooksMixin.disable_hooks():
                        prop_start = time.time()
                        for batch_idx in tqdm(range(len(dataloader)), desc=prop_desc):
                            inputs = activations.fetch(batch_idx, subgraph.input_names)
                            output = subgraph.forward(model, **inputs)

                            if subgraph_index < num_subgraphs - 1:
                                activations.update(batch_idx, output)
                                activations.delete(batch_idx, subgraph.consumed_names)
                        timing_stats["propagation_forward"] += time.time() - prop_start

            # redundant, finish any remaining compression
            LifecycleCallbacks.calibration_epoch_end()

        # Print timing summary
        _print_sequential_pipeline_timing(timing_stats)


def _print_sequential_pipeline_timing(timing_stats: dict):
    """Print a summary of sequential pipeline timing statistics"""
    if not timing_stats:
        return

    logger.info("\n" + "=" * 80)
    logger.info("Sequential Pipeline Timing Summary")
    logger.info("=" * 80)
    logger.info("\nPipeline Phases:")
    logger.info("-" * 80)

    total_time = 0.0
    for metric, value in timing_stats.items():
        logger.info(f"  {metric:30s}: {value:8.2f}s")
        total_time += value

    logger.info("-" * 80)
    logger.info(f"  {'PIPELINE TOTAL':30s}: {total_time:8.2f}s")
    logger.info("=" * 80 + "\n")
