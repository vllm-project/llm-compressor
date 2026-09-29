from __future__ import annotations

import contextlib
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Iterator

import torch
from compressed_tensors.compressors import compress_module, decompress_module
from compressed_tensors.distributed import is_distributed, replace_module_parallel
from compressed_tensors.offload import set_onload_device
from compressed_tensors.quantization.utils import is_module_quantized
from compressed_tensors.offload.module import (
    subgraph_offload_modules,
    subgraph_onload_modules,
    subgraph_stage_modules,
)
from loguru import logger
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from llmcompressor.core import LifecycleCallbacks, active_session
from llmcompressor.modeling.moe.linearize import (
    linearize_moe,
    repack_moe,
)
from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.pipelines.cache import IntermediatesCache
from llmcompressor.pipelines.registry import CalibrationPipeline
from llmcompressor.pipelines.sequential.error_logging import (
    compute_subgraph_sqnr,
    process_batch_error,
)
from llmcompressor.pipelines.sequential.helpers import (
    find_modules_outside_subgraphs,
    handle_sequential_oom,
    trace_subgraphs,
)
from llmcompressor.utils.dev import get_main_device
from llmcompressor.utils.helpers import DisableQuantization, calibration_forward_context
from llmcompressor.utils.pytorch.module import infer_sequential_targets

if TYPE_CHECKING:
    from llmcompressor.args.dataset_arguments import DatasetArguments

__all__ = ["SequentialPipeline"]


def _submit_subgraph_staging(
    executor: ThreadPoolExecutor,
    modules: dict[str, torch.nn.Module],
    current_modules: dict[str, torch.nn.Module],
    pin_memory: bool,
) -> tuple[dict[str, torch.nn.Module], Future] | None:
    """Stage disjoint next-subgraph modules in the background."""
    if set(modules.values()) & set(current_modules.values()):
        return None

    future = executor.submit(
        subgraph_stage_modules,
        modules,
        pin_memory=pin_memory,
    )
    return modules, future


def _get_batches(
    activations: IntermediatesCache,
    num_batches: int,
    input_names: list[str],
    desc: str,
) -> Iterator[tuple[int, dict]]:
    """
    Yield (batch_idx, inputs) while prefetching the next batch in a background thread to
    overlap fetch (onload from offload device) with the main-thread forward pass.
    """
    batch_source = activations.iter_prefetch(input_names)
    for batch_idx, inputs in tqdm(
        enumerate(batch_source), total=num_batches, desc=desc
    ):
        yield batch_idx, inputs


@CalibrationPipeline.register("sequential")
class SequentialPipeline(CalibrationPipeline):
    @staticmethod
    @handle_sequential_oom
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
        _logger = logger.patch(lambda r: r.update(function="SequentialPipeline"))

        if getattr(dataset_args, "sequential_prefetch", False):
            warnings.warn(
                "sequential_prefetch is deprecated and has no effect because "
                "activation prefetching is always enabled.",
                DeprecationWarning,
                stacklevel=2,
            )

        session = active_session()

        # prepare model for sequential onloading
        onload_device = get_main_device()
        offload_device = torch.device(dataset_args.sequential_offload_device)
        set_onload_device(model, onload_device)

        # AutoRoundModifier optimizes each layer independently using its own
        # forward passes, so quantization error should not be propagated between
        # layers during the calibration stage
        modifiers = session.lifecycle.recipe.modifiers
        if any(type(m).__name__ == "AutoRoundModifier" for m in modifiers):
            dataset_args.propagate_error = False

        # prepare to trace subgraphs
        sequential_targets = infer_sequential_targets(
            model, dataset_args.sequential_targets
        )
        ignore = dataset_args.tracing_ignore

        # trace subgraphs
        sample_input = next(iter(dataloader))
        subgraphs = trace_subgraphs(
            model,
            sample_input,
            sequential_targets,
            ignore,
            dataset_args.sequential_targets_per_subgraph,
        )
        num_subgraphs = len(subgraphs)
        persistent_modules = find_modules_outside_subgraphs(model, subgraphs)
        if persistent_modules:
            subgraph_onload_modules(persistent_modules)

        LifecycleCallbacks.calibration_start()

        with contextlib.ExitStack() as stack:
            stack.enter_context(calibration_forward_context(model))
            stack.enter_context(DisableQuantization(model))

            # prepare intermediates cache
            activations = IntermediatesCache.from_dataloader(
                dataloader, onload_device, offload_device
            )

            # prepare error-logging cache (separate from activations so that
            # log_sequential_error does not force propagate_error=True)
            #
            # how activations are updated per (propagate_error, log_sequential_error):
            #
            #   (True,  False) - pass 2 overwrites activations with quantized outputs
            #   (False, False) - pass 1 overwrites activations immediately, no pass 2
            #   (True,  True)  - same as (True, False) + SQNR measured from
            #                    seq_error_cache
            #   (False, True)  - pass 2 transfers unquantized outputs from
            #                    seq_error_cache into activations (zero-copy) +
            #                    SQNR measured
            seq_error_cache = (
                IntermediatesCache.empty(len(dataloader), offload_device)
                if dataset_args.log_sequential_error
                else None
            )

            # Populate loss_masks once from cached activations for AWQ masking support
            use_loss_mask = getattr(dataset_args, "use_loss_mask", False)
            if use_loss_mask:
                session.state.loss_masks = [
                    activations.fetch(batch_idx, ["loss_mask"]).get("loss_mask")
                    for batch_idx in range(len(dataloader))
                ]
            else:
                session.state.loss_masks = None

            stage_weights_in_pinned_memory = getattr(
                dataset_args, "stage_weights_in_pinned_memory", False
            )

            # A single worker preserves staging order and bounds staged memory.
            stage_executor = stack.enter_context(ThreadPoolExecutor(max_workers=1))

            # Prefetch first subgraph modules
            next_subgraph_modules = subgraphs[0].submodule_dict(model)
            prefetched_staging = _submit_subgraph_staging(
                stage_executor,
                next_subgraph_modules,
                {},
                stage_weights_in_pinned_memory,
            )

            for subgraph_index, subgraph in enumerate(subgraphs):
                if prefetched_staging is None:
                    subgraph_modules = subgraph.submodule_dict(model)
                    subgraph_stage_modules(
                        subgraph_modules, pin_memory=stage_weights_in_pinned_memory
                    )
                    warnings.warn(
                        "Subgraph prefetching failed for subgraphs "
                        f"{subgraph_index - 1} to {subgraph_index}. "
                        "This may be due to overlapping modules between subgraphs. ",
                        UserWarning,
                        stacklevel=2,
                    )
                else:
                    subgraph_modules, stage_future = prefetched_staging
                    stage_future.result()
                prefetched_staging = None

                # prepare tqdm description texts
                calib_desc = f"({subgraph_index + 1}/{num_subgraphs}): Calibrating"
                prop_desc = f"({subgraph_index + 1}/{num_subgraphs}): Propagating"

                # whether a downstream subgraph still needs this subgraph's outputs
                has_next_subgraph = subgraph_index < num_subgraphs - 1

                # reduce memory movement by keeping modules onloaded
                num_batches = len(dataloader)

                # Submit the next subgraph's staging before onloading the
                # current subgraph so staging can overlap with onload work.
                if subgraph_index + 1 < num_subgraphs:
                    next_subgraph_modules = subgraphs[
                        subgraph_index + 1
                    ].submodule_dict(model)
                    prefetched_staging = _submit_subgraph_staging(
                        stage_executor,
                        next_subgraph_modules,
                        subgraph_modules,
                        stage_weights_in_pinned_memory,
                    )

                #######################
                ### START OF ONLOAD ###
                #######################
                offload_kwargs = subgraph_onload_modules(subgraph_modules)

                # This is a no-op for already-linearized MoE layers.
                linearize_moe(model, subgraph_modules, offload_kwargs=offload_kwargs)

                modules = subgraph.submodules(model)
                if dataset_args.layerwise_decompression:
                    compressed = [
                        module for module in modules if is_module_quantized(module)
                    ]
                    for module in tqdm(compressed, desc="Decompressing modules"):
                        decompress_module(module, leave_decompressed=False)
                    for modifier in modifiers:
                        if hasattr(modifier, "start_layerwise_calibration"):
                            modifier.start_layerwise_calibration(model, modules)

                # do a preliminary pass to trigger modifier hooks
                for batch_idx, inputs in _get_batches(
                    activations,
                    num_batches,
                    subgraph.input_names,
                    calib_desc,
                ):
                    session.state.current_batch_idx = batch_idx
                    outputs = subgraph.forward(model, **inputs)

                    # update activations immediately only when no pass 2
                    # is needed; otherwise defer to after pass 2
                    if not dataset_args.propagate_error:
                        if not dataset_args.log_sequential_error and has_next_subgraph:
                            activations.update(batch_idx, outputs)
                            activations.delete(batch_idx, subgraph.consumed_names)

                    if seq_error_cache is not None and has_next_subgraph:
                        seq_error_cache.update(batch_idx, outputs)

                LifecycleCallbacks.sequential_epoch_end(modules)

                if dataset_args.propagate_error or dataset_args.log_sequential_error:
                    # this pass does not trigger modifier hooks; it captures
                    # outputs of compressed modules (for propagate_error) and/or
                    # per-batch signal/noise power for SQNR (for
                    # log_sequential_error)
                    batch_powers: list[tuple[float, float]] = []
                    with HooksMixin.disable_hooks():
                        for batch_idx, inputs in _get_batches(
                            activations, num_batches, subgraph.input_names, prop_desc
                        ):
                            output = subgraph.forward(model, **inputs)
                            if dataset_args.propagate_error and has_next_subgraph:
                                activations.update(batch_idx, output)
                                activations.delete(batch_idx, subgraph.consumed_names)

                            if seq_error_cache is not None and has_next_subgraph:
                                batch_power = process_batch_error(
                                    seq_error_cache,
                                    activations,
                                    batch_idx,
                                    output,
                                    dataset_args.propagate_error,
                                    subgraph.consumed_names,
                                )
                                if batch_power is not None:
                                    batch_powers.append(batch_power)

                    if batch_powers:
                        sqnr = compute_subgraph_sqnr(batch_powers)
                        _logger.log(
                            "METRIC",
                            f"subgraph {subgraph_index + 1}/{num_subgraphs} | "
                            f"sequential error (SQNR dB): {sqnr:.2f}",
                        )
                if (
                    dataset_args.repack_moe_layers
                    and not dataset_args.layerwise_compression
                ):
                    repack_moe(model, subgraph_modules, offload_kwargs=offload_kwargs)

                if dataset_args.layerwise_compression:
                    quantized = [
                        module
                        for module in subgraph.submodules(model)
                        if is_module_quantized(module)
                    ]
                    if not is_distributed():
                        for module in tqdm(quantized, desc="Compressing modules"):
                            compress_module(module)
                    else:
                        replace_module_parallel(
                            quantized, compress_module, desc="Compressing modules"
                        )

                subgraph_offload_modules(subgraph_modules, offload_kwargs)
                #######################
                #### END OF ONLOAD ####
                #######################

            # redundant, finish any remaining compression
            LifecycleCallbacks.calibration_end()
