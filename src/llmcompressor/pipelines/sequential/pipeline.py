import contextlib
from typing import TYPE_CHECKING, Iterator

import torch
from compressed_tensors.offload import disable_offloading, set_onload_device
from loguru import logger
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from llmcompressor.core import LifecycleCallbacks, active_session
from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.pipelines.cache import IntermediatesCache
from llmcompressor.pipelines.registry import CalibrationPipeline
from llmcompressor.pipelines.sequential.error_logging import (
    cache_pre_compression_output,
    compute_subgraph_sqnr,
    record_batch_error,
)
from llmcompressor.pipelines.sequential.helpers import (
    handle_sequential_oom,
    trace_subgraphs,
)
from llmcompressor.utils.dev import get_main_device
from llmcompressor.utils.helpers import DisableQuantization, calibration_forward_context
from llmcompressor.utils.pytorch.module import infer_sequential_targets

if TYPE_CHECKING:
    from llmcompressor.args.dataset_arguments import DatasetArguments

__all__ = ["SequentialPipeline"]


def _get_batches(
    activations: IntermediatesCache,
    num_batches: int,
    input_names: list[str],
    desc: str,
    sequential_prefetch: bool = False,
) -> Iterator[tuple[int, dict]]:
    """
    Yield (batch_idx, inputs) with the next batch optionally prefetched in a
    background thread to overlap fetch (onload from offload device) with the
    main-thread forward pass. Delegates to
    :meth:`IntermediatesCache.iter_prefetch` when prefetching is enabled.
    """
    batch_source = (
        activations.iter_prefetch(input_names)
        if sequential_prefetch
        else activations.iter(input_names)
    )
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

            sequential_prefetch = getattr(dataset_args, "sequential_prefetch", False)
            session.state.sequential_prefetch = sequential_prefetch

            for subgraph_index, subgraph in enumerate(subgraphs):
                # prepare tqdm description texts
                calib_desc = f"({subgraph_index + 1}/{num_subgraphs}): Calibrating"
                prop_desc = f"({subgraph_index + 1}/{num_subgraphs}): Propagating"

                # whether a downstream subgraph still needs this subgraph's outputs
                has_next_subgraph = subgraph_index < num_subgraphs - 1

                # reduce memory movement by keeping modules onloaded
                num_batches = len(dataloader)
                with disable_offloading():
                    # do a preliminary pass to trigger modifier hooks
                    for batch_idx, inputs in _get_batches(
                        activations,
                        num_batches,
                        subgraph.input_names,
                        calib_desc,
                        sequential_prefetch,
                    ):
                        session.state.current_batch_idx = batch_idx
                        outputs = subgraph.forward(model, **inputs)

                        # update activations immediately only when no pass 2
                        # is needed; otherwise defer to after pass 2
                        if not dataset_args.propagate_error:
                            if (
                                not dataset_args.log_sequential_error
                                and has_next_subgraph
                            ):
                                activations.update(batch_idx, outputs)
                                activations.delete(batch_idx, subgraph.consumed_names)

                        if seq_error_cache is not None and has_next_subgraph:
                            cache_pre_compression_output(
                                seq_error_cache, batch_idx, outputs
                            )

                    LifecycleCallbacks.sequential_epoch_end(subgraph.submodules(model))

                    if (
                        dataset_args.propagate_error
                        or dataset_args.log_sequential_error
                    ):
                        # this pass does not trigger modifier hooks; it captures
                        # outputs of compressed modules (for propagate_error) and/or
                        # per-batch signal/noise power for SQNR (for
                        # log_sequential_error)
                        batch_powers: list[tuple[float, float]] = []
                        with HooksMixin.disable_hooks():
                            for batch_idx, inputs in _get_batches(
                                activations,
                                num_batches,
                                subgraph.input_names,
                                prop_desc,
                                sequential_prefetch,
                            ):
                                output = subgraph.forward(model, **inputs)
                                if dataset_args.propagate_error and has_next_subgraph:
                                    activations.update(batch_idx, output)
                                    activations.delete(
                                        batch_idx, subgraph.consumed_names
                                    )

                                if seq_error_cache is not None and has_next_subgraph:
                                    batch_power = record_batch_error(
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

            # redundant, finish any remaining compression
            LifecycleCallbacks.calibration_end()
