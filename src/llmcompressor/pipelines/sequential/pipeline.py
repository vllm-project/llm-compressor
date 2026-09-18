import contextlib
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Iterator

import torch
from compressed_tensors.offload import set_onload_device
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from llmcompressor.core import LifecycleCallbacks, active_session
from llmcompressor.modeling.moe.linearize import (
    linearize_moe_model,
    linearize_moe_subgraph,
    repack_moe_model,
    repack_moe_subgraph,
)
from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.pipelines.cache import IntermediatesCache
from llmcompressor.pipelines.registry import CalibrationPipeline
from llmcompressor.pipelines.sequential.helpers import (
    handle_sequential_oom,
    trace_subgraphs,
)
from llmcompressor.utils.dev import get_main_device
from llmcompressor.utils.helpers import DisableQuantization, calibration_forward_context
from llmcompressor.utils.pytorch.module import infer_sequential_targets

from .offloading import offload, onload

if TYPE_CHECKING:
    from llmcompressor.args.dataset_arguments import DatasetArguments

__all__ = ["SequentialPipeline"]


def _wait_for_overlapping_offloads(
    pending_offloads: list[tuple[set[torch.nn.Module], Future]],
    modules: set[torch.nn.Module],
) -> list[tuple[set[torch.nn.Module], Future]]:
    """Wait for offloads which could race with the next subgraph."""
    remaining = []
    for offloaded_modules, future in pending_offloads:
        if future.done() or offloaded_modules & modules:
            future.result()
        else:
            remaining.append((offloaded_modules, future))
    return remaining


def _get_batches(
    activations: IntermediatesCache,
    num_batches: int,
    input_names: list[str],
    desc: str,
    activation_prefetch: bool = False,
) -> Iterator[tuple[int, dict]]:
    """
    Yield (batch_idx, inputs) with the next batch optionally prefetched in a
    background thread to overlap fetch (onload from offload device) with the
    main-thread forward pass. Delegates to
    :meth:`IntermediatesCache.iter_prefetch` when prefetching is enabled.
    """
    batch_source = (
        activations.iter_prefetch(input_names)
        if activation_prefetch
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

            # linearize MoE layers upfront if not using layer-wise linearization
            if not dataset_args.moe_eager_linearization_and_repack:
                linearize_moe_model(model)

            # prepare intermediates cache
            activations = IntermediatesCache.from_dataloader(
                dataloader, onload_device, offload_device
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

            sequential_activation_prefetch = getattr(
                dataset_args, "sequential_activation_prefetch", False
            )
            sequential_module_prefetch = getattr(
                dataset_args, "sequential_module_prefetch", False
            )
            session.state.sequential_activation_prefetch = (
                sequential_activation_prefetch
            )

            pending_offloads: list[tuple[set[torch.nn.Module], Future]] = []
            prefetched_onload: tuple[dict[str, torch.nn.Module], Future] | None = None
            if sequential_module_prefetch:
                offload_executor = ThreadPoolExecutor(max_workers=8)
                onload_executor = ThreadPoolExecutor(max_workers=8)
                stack.callback(offload_executor.shutdown, True)
                stack.callback(onload_executor.shutdown, True)
            else:
                offload_executor = None
                onload_executor = None

            for subgraph_index, subgraph in enumerate(subgraphs):
                if not sequential_module_prefetch or prefetched_onload is None:
                    subgraph_modules = subgraph.submodule_dict(model)
                    onload_future = None
                else:
                    subgraph_modules, onload_future = prefetched_onload
                    prefetched_onload = None

                if sequential_module_prefetch:
                    pending_offloads = _wait_for_overlapping_offloads(
                        pending_offloads, set(subgraph_modules.values())
                    )

                #######################
                ### START OF ONLOAD ###
                #######################
                if onload_future is None:
                    offload_kwargs = onload(subgraph_modules)
                else:
                    offload_kwargs = onload_future.result()
                # subgraph_modules and their corresponding offload_kwargs
                # are now onloaded and ready for calibration. Be very
                # careful with how you use subgraph_modules and offload_kwargs

                # prepare tqdm description texts
                calib_desc = f"({subgraph_index + 1}/{num_subgraphs}): Calibrating"
                prop_desc = f"({subgraph_index + 1}/{num_subgraphs}): Propagating"

                # reduce memory movement by keeping modules onloaded
                num_batches = len(dataloader)

                # linearize moe layers just before calibration,
                if dataset_args.moe_eager_linearization_and_repack:
                    linearize_moe_subgraph(model, subgraph_modules)

                # Start loading the next independent subgraph while this one is
                # calibrating and propagating. Shared modules must remain synchronous
                # to avoid racing this subgraph's use of their tensors.
                if sequential_module_prefetch and subgraph_index + 1 < num_subgraphs:
                    next_subgraph_modules = subgraphs[
                        subgraph_index + 1
                    ].submodule_dict(model)
                    current_module_set = set(subgraph_modules.values())
                    next_module_set = set(next_subgraph_modules.values())
                    has_shared_modules = bool(current_module_set & next_module_set)
                    has_conflicting_offload = any(
                        not future.done() and bool(offloaded_modules & next_module_set)
                        for offloaded_modules, future in pending_offloads
                    )
                    if not has_shared_modules and not has_conflicting_offload:
                        onload_future = onload_executor.submit(
                            onload, next_subgraph_modules
                        )
                        stack.callback(onload_future.result)
                        prefetched_onload = (
                            next_subgraph_modules,
                            onload_future,
                        )

                # do a preliminary pass to trigger modifier hooks
                for batch_idx, inputs in _get_batches(
                    activations,
                    num_batches,
                    subgraph.input_names,
                    calib_desc,
                    sequential_activation_prefetch,
                ):
                    session.state.current_batch_idx = batch_idx
                    outputs = subgraph.forward(model, **inputs)

                    if not dataset_args.propagate_error:
                        if subgraph_index < num_subgraphs - 1:
                            activations.update(batch_idx, outputs)
                            activations.delete(batch_idx, subgraph.consumed_names)

                LifecycleCallbacks.sequential_epoch_end(subgraph.submodules(model))

                if dataset_args.propagate_error:
                    # this pass does not trigger modifier hooks
                    # and is only used for capturing outputs of compressed modules
                    with HooksMixin.disable_hooks():
                        for batch_idx, inputs in _get_batches(
                            activations,
                            num_batches,
                            subgraph.input_names,
                            prop_desc,
                            sequential_activation_prefetch,
                        ):
                            output = subgraph.forward(model, **inputs)
                            if subgraph_index < num_subgraphs - 1:
                                activations.update(batch_idx, output)
                                activations.delete(batch_idx, subgraph.consumed_names)

                if (
                    dataset_args.moe_eager_linearization_and_repack
                    and dataset_args.repack_moe_layers
                ):
                    repack_moe_subgraph(model, subgraph_modules)

                # Offloading is independent of the next subgraph unless modules are
                # shared, so let it run while the next subgraph onloads and calibrates.
                offload_modules = dict(subgraph_modules)
                if sequential_module_prefetch:
                    offload_future = offload_executor.submit(
                        offload,
                        offload_modules,
                        dict(offload_kwargs),
                    )
                    stack.callback(offload_future.result)
                    pending_offloads.append(
                        (
                            set(offload_modules.values()),
                            offload_future,
                        )
                    )
                else:
                    offload(offload_modules, dict(offload_kwargs))
                #######################
                #### END OF ONLOAD ####
                #######################

            if sequential_module_prefetch:
                # Wait for offloads to finish before final model-wide repacking.
                for _, future in pending_offloads:
                    future.result()
                pending_offloads.clear()

            if (
                not dataset_args.moe_eager_linearization_and_repack
                and dataset_args.repack_moe_layers
            ):
                repack_moe_model(model)

            # redundant, finish any remaining compression
            LifecycleCallbacks.calibration_end()
