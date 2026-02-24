import contextlib
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Iterator

import torch
from compressed_tensors.utils import disable_offloading
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from llmcompressor.core import LifecycleCallbacks, active_session
from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.pipelines.cache import IntermediatesCache
from llmcompressor.pipelines.registry import CalibrationPipeline
from llmcompressor.pipelines.sequential.helpers import (
    dispatch_for_sequential,
    get_sequential_targets,
    handle_sequential_oom,
    trace_subgraphs,
)
from llmcompressor.utils.dev import get_main_device
from llmcompressor.utils.helpers import (
    DISABLE_QAC_MODIFIERS,
    DisableQuantization,
    calibration_forward_context,
)

if TYPE_CHECKING:
    from llmcompressor.args.dataset_arguments import DatasetArguments

__all__ = ["SequentialPipeline"]


def _get_batches(
    activations: IntermediatesCache,
    num_batches: int,
    input_names: list[str],
    desc: str,
    use_prefetch: bool = False,
) -> Iterator[tuple[int, dict]]:
    """
    Yield (batch_idx, inputs) with the next batch optionally prefetched in a
    background thread to overlap fetch (onload from offload device) with the
    main-thread forward pass.
    """
    if not use_prefetch:
        for batch_idx in tqdm(range(num_batches), desc=desc):
            inputs = activations.fetch(batch_idx, input_names)
            yield batch_idx, inputs
        return
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = None
        for batch_idx in tqdm(range(num_batches), desc=desc):
            if future is not None:
                inputs = future.result()
            else:
                inputs = activations.fetch(batch_idx, input_names)
            if batch_idx + 1 < num_batches:
                future = executor.submit(activations.fetch, batch_idx + 1, input_names)
            else:
                future = None
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
        dispatch_for_sequential(model, onload_device)

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

        with contextlib.ExitStack() as stack:
            stack.enter_context(calibration_forward_context(model))
            # Optionally disable quantization
            if not dataset_args.quantization_aware_calibration or disable_qac:
                stack.enter_context(DisableQuantization(model))

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

            for subgraph_index, subgraph in enumerate(subgraphs):
                # prepare tqdm description texts
                calib_desc = f"({subgraph_index + 1}/{num_subgraphs}): Calibrating"
                prop_desc = f"({subgraph_index + 1}/{num_subgraphs}): Propagating"

                # reduce memory movement by keeping modules onloaded
                num_batches = len(dataloader)
                use_prefetch = getattr(dataset_args, "sequential_prefetch", False)
                with disable_offloading():
                    # do a preliminary pass to trigger modifier hooks
                    for batch_idx, inputs in _get_batches(
                        activations,
                        num_batches,
                        subgraph.input_names,
                        calib_desc,
                        use_prefetch,
                    ):
                        session.state.current_batch_idx = batch_idx
                        subgraph.forward(model, **inputs)

                    LifecycleCallbacks.sequential_epoch_end(subgraph)

                    # this pass does not trigger modifier hooks
                    # and is only used for capturing outputs of newly compressed modules
                    with HooksMixin.disable_hooks():
                        for batch_idx, inputs in _get_batches(
                            activations,
                            num_batches,
                            subgraph.input_names,
                            prop_desc,
                            use_prefetch,
                        ):
                            output = subgraph.forward(model, **inputs)
                            if subgraph_index < num_subgraphs - 1:
                                activations.update(batch_idx, output)
                                activations.delete(batch_idx, subgraph.consumed_names)

            # redundant, finish any remaining compression
            LifecycleCallbacks.calibration_epoch_end()
