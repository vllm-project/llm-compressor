import contextlib
from typing import TYPE_CHECKING

import torch
from loguru import logger
from torch.utils.data.dataloader import DataLoader

from llmcompressor.core import LifecycleCallbacks, active_session
from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.pipelines.cache import IntermediatesCache
from llmcompressor.pipelines.registry import CalibrationPipeline
from llmcompressor.pipelines.sequential.helpers import (
    handle_sequential_oom,
    trace_subgraphs,
)
from llmcompressor.pipelines.sequential.pipeline import _get_batches
from llmcompressor.pipelines.streaming.checkpoint import (
    CheckpointMap,
    commit_staged,
    materialize_buffers,
    materialize_modules,
    release_modules,
    stage_modules,
)
from llmcompressor.pipelines.streaming.prefetch import SubgraphPrefetcher
from llmcompressor.utils.dev import get_main_device
from llmcompressor.utils.helpers import DisableQuantization, calibration_forward_context
from llmcompressor.utils.pytorch.module import infer_sequential_targets

if TYPE_CHECKING:
    from llmcompressor.args.dataset_arguments import DatasetArguments

__all__ = ["StreamingPipeline"]


@CalibrationPipeline.register("streaming")
class StreamingPipeline(CalibrationPipeline):
    @staticmethod
    @handle_sequential_oom
    def __call__(
        model: torch.nn.Module,
        dataloader: DataLoader,
        dataset_args: "DatasetArguments",
    ):
        """
        Run a sequential data pipeline which materializes weights directly from
        the original checkpoint, one subgraph at a time:

        1. The model is expected on the meta device (``device_map="meta"``);
           no weights are loaded before this pipeline runs
        2. Before each subgraph executes, its parameters are read in bulk from
           the original safetensors shards; the next subgraph's weights are
           prefetched in a background thread during calibration
        3. After each subgraph is calibrated and compressed, its parameters are
           moved to the offload device for the final save

        Unquantized weights are never written to a secondary disk location.
        Models whose checkpoints require tensor conversions on load are not
        supported (see `CheckpointReferenceError`); use the "sequential"
        pipeline for those.

        :param model: model being calibrated
        :param dataloader: loads data for calibration
        :param dataset_args: dataset arguments relevant to pipelines
        """
        session = active_session()
        onload_device = get_main_device()
        offload_device = torch.device(dataset_args.sequential_offload_device)

        ckpt_map = CheckpointMap.from_model(model)
        # buffers without a checkpoint source (e.g. non-persistent rotary
        # inv_freq) are recomputed once on the execution device
        materialize_buffers(model, onload_device, ckpt_map)

        # AutoRoundModifier optimizes each layer independently using its own
        # forward passes, so quantization error should not be propagated between
        # layers during the calibration stage
        modifiers = session.lifecycle.recipe.modifiers
        if any(type(m).__name__ == "AutoRoundModifier" for m in modifiers):
            dataset_args.propagate_error = False

        # weight release mode after each subgraph: "cpu" keeps (possibly
        # modifier-updated) weights in host memory; "meta" drops
        # checkpoint-backed weights back to the meta device (they are
        # re-readable from the original checkpoint), keeping host memory flat
        # for models larger than RAM. "meta" is only valid when no modifier
        # mutates weights.
        release_mode = dataset_args.streaming_release
        weight_mutating = {
            "GPTQModifier",
            "SparseGPTModifier",
            "AWQModifier",
            "SmoothQuantModifier",
            "AutoRoundModifier",
            "SpinQuantModifier",
            "QuIPModifier",
        }
        if release_mode == "meta" and any(
            type(m).__name__ in weight_mutating for m in modifiers
        ):
            logger.warning(
                "streaming_release='meta' requires modifiers which "
                "do not mutate weights; falling back to 'cpu'"
            )
            release_mode = "cpu"
        weight_release_device = torch.device(release_mode)

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

            # modules outside every subgraph (e.g. embed_tokens, whose call is
            # not traced as a call_module node) are materialized once upfront
            # and stay resident
            claimed = set()
            for subgraph in subgraphs:
                claimed.update(subgraph.submodules(model))
            unclaimed = [
                module
                for module in model.modules()
                if module not in claimed
                and any(
                    param is not None and param.device.type == "meta"
                    for param in module._parameters.values()
                )
            ]
            if unclaimed:
                materialize_modules(model, unclaimed, ckpt_map, onload_device)

            # synchronously stage the first subgraph; later subgraphs are
            # staged by the prefetcher during the previous subgraph's passes
            staged = stage_modules(model, subgraphs[0].submodules(model), ckpt_map)
            prefetcher = SubgraphPrefetcher(model, ckpt_map)

            for subgraph_index, subgraph in enumerate(subgraphs):
                commit_staged(staged, onload_device)
                if subgraph_index + 1 < num_subgraphs:
                    prefetcher.start(subgraphs[subgraph_index + 1].submodules(model))

                # prepare tqdm description texts
                calib_desc = f"({subgraph_index + 1}/{num_subgraphs}): Calibrating"
                prop_desc = f"({subgraph_index + 1}/{num_subgraphs}): Propagating"

                submodules = subgraph.submodules(model)
                num_batches = len(dataloader)

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

                    if not dataset_args.propagate_error:
                        if subgraph_index < num_subgraphs - 1:
                            activations.update(batch_idx, outputs)
                            activations.delete(batch_idx, subgraph.consumed_names)

                LifecycleCallbacks.sequential_epoch_end(submodules)

                if dataset_args.propagate_error:
                    # this pass does not trigger modifier hooks
                    # and is only used for capturing outputs of compressed modules
                    with HooksMixin.disable_hooks():
                        for batch_idx, inputs in _get_batches(
                            activations,
                            num_batches,
                            subgraph.input_names,
                            prop_desc,
                            sequential_prefetch,
                        ):
                            output = subgraph.forward(model, **inputs)
                            if subgraph_index < num_subgraphs - 1:
                                activations.update(batch_idx, output)
                                activations.delete(batch_idx, subgraph.consumed_names)

                # keep (compressed) params on the offload device for the final
                # save and free execution-device memory
                release_modules(
                    submodules, weight_release_device, ckpt_map=ckpt_map, model=model
                )

                if subgraph_index + 1 < num_subgraphs:
                    staged = prefetcher.join()

            # recipe-targeted modules which were never traced into a subgraph
            # (e.g. a vision tower when calibrating with text-only data) are
            # compressed during `calibration_end`; materialize them first
            remaining = [
                module
                for module in model.modules()
                if any(
                    param is not None and param.device.type == "meta"
                    for param in module._parameters.values()
                )
            ]
            if remaining:
                materialize_modules(model, remaining, ckpt_map, onload_device)

            # redundant, finish any remaining compression
            LifecycleCallbacks.calibration_end()
            if remaining:
                release_modules(
                    remaining, weight_release_device, ckpt_map=ckpt_map, model=model
                )
