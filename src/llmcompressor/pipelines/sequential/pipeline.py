from typing import TYPE_CHECKING

import torch
from compressed_tensors.utils import align_modules
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from llmcompressor.core import LifecycleCallbacks, active_session
from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.pipelines.cache import IntermediatesCache
from llmcompressor.pipelines.registry import CalibrationPipeline
from llmcompressor.pipelines.sequential.helpers import (
    get_targets_from_modifiers,
    set_execution_device,
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
        model: torch.nn.Module, dataloader: DataLoader, args: "DatasetArguments"
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

        # infer sequential targets
        modifiers = session.get_modifiers()
        sequential_targets, ignore = get_targets_from_modifiers(modifiers, model)

        # trace subgraphs
        sample_input = next(iter(dataloader))
        subgraphs = trace_subgraphs(model, sample_input, sequential_targets, ignore)

        # set execution device for sequential onloading
        model = set_execution_device(model, args.oneshot_device)

        with calibration_forward_context(model), DisableQuantization(model):
            # prepare intermediates cache
            LifecycleCallbacks.calibration_epoch_start()
            cache = IntermediatesCache.from_dataloader(dataloader, args.oneshot_device)

            num_subgraphs = len(subgraphs)
            for subgraph_index, subgraph in enumerate(subgraphs):
                # prepare tqdm description texts
                calib_desc = f"({subgraph_index + 1}/{num_subgraphs}): Calibrating"
                prop_desc = f"({subgraph_index + 1}/{num_subgraphs}): Propagating"

                # sequential onloading: only onload one layer at a time
                with align_modules(subgraph.modules, args.oneshot_device):
                    # do an preliminary pass to trigger modifier hooks
                    for batch_idx in tqdm(range(len(dataloader)), desc=calib_desc):
                        inputs = cache.fetch(batch_idx, subgraph.input_names)
                        subgraph.forward(model, **inputs)

                    # trigger compression
                    LifecycleCallbacks.sequential_epoch_end()

                    # this pass does not trigger modifier hooks
                    # and is only used for capturing outputs from compressed modules
                    with HooksMixin.disable_hooks():
                        for batch_idx in tqdm(range(len(dataloader)), desc=prop_desc):
                            inputs = cache.fetch(batch_idx, subgraph.input_names)
                            output = subgraph.forward(model, **inputs)

                            if subgraph_index < num_subgraphs - 1:
                                cache.update(batch_idx, output)
                                cache.delete(batch_idx, subgraph.consumed_names)

            # redudant, finish any remaining compression
            LifecycleCallbacks.calibration_epoch_end()
