import contextlib
from typing import TYPE_CHECKING

import torch
import tqdm
from compressed_tensors.utils import disable_offloading, get_execution_device
from torch.utils.data.dataloader import DataLoader

from llmcompressor.core import LifecycleCallbacks, active_session
from llmcompressor.modeling.prepare import moe_calibration_context
from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.pipelines.cache import IntermediatesCache
from llmcompressor.pipelines.layer_sequential.helpers import (
    capture_first_layer_intermediates,
    match_modules,
    maybe_inject_pos_embeddings,
    to_next_layer_kwargs,
)
from llmcompressor.pipelines.registry import CalibrationPipeline
from llmcompressor.pipelines.sequential.helpers import (
    dispatch_for_sequential,
    get_sequential_targets,
)
from llmcompressor.utils.helpers import DisableQuantization, calibration_forward_context

if TYPE_CHECKING:
    from llmcompressor.args.dataset_arguments import DatasetArguments


__all__ = ["LayerSequentialPipeline"]


@CalibrationPipeline.register("layer_sequential")
class LayerSequentialPipeline(CalibrationPipeline):
    @staticmethod
    def __call__(
        model: torch.nn.Module, dataloader: DataLoader, dataset_args: "DatasetArguments"
    ):
        """
        Run a layer-wise sequential data pipeline according to the following steps:

        1. Layers are identified according to `sequential_targets`
        2. A hook is attached to the first layer. This hook raises an exception which is
            then caught and used to capture the input arguments to the first layer
        3. The inputs to the first layer are used to calibrate the first layer, and the
            output of the previous layer is used as inputs to calibrate the next layer

        This pipeline requires that the model have distinct layers defined in its
        architecture and that the outputs of the previous layer are exactly the inputs
        to the next layer. This is violated by encoder-decoder architectures, among
        others.

        If your model architecture violates these assumptions, consider using the
        sequential pipeline (see llmcompressor.pipelines.sequential). Architectures
        which are known to fail these assumptions include GPT-J and most vision models

        :param model: model being calibrated
        :param dataloader: loads data for calibration
        :param dataset_args: dataset arguments relevant to pipelines
        """
        session = active_session()

        # prepare model for sequential onloading
        dispatch_for_sequential(model)
        model_device = get_execution_device(model)

        # find layers
        modifiers = session.lifecycle.recipe.modifiers
        sequential_targets = get_sequential_targets(modifiers, model, dataset_args)
        layers = match_modules(model, sequential_targets)

        LifecycleCallbacks.calibration_epoch_start()

        # TODO: remove this to enable quantization aware calibration for GPTQ and AWQ
        disable_qac = any(
            type(mod).__name__ in ["GPTQModifier", "AWQModifier"]
            for mod in session.lifecycle.recipe.modifiers
        )

        with contextlib.ExitStack() as stack:
            stack.enter_context(calibration_forward_context(model))
            if not dataset_args.quantization_aware_calibration or disable_qac:
                stack.enter_context(DisableQuantization(model))

            if dataset_args.calibrate_moe_context:
                moe_calibration_context(model, stack)

            # prepare intermediates cache
            intermediates: IntermediatesCache = capture_first_layer_intermediates(
                model, layers[0], dataloader, model_device
            )

            num_layers = len(layers)
            for layer_index, layer in enumerate(layers):
                # prepare tqdm description texts
                calib_desc = f"({layer_index + 1}/{num_layers}): Calibrating"
                prop_desc = f"({layer_index + 1}/{num_layers}): Propagating"

                # reduce memory movement by keeping modules onloaded
                with disable_offloading():
                    # do a preliminary pass to trigger modifier hooks
                    for batch_idx in tqdm.tqdm(range(len(dataloader)), desc=calib_desc):
                        inputs = intermediates.fetch(batch_idx)
                        layer(**inputs)

                    LifecycleCallbacks.sequential_epoch_end()

                    # this pass does not trigger modifier hooks
                    # and is only used for capturing outputs from
                    # newly compressed modules
                    with HooksMixin.disable_hooks():
                        for batch_idx in tqdm.tqdm(
                            range(len(dataloader)), desc=prop_desc
                        ):
                            inputs = intermediates.fetch(batch_idx)
                            output = layer(**inputs)

                            if layer_index < num_layers - 1:
                                next_layer = layers[layer_index + 1]
                                output = to_next_layer_kwargs(output, next_layer)
                                output = maybe_inject_pos_embeddings(
                                    output, next_layer, inputs
                                )

                                intermediates.delete(batch_idx)
                                intermediates.update(batch_idx, output)

            # redundant, finish any remaining compression
            LifecycleCallbacks.calibration_epoch_end()
