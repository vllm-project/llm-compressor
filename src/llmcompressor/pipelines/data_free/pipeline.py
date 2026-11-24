from typing import TYPE_CHECKING, Optional

import torch
from compressed_tensors.offload import set_onload_device
from torch.utils.data.dataloader import DataLoader

from llmcompressor.core.session_functions import LifecycleCallbacks, active_session
from llmcompressor.pipelines.registry import CalibrationPipeline
from llmcompressor.pipelines.sequential.offload import (
    subgraph_offload_modules,
    subgraph_onload_modules,
)
from llmcompressor.utils.dev import get_main_device

if TYPE_CHECKING:
    from llmcompressor.args.dataset_arguments import DatasetArguments
    from llmcompressor.pipelines.sequential.restructure import RestructureHandler

__all__ = ["DataFreePipeline"]


@CalibrationPipeline.register("datafree")
class DataFreePipeline(CalibrationPipeline):
    @staticmethod
    def __call__(
        model: torch.nn.Module,
        dataloader: Optional[DataLoader],
        dataset_args: "DatasetArguments",
        restructure_handler: "RestructureHandler",
    ):
        """
        A pipeline for data-free calibration

        :param model: model being calibrated
        :param dataloader: loads data for calibration
        :param dataset_args: dataset arguments relevant to pipelines
        :param restructure_handler: handler prepared before modifier initialization
        """
        # some ops are still performed on the model by modifiers
        # we want those ops to occur on the GPU
        onload_device = get_main_device()
        set_onload_device(model, onload_device)
        restructure_handler.configure_onload(onload_device)

        # Materialize through the regular onload path. Callbacks can then stream
        # real weights from their offload caches instead of holding them all on GPU.
        modules = dict(model.named_modules())
        offload_kwargs = subgraph_onload_modules(
            modules, restructure_handler, onload_device
        )
        subgraph_offload_modules(modules, offload_kwargs)

        LifecycleCallbacks.calibration_start()
        if dataset_args.layerwise_decompression:
            session = active_session()
            for modifier in session.lifecycle.recipe.modifiers:
                if hasattr(modifier, "start_layerwise_calibration"):
                    modifier.start_layerwise_calibration(
                        session.state, model, list(model.modules())
                    )
        LifecycleCallbacks.sequential_epoch_end(dict(model.named_modules()))
        LifecycleCallbacks.calibration_end()
