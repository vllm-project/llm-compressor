from typing import TYPE_CHECKING, Optional

import torch
from compressed_tensors.offload import set_onload_device
from torch.utils.data.dataloader import DataLoader

from llmcompressor.core.session_functions import LifecycleCallbacks
from llmcompressor.pipelines.registry import CalibrationPipeline
from llmcompressor.utils.dev import get_main_device

if TYPE_CHECKING:
    from llmcompressor.args.dataset_arguments import DatasetArguments

__all__ = ["DataFreePipeline"]


@CalibrationPipeline.register("datafree")
class DataFreePipeline(CalibrationPipeline):
    @staticmethod
    def __call__(
        model: torch.nn.Module,
        dataloader: Optional[DataLoader],
        dataset_args: "DatasetArguments",
    ):
        """
        A pipeline for data-free calibration

        :param model: model being calibrated
        :param dataloader: loads data for calibration
        :param dataset_args: dataset arguments relevant to pipelines
        """
        # a model loaded compressed for sequential decompression must be decompressed
        # fully here, since this pipeline does not decompress per subgraph
        from llmcompressor.pipelines.sequential.decompression import (
            ensure_dense_for_nonsequential,
        )

        ensure_dense_for_nonsequential(model)

        # some ops are still performed on the model by modifiers
        # we want those ops to occur on the GPU
        onload_device = get_main_device()
        set_onload_device(model, onload_device)

        LifecycleCallbacks.calibration_start()
        LifecycleCallbacks.sequential_epoch_end(list(model.modules()))
        LifecycleCallbacks.calibration_end()
