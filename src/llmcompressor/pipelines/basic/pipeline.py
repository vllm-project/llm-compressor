import contextlib
from typing import TYPE_CHECKING

import torch
import tqdm
from compressed_tensors.utils import get_execution_device
from torch.utils.data.dataloader import DataLoader

from llmcompressor.core import LifecycleCallbacks
from llmcompressor.modifiers.utils.pytorch_helpers import apply_pad_mask_to_batch
from llmcompressor.pipelines.registry import CalibrationPipeline
from llmcompressor.pytorch.utils.helpers import tensors_to_device
from llmcompressor.utils import calibration_forward_context, dispatch_for_generation
from llmcompressor.utils.helpers import DisableQuantization

if TYPE_CHECKING:
    from llmcompressor.args.dataset_arguments import DatasetArguments

__all__ = ["BasicPipeline", "run_calibration"]


@CalibrationPipeline.register("basic")
class BasicPipeline(CalibrationPipeline):
    @staticmethod
    def __call__(
        model: torch.nn.Module,
        dataloader: DataLoader,
        dataset_args: "DatasetArguments",
    ):
        """
        Run a basic data pipeline.

        Batches are fetched from the data loader and are used to perform forward passes
        through the model. This pipeline is typically used for basic model calibration
        and, unlike the sequential pipelines, does not propagate compression error when
        used to calibrate model compression

        :param model: model being calibrated
        :param dataloader: loads data for calibration
        :param dataset_args: dataset arguments relevant to pipelines
        """
        dispatch_for_generation(model)  # basic dispatch is identical to generation
        model_device = get_execution_device(model)

        LifecycleCallbacks.calibration_epoch_start()

        # disable gradients, kv cache, ect.
        with calibration_forward_context(model):
            with DisableQuantization(
                model
            ) if not dataset_args.propagate_error else contextlib.nullcontext():
                for batch in tqdm.tqdm(dataloader, desc="Calibrating"):
                    batch = apply_pad_mask_to_batch(batch)
                    batch = tensors_to_device(batch, model_device)
                    model(**batch)

        LifecycleCallbacks.calibration_epoch_end()


def run_calibration(model: torch.nn.Module, dataloader: DataLoader):
    from llmcompressor.args.dataset_arguments import DatasetArguments

    pipeline = BasicPipeline()
    pipeline(model, dataloader, DatasetArguments())
