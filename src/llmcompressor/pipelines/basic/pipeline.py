import contextlib
from typing import TYPE_CHECKING, Union

import torch
import tqdm
from compressed_tensors.offload import dispatch_model, get_execution_device
from torch.utils.data.dataloader import DataLoader

from llmcompressor.core import LifecycleCallbacks, active_session
from llmcompressor.pipelines.registry import CalibrationPipeline
from llmcompressor.pytorch.utils.helpers import tensors_to_device
from llmcompressor.utils import calibration_forward_context

if TYPE_CHECKING:
    from llmcompressor.args.dataset_arguments import DatasetArguments

__all__ = ["BasicPipeline", "run_calibration"]


@CalibrationPipeline.register("basic")
class BasicPipeline(CalibrationPipeline):
    @staticmethod
    def __call__(
        model: torch.nn.Module,
        dataloader: DataLoader,
        dataset_args: Union["DatasetArguments", None],
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
        session = active_session()
        dispatch_model(model)  # basic dispatch is identical to generation
        model_device = get_execution_device(model)
        use_loss_mask = (
            getattr(dataset_args, "use_loss_mask", False) if dataset_args else False
        )

        # Initialize loss_masks list for AWQ masking support
        if use_loss_mask:
            session.state.loss_masks = []

        LifecycleCallbacks.calibration_epoch_start()

        with contextlib.ExitStack() as stack:
            stack.enter_context(calibration_forward_context(model))
            for batch_idx, batch in enumerate(
                tqdm.tqdm(dataloader, desc="Calibrating")
            ):
                # Collect loss mask from this batch before moving to device
                if use_loss_mask:
                    session.state.loss_masks.append(batch.get("loss_mask"))

                # Set current batch index before forward pass
                session.state.current_batch_idx = batch_idx

                batch = tensors_to_device(batch, model_device)
                model(**batch)

        LifecycleCallbacks.calibration_epoch_end()


def run_calibration(model: torch.nn.Module, dataloader: DataLoader):
    pipeline = BasicPipeline()
    pipeline(model, dataloader, None)
