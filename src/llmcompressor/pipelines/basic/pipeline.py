from dataclasses import replace
from typing import TYPE_CHECKING

from torch.utils.data.dataloader import DataLoader

from llmcompressor.args.dataset_arguments import DatasetArguments
from llmcompressor.pipelines.registry import CalibrationPipeline
from llmcompressor.pipelines.sequential import SequentialPipeline

if TYPE_CHECKING:
    from llmcompressor.pipelines.sequential.lazy import WholeModelMetaTransform

__all__ = ["BasicPipeline", "run_calibration"]


@CalibrationPipeline.register("basic")
class BasicPipeline(CalibrationPipeline):
    @staticmethod
    def __call__(
        whole_model_transform: "WholeModelMetaTransform",
        dataloader: DataLoader,
        dataset_args: DatasetArguments | None,
    ):
        """
        Run a basic data pipeline.

        Calibrate the entire model as a single sequential subgraph, without tracing
        its forward method or propagating compression error between layers.

        :param whole_model_transform: owns the initialized meta model and source model
        :param dataloader: loads data for calibration
        :param dataset_args: dataset arguments relevant to pipelines
        """
        dataset_args = replace(
            dataset_args if dataset_args is not None else DatasetArguments(),
            sequential_targets=[""],  # root module
            sequential_targets_per_subgraph=1,
            propagate_error=False,
            log_sequential_error=False,
        )
        SequentialPipeline()(
            whole_model_transform,
            dataloader,
            dataset_args,
        )


def run_calibration(
    whole_model_transform: "WholeModelMetaTransform",
    dataloader: DataLoader,
):
    """Run basic calibration using the transform created before initialization."""
    pipeline = BasicPipeline()
    pipeline(
        whole_model_transform,
        dataloader,
        None,
    )
