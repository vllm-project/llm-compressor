from dataclasses import replace

import torch
from torch.utils.data.dataloader import DataLoader

from llmcompressor.args.dataset_arguments import DatasetArguments
from llmcompressor.pipelines.registry import CalibrationPipeline
from llmcompressor.pipelines.sequential import SequentialPipeline

__all__ = ["BasicPipeline", "run_calibration"]


@CalibrationPipeline.register("basic")
class BasicPipeline(CalibrationPipeline):
    @staticmethod
    def __call__(
        model: torch.nn.Module,
        dataloader: DataLoader,
        dataset_args: DatasetArguments | None,
    ):
        """
        Run a basic data pipeline.

        Calibrate the entire model as a single sequential subgraph, without tracing
        its forward method or propagating compression error between layers.

        :param model: model being calibrated
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
        SequentialPipeline()(model, dataloader, dataset_args)


def run_calibration(model: torch.nn.Module, dataloader: DataLoader):
    pipeline = BasicPipeline()
    pipeline(model, dataloader, None)
