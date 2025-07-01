from typing import TYPE_CHECKING

import torch
from loguru import logger
from torch.utils.data.dataloader import DataLoader

from llmcompressor.core import active_session
from llmcompressor.pipelines.registry import CalibrationPipeline
from llmcompressor.utils.helpers import patch_attr

if TYPE_CHECKING:
    from llmcompressor.args.dataset_arguments import DatasetArguments

__all__ = ["IndependentPipeline"]


@CalibrationPipeline.register("independent")
class IndependentPipeline(CalibrationPipeline):
    @staticmethod
    def __call__(
        model: torch.nn.Module,
        dataloader: DataLoader,
        dataset_args: "DatasetArguments",
    ):
        """
        Data pipeline where each modifier is assigned its own calibration epoch and data
        pipeline

        :param model: model being calibrated
        :param dataloader: loads data for calibration
        :param dataset_args: dataset arguments relevant to pipelines
        """
        _logger = logger.patch(lambda r: r.update(function="IndependentPipeline"))

        session = active_session()
        modifiers = session.lifecycle.recipe.modifiers
        with patch_attr(session.lifecycle.recipe, "modifiers", None):
            for modifier in modifiers:
                mod_type = type(modifier).__name__
                session.lifecycle.recipe.modifiers = [modifier]
                pipeline = CalibrationPipeline.from_modifiers([modifier])
                pipeline_name = pipeline.__class__.__name__
                _logger.info(f"Inferred `{pipeline_name}` for `{mod_type}`")

                pipeline(model, dataloader, dataset_args)

                # restore modifiers on exit so model can be compressed based on recipe
