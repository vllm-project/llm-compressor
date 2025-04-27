from typing import TYPE_CHECKING

import torch
from loguru import logger
from torch.utils.data.dataloader import DataLoader

from llmcompressor.core import active_session
from llmcompressor.modifiers.stage import StageModifiers
from llmcompressor.utils.helpers import patch_attr

if TYPE_CHECKING:
    from llmcompressor.args.dataset_arguments import DatasetArguments

__all__ = ["run_pipeline"]


def run_pipeline(
    model: torch.nn.Module,
    dataloader: DataLoader,
    dataset_args: "DatasetArguments",
):
    # avoid circular import
    from llmcompressor.pipelines.registry import get_pipeline_fn

    session = active_session()

    modifiers = session.get_modifiers()
    with patch_attr(session.lifecycle, "modifiers", None):
        for index, modifier in enumerate(modifiers):
            mod_type = str(type(modifier).__name__)
            session.lifecycle.modifiers = [
                StageModifiers(modifiers=[modifier], group=mod_type, index=index)
            ]

            pipeline, pipeline_fn = get_pipeline_fn(user=None, modifiers=[modifier])
            logger.info(f"Inferred `{pipeline}` calibration pipeline for `{mod_type}`")

            pipeline_fn(model, dataloader, dataset_args)

        # restore modifiers on exit for proper model compression inference from recipe
