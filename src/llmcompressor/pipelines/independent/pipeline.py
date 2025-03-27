from typing import TYPE_CHECKING

import torch
from loguru import logger
from torch.utils.data.dataloader import DataLoader

from llmcompressor.core import active_session
from llmcompressor.core.llmcompressor.globals import SingletonException, get_compressor
from llmcompressor.modifiers.stage import StageModifiers

if TYPE_CHECKING:
    from llmcompressor.args.post_train_arguments import PostTrainArguments

__all__ = ["run_pipeline"]


def run_pipeline(
    model: torch.nn.Module,
    dataloader: DataLoader,
    args: "PostTrainArguments",
):
    # avoid circular import
    from llmcompressor.pipelines.registry import get_pipeline_fn

    try:
        compressor = get_compressor()
    except SingletonException:
        session = active_session()
        compressor = None

    if compressor is not None:
        modifiers = compressor.modifiers

        for index, modifier in enumerate(modifiers):
            mod_type = str(type(modifier).__name__)
            compressor.modifiers = [modifier]

            pipeline, pipeline_fn = get_pipeline_fn(user=None, modifiers=[modifier])
            logger.info(f"Inferred `{pipeline}` calibration pipeline for `{mod_type}`")

            pipeline_fn(model, dataloader, args)

        compressor.modifiers = modifiers

    else:
        modifiers = session.get_modifiers()

        for index, modifier in enumerate(modifiers):
            mod_type = str(type(modifier).__name__)
            session.lifecycle.modifiers = [
                StageModifiers(modifiers=[modifier], group=mod_type, index=index)
            ]

            pipeline, pipeline_fn = get_pipeline_fn(user=None, modifiers=[modifier])
            logger.info(f"Inferred `{pipeline}` calibration pipeline for `{mod_type}`")

            pipeline_fn(model, dataloader, args)

        session.lifecycle.modifiers = modifiers
