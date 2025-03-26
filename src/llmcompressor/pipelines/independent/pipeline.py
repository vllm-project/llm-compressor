import torch
from loguru import logger
from torch.utils.data.dataloader import DataLoader

from llmcompressor.core import active_session
from llmcompressor.modifiers.stage import StageModifiers

from llmcompressor.core.llmcompressor.globals import get_compressor

__all__ = ["run_pipeline"]


def run_pipeline(
    model: torch.nn.Module,
    dataloader: DataLoader,
):
    # avoid circular import
    from llmcompressor.pipelines.registry import get_pipeline_fn

    try:
        compressor = get_compressor()
    except:
        session = active_session()
        compressor = None

    if compressor is not None:
        modifiers = compressor.modifiers
        
        for index, modifier in enumerate(modifiers):
            mod_type = str(type(modifier).__name__)
            compressor.modifiers = [modifier]

            pipeline, pipeline_fn = get_pipeline_fn(user=None, modifiers=[modifier])
            logger.info(f"Inferred `{pipeline}` calibration pipeline for `{mod_type}`")

            pipeline_fn(model, dataloader)

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

            pipeline_fn(model, dataloader)

        session.lifecycle.modifiers = modifiers
