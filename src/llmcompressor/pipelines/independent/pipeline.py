import torch
from loguru import logger
from torch.utils.data.dataloader import DataLoader

from llmcompressor.core import active_session
from llmcompressor.modifiers.stage import StageModifiers

__all__ = ["run_pipeline"]


def run_pipeline(
    model: torch.nn.Module,
    dataloader: DataLoader,
):
    # avoid circular import
    from llmcompressor.pipelines.registry import get_pipeline_fn

    session = active_session()

    modifiers = session.get_modifiers()
    for modifier in modifiers:
        session.lifecycle.modifiers = StageModifiers(modifiers=[modifier])

        pipeline, pipeline_fn = get_pipeline_fn(user=None, modifiers=[modifier])
        logger.info(f"Inferred {pipeline} calibration pipeline for {type(modifier)}")

        pipeline_fn(model, dataloader)
