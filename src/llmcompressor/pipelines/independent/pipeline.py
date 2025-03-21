from typing import TYPE_CHECKING, List

import torch
import torch.utils.data.dataloader

from llmcompressor.modifiers.utils.hooks import HooksMixin

if TYPE_CHECKING:
    from llmcompressor.modifiers import Modifier

__all__ = ["run_pipeline"]

resolve_pipeline_kwargs = None


def run_pipeline(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    modifiers: List["Modifier"],
    independent_pipelines: List[str],
):
    """
    Pipeline which runs each modifier independently
    """
    from llmcompressor.pipelines import get_pipeline

    for pipeline, modifier in zip(independent_pipelines, modifiers):
        pipeline = get_pipeline(pipeline)
        pipeline_kwargs = resolve_pipeline_kwargs(modifier)

        modifier_hooks = modifier._hooks
        with HooksMixin.disable_hooks(keep=modifier_hooks):
            pipeline(**pipeline_kwargs)
