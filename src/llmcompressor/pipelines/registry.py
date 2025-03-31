from typing import Dict, List, Optional, Tuple

from loguru import logger

from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.obcq.sgpt_mixin import SparsityModifierMixin
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.pipelines import basic, independent, layer_sequential, sequential
from llmcompressor.typing import PipelineFn

__all__ = ["PIPELINES", "get_pipeline_fn"]

SEQUENTIAL_MODIFIERS = (GPTQModifier, SparsityModifierMixin)

PIPELINES: Dict[str, PipelineFn] = {
    "sequential": sequential.run_pipeline,
    "layer_sequential": layer_sequential.run_pipeline,
    "basic": basic.run_pipeline,
    "independent": independent.run_pipeline,
}


def get_pipeline_fn(
    user: Optional[str], modifiers: List[Modifier]
) -> Tuple[str, PipelineFn]:
    inferred_pipeline = infer_pipeline_fn(modifiers)

    if user is not None and user != inferred_pipeline:
        logger.warning(
            f"Calibration pipeline is set to `{user}`, but it is recommend to "
            f"use `{inferred_pipeline}`"
        )

    pipeline = user or inferred_pipeline

    if pipeline not in PIPELINES:
        raise ValueError(
            f"Cannot find `{pipeline}` in registered pipelines {PIPELINES.keys()}"
        )

    return pipeline, PIPELINES[pipeline]


def infer_pipeline_fn(modifiers: List[Modifier]) -> str:
    if any(isinstance(modifier, SEQUENTIAL_MODIFIERS) for modifier in modifiers):
        return "sequential"

    else:
        return "basic"
