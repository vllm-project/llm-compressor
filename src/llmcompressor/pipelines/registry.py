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
    inferred = infer_pipeline(modifiers)
    pipeline = resolve_pipeline(user, inferred)

    if pipeline not in PIPELINES:
        raise ValueError(
            f"Cannot find `{pipeline}` in registered pipelines {PIPELINES.keys()}"
        )
    return pipeline, PIPELINES[pipeline]


def infer_pipeline(modifiers: List[Modifier]) -> str:
    if any(isinstance(modifier, SEQUENTIAL_MODIFIERS) for modifier in modifiers):
        return "sequential"

    else:
        return "basic"


def resolve_pipeline(user: str, inferred: str) -> str:
    if user is not None and user != "independent" and user != inferred:
        logger.warning(
            f"Calibration pipeline is set to `{user}`, but it is recommend to "
            f"use `{inferred}`"
        )

    return user or inferred
