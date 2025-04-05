from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from loguru import logger

from llmcompressor.core.utils import recipe_requires_data
from llmcompressor.typing import PipelineFn

if TYPE_CHECKING:
    from llmcompressor.modifiers import Modifier

__all__ = ["get_pipeline_fn", "get_sequential_modifiers"]


def _get_pipelines() -> Dict[str, PipelineFn]:
    # avoid circular imports by avoiding dependency on
    # modifiers (which are imported by individual pipelines)
    from llmcompressor.pipelines import (
        basic,
        independent,
        layer_sequential,
        sequential,
        skip,
    )

    return {
        "sequential": sequential.run_pipeline,
        "layer_sequential": layer_sequential.run_pipeline,
        "basic": basic.run_pipeline,
        "independent": independent.run_pipeline,
        "skip": skip.run_pipeline,
    }


def get_sequential_modifiers() -> Tuple["Modifier", ...]:
    # avoid circular imports
    from llmcompressor.modifiers.obcq.sgpt_mixin import SparsityModifierMixin
    from llmcompressor.modifiers.quantization import GPTQModifier

    return (GPTQModifier, SparsityModifierMixin)


def get_pipeline_fn(
    user: Optional[str], modifiers: List["Modifier"]
) -> Tuple[str, PipelineFn]:
    inferred = infer_pipeline(modifiers)
    pipeline = resolve_pipeline(user, inferred)

    all_pipelines = _get_pipelines()

    if pipeline not in all_pipelines:
        raise ValueError(
            f"Cannot find `{pipeline}` in registered pipelines {all_pipelines.keys()}"
        )
    return pipeline, all_pipelines[pipeline]


def infer_pipeline(modifiers: List["Modifier"]) -> str:
    if any(isinstance(modifier, get_sequential_modifiers()) for modifier in modifiers):
        return "sequential"

    if not recipe_requires_data(modifiers):
        return "skip"

    else:
        return "basic"


def resolve_pipeline(user: str, inferred: str) -> str:
    if user is not None and user != "independent" and user != inferred:
        logger.warning(
            f"Calibration pipeline is set to `{user}`, but it is recommend to "
            f"use `{inferred}`"
        )

    return user or inferred
