from enum import Enum
from typing import Dict, List, Optional, Tuple

from loguru import logger

from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.obcq.sgpt_mixin import SparsityModifierMixin
from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationMixin
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.pipelines import (
    basic,
    data_free,
    independent,
    layer_sequential,
    sequential,
)
from llmcompressor.typing import PipelineFn

__all__ = ["PIPELINES", "get_pipeline_fn"]

SEQUENTIAL_MODIFIERS = (GPTQModifier, SparsityModifierMixin)


class PipelineName(str, Enum):
    DATA_FREE = "data_free"
    BASIC = "basic"
    SEQUENTIAL = "sequential"
    LAYER_SEQUENTIAL = "layer_sequential"
    INDEPENDENT = "independent"


PIPELINES: Dict[PipelineName, PipelineFn] = {
    PipelineName.DATA_FREE: data_free.run_pipeline,
    PipelineName.BASIC: basic.run_pipeline,
    PipelineName.SEQUENTIAL: sequential.run_pipeline,
    PipelineName.LAYER_SEQUENTIAL: layer_sequential.run_pipeline,
    PipelineName.INDEPENDENT: independent.run_pipeline,
}


def get_pipeline_fn(
    user: Optional[str], modifiers: List[Modifier]
) -> Tuple[str, PipelineFn]:
    inferred_pipeline = (
        None if user == PipelineName.INDEPENDENT else infer_pipeline_fn(modifiers)
    )
    pipeline = user or inferred_pipeline

    if user not in (None, PipelineName.INDEPENDENT) and user != inferred_pipeline:
        logger.warning(
            f"Calibration pipeline is set to `{user}`, but it is recommended to "
            f"use `{inferred_pipeline}`"
        )

    if pipeline not in PIPELINES:
        raise ValueError(
            f"Cannot find `{pipeline}` in registered pipelines {PIPELINES.keys()}"
        )

    return pipeline, PIPELINES[pipeline]


def infer_pipeline_fn(modifiers: List[Modifier]) -> PipelineName:
    if any(isinstance(modifier, AWQModifier) for modifier in modifiers):
        if len(modifiers) > 1:
            raise ValueError(
                "AWQ does not currently support sharing a data pipeline with other "
                "modifiers. Please use oneshot(pipeline='independent')"
            )
        return PipelineName.DATA_FREE

    if any(isinstance(modifier, SEQUENTIAL_MODIFIERS) for modifier in modifiers):
        return PipelineName.SEQUENTIAL

    quant_modifiers = _get_quantization_modifiers(modifiers)
    if len(quant_modifiers) > 1:
        raise ValueError(
            f"Recipe contains more than one quantization modifier ({quant_modifiers})."
            "Please modify your recipe to use at most one quantization modifier"
        )

    if len(quant_modifiers) == 1:
        quant_modifier = quant_modifiers[0]
        config = quant_modifier.resolve_quantization_config()
        if config.requires_calibration_data():
            return PipelineName.BASIC
        else:
            return PipelineName.DATA_FREE

    if any(isinstance(modifier, SmoothQuantModifier) for modifier in modifiers):
        return PipelineName.BASIC

    return PipelineName.DATA_FREE


def _get_quantization_modifiers(modifiers: List[Modifier]) -> List[QuantizationMixin]:
    return [
        modifier
        for modifier in modifiers
        if isinstance(modifier, QuantizationMixin) and modifier.has_config()
    ]
