import pytest
from loguru import logger

from llmcompressor.modifiers.autoround import AutoRoundModifier
from llmcompressor.modifiers.pruning import REAPPruningModifier
from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
from llmcompressor.modifiers.transform import (
    AWQModifier,
    QuIPModifier,
    SmoothQuantModifier,
    SpinQuantModifier,
)
from llmcompressor.pipelines import (
    CalibrationPipeline,
    DataFreePipeline,
    IndependentPipeline,
    SequentialPipeline,
)


@pytest.mark.parametrize(
    "modifiers,exp_pipeline",
    [
        ([QuantizationModifier(scheme="FP8")], SequentialPipeline),
        ([QuantizationModifier(scheme="W4A16")], DataFreePipeline),
        (
            [QuantizationModifier(scheme="W4A16", weight_observer="imatrix_mse")],
            SequentialPipeline,
        ),
        (
            [QuantizationModifier(scheme="W4A16", observer={"weights": "imatrix_mse"})],
            SequentialPipeline,
        ),
        ([GPTQModifier(scheme="FP8")], SequentialPipeline),
        ([GPTQModifier(scheme="W4A16")], SequentialPipeline),
        ([SmoothQuantModifier(), GPTQModifier(scheme="W4A16")], IndependentPipeline),
        (
            [SmoothQuantModifier(), QuantizationModifier(scheme="W4A16")],
            IndependentPipeline,
        ),
        ([AWQModifier(), QuantizationModifier(scheme="W4A16")], IndependentPipeline),
        ([AWQModifier(), QuantizationModifier(scheme="FP8")], IndependentPipeline),
        ([AWQModifier(), GPTQModifier(scheme="W4A16")], IndependentPipeline),
        (
            [REAPPruningModifier(sparsity=0.5, report_path="reap.json")],
            SequentialPipeline,
        ),
        ([QuIPModifier()], DataFreePipeline),
        ([SpinQuantModifier()], DataFreePipeline),
        ([QuIPModifier(), QuantizationModifier(scheme="FP8")], SequentialPipeline),
        ([QuIPModifier(), QuantizationModifier(scheme="W4A16")], DataFreePipeline),
        ([AutoRoundModifier()], SequentialPipeline),
    ],
)
def test_infer_pipeline(modifiers, exp_pipeline):
    pipeline = CalibrationPipeline.from_modifiers(modifiers)
    assert isinstance(pipeline, exp_pipeline)


@pytest.mark.parametrize(
    "modifiers",
    [
        [AWQModifier(), GPTQModifier(scheme="W4A16")],
        [SmoothQuantModifier(), GPTQModifier(scheme="W4A16")],
    ],
)
def test_deferred_transform_quantization_defaults_to_independent_but_allows_override(
    modifiers,
):
    messages = []
    handler_id = logger.add(messages.append, level="WARNING", format="{message}")
    try:
        pipeline = CalibrationPipeline.from_modifiers(modifiers, user="sequential")
    finally:
        logger.remove(handler_id)

    assert isinstance(pipeline, SequentialPipeline)
    assert any(
        "independent calibration pipeline is recommended" in message
        for message in messages
    )
