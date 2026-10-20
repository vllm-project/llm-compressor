import pytest

from llmcompressor.modifiers.autoround import AutoRoundModifier
from llmcompressor.modifiers.pruning import SparseGPTModifier, WandaPruningModifier
from llmcompressor.modifiers.qad import QADModifier
from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
from llmcompressor.modifiers.transform import (
    AWQModifier,
    QuIPModifier,
    SmoothQuantModifier,
    SpinQuantModifier,
)
from llmcompressor.pipelines import (
    BasicPipeline,
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
        ([SmoothQuantModifier(), GPTQModifier(scheme="W4A16")], SequentialPipeline),
        ([AWQModifier(), QuantizationModifier(scheme="W4A16")], SequentialPipeline),
        ([AWQModifier(), QuantizationModifier(scheme="FP8")], SequentialPipeline),
        ([SparseGPTModifier(sparsity=1.0)], SequentialPipeline),
        ([WandaPruningModifier(sparsity=1.0)], SequentialPipeline),
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


@pytest.mark.parametrize("modifier_cls", [SparseGPTModifier, WandaPruningModifier])
def test_owl_requires_basic_pipeline(modifier_cls):
    modifier = modifier_cls(
        sparsity=0.5, sparsity_profile="owl", owl_m=5, owl_lmbda=0.05
    )
    assert isinstance(CalibrationPipeline.from_modifiers([modifier]), BasicPipeline)
    with pytest.raises(ValueError, match="requires pipeline='basic'"):
        CalibrationPipeline.from_modifiers([modifier], user="sequential")
    assert isinstance(
        CalibrationPipeline.from_modifiers([modifier], user="independent"),
        IndependentPipeline,
    )


@pytest.mark.parametrize("modifier", [AutoRoundModifier(), QADModifier()])
def test_sequential_modifiers_reject_incompatible_pipelines(modifier):
    assert isinstance(
        CalibrationPipeline.from_modifiers([modifier]), SequentialPipeline
    )
    with pytest.raises(ValueError, match="requires pipeline='sequential'"):
        CalibrationPipeline.from_modifiers([modifier], user="basic")


def test_incompatible_pipeline_requirements_are_reported():
    modifiers = [
        SparseGPTModifier(
            sparsity=0.5, sparsity_profile="owl", owl_m=5, owl_lmbda=0.05
        ),
        AutoRoundModifier(),
    ]
    with pytest.raises(ValueError, match="incompatible pipelines"):
        CalibrationPipeline.from_modifiers(modifiers)
