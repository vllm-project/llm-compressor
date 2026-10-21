import pytest

from llmcompressor.modifiers.autoround import AutoRoundModifier
from llmcompressor.modifiers.pruning import SparseGPTModifier, WandaPruningModifier
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
