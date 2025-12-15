import pytest

from llmcompressor.modifiers.quantization import QuantizationModifier, GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.pruning import SparseGPTQModifier, WandaPruningModifier
from llmcompressor.modifiers.transform import QuIPModifier, SpinQuantModifier
from llmcompressor.pipelines import CalibrationPipeline, SequentialPipeline, DataFreePipeline


@pytest.mark.parametrize("modifiers", [
    ([QuantizationModifier(scheme="FP8")], SequentialPipeline)
    ([QuantizationModifier(scheme="W4A16")], DataFreePipeline)
    ([GPTQModifier(scheme="FP8")], SequentialPipeline)
    ([GPTQModifier(scheme="W4A16")], DataFreePipeline)
    ([SmoothQuantModifier(), GPTQModifier(scheme="W4A16")], SequentialPipeline),
    ([AWQModifier(scheme="W4A16")], SequentialPipeline)
    ([AWQModifier(scheme="FP8")], SequentialPipeline)
    ([SparseGPTQModifier()], SequentialPipeline)
    ([WandaPruningModifier()], SequentialPipeline)
    ([QuIPModifier()], DataFreePipeline)
    ([SpinQuantModifier()], DataFreePipeline)
    ([QuIPModifier(), QuantizationModifier(scheme="FP8")], SequentialPipeline)
    ([QuIPModifier(), QuantizationModifier(scheme="W4A16")], DataFreePipeline)
])
def test_infer_pipeline(modifiers, exp_pipeline):
    pipeline = CalibrationPipeline.from_modifiers(modifiers)
    assert isinstance(pipeline, exp_pipeline)