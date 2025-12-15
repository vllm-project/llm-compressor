import pytest

from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.pipelines import (
    CalibrationPipeline,
    SequentialPipeline,
)


@pytest.mark.parametrize(
    "modifiers",
    [
        (
            [
                QuantizationModifier(),
            ],
            SequentialPipeline,
        )
    ],
)
def test_infer_pipeline(modifiers, exp_pipeline):
    pipeline = CalibrationPipeline.from_modifiers(modifiers)
    assert isinstance(pipeline, exp_pipeline)
