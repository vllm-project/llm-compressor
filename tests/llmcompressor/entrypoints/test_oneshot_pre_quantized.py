import pytest

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

SMOKE_MODEL = "nm-testing/tinysmokellama-3.2"
QUANTIZED_MODEL = "nm-testing/SmolLM-1.7B-Instruct-quantized.w4a16"


def _recipe():
    return QuantizationModifier(
        targets="Linear",
        scheme="FP8_DYNAMIC",
        ignore=["lm_head"],
    )


@pytest.mark.smoke
@pytest.mark.integration
def test_oneshot_allows_unquantized_smoke_model(tmp_path):
    model = oneshot(
        model=SMOKE_MODEL,
        recipe=_recipe(),
        num_calibration_samples=1,
        max_seq_length=128,
        output_dir=str(tmp_path),
    )

    assert model is not None


@pytest.mark.smoke
@pytest.mark.integration
def test_oneshot_rejects_pre_quantized_smoke_model():
    with pytest.raises(ValueError, match="already quantized") as exc_info:
        oneshot(
            model=QUANTIZED_MODEL,
            recipe=_recipe(),
            num_calibration_samples=1,
            max_seq_length=128,
        )

    assert "full-precision checkpoint" in str(exc_info.value)
    assert "convert_checkpoint" in str(exc_info.value)
