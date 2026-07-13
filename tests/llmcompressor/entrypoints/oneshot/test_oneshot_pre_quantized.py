import pytest
from loguru import logger

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils.dev import skip_weights_initialize

SMOKE_MODEL = "nm-testing/tinysmokellama-3.2"
CT_MODEL = "nm-testing/SmolLM-1.7B-Instruct-quantized.w4a16"
QUANTIZED_MODEL = "google/gemma-4-E2B-it-qat-mobile-transformers"


def _recipe():
    return [
        QuantizationModifier(
            targets="Linear",
            scheme="FP8_DYNAMIC",
            ignore=["lm_head"],
        )
    ]


@pytest.mark.smoke
@pytest.mark.integration
def test_oneshot_allows_unquantized_smoke_model(tmp_path):
    model = oneshot(
        model=SMOKE_MODEL,
        recipe=_recipe(),
        output_dir=str(tmp_path),
    )

    assert model is not None


@pytest.mark.smoke
@pytest.mark.integration
def test_oneshot_warns_pre_quantized_smoke_model():
    logs = []
    handler_id = logger.add(logs.append, format="{message}", level="WARNING")

    with skip_weights_initialize():
        oneshot(
            model=CT_MODEL,
            recipe=_recipe(),
        )

    assert "already quantized" in logs[0]
    logger.remove(handler_id)


@pytest.mark.smoke
@pytest.mark.integration
def test_oneshot_rejects_pre_quantized_smoke_model():
    with skip_weights_initialize(), pytest.raises(
        ValueError, match="already quantized"
    ):
        oneshot(
            model=QUANTIZED_MODEL,
            recipe=_recipe(),
        )
