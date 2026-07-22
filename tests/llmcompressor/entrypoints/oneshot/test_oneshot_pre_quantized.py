import pytest
from compressed_tensors.quantization import QuantizationConfig
from loguru import logger
from transformers import AutoConfig, AutoModelForCausalLM

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


@pytest.mark.smoke
@pytest.mark.integration
def test_oneshot_stacks(tmp_path):
    outdir1 = tmp_path / "out1"
    outdir2 = tmp_path / "out2"

    attn_target = "re:.*self_attn.(q|k|v|o)_proj*"
    mlp_target = "re:.*mlp.(gate|up|down)_proj*"

    def oneshot1():
        model = AutoModelForCausalLM.from_pretrained("nm-testing/tinysmokeqwen3")
        oneshot(
            model=model,
            recipe=QuantizationModifier(
                targets=[attn_target],
                ignore=["lm_head", "model.layers.1.mlp.down_proj"],
                scheme="FP8_DYNAMIC",
            ),
        )
        model.save_pretrained(outdir1)

    def oneshot2():
        model = AutoModelForCausalLM.from_pretrained(outdir1)
        oneshot(
            model=model,
            recipe=QuantizationModifier(
                targets=[mlp_target],
                ignore=["lm_head"],
                scheme="W8A8",
            ),
        )
        model.save_pretrained(outdir2, save_original_format=False)

    oneshot1()
    oneshot2()

    config = AutoConfig.from_pretrained(outdir2)
    quant_config = QuantizationConfig.model_validate(config.quantization_config)

    ordered_names = list(quant_config.config_groups.keys())
    ordered_schemes = list(quant_config.config_groups.values())

    assert ordered_names == ["group_0", "group_1"]
    assert set(quant_config.ignore) == set(["lm_head"])
    assert ordered_schemes[0].targets[0] == attn_target
    assert ordered_schemes[1].targets[0] == mlp_target

    assert quant_config.format == "mixed-precision"
    assert ordered_schemes[0].format == "float-quantized"
    assert ordered_schemes[1].format == "int-quantized"
