import json

import pytest
from compressed_tensors.quantization import QuantizationConfig
from loguru import logger
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils.dev import skip_weights_download

SMOKE_MODEL = "nm-testing/tinysmokellama-3.2"
CT_MODEL = "nm-testing/SmolLM-1.7B-Instruct-quantized.w4a16"
NON_CT_MODEL = "inference-optimization/DeepSeek-R1-Distill-Llama-8B-NVFP4"  # modelopt


@pytest.mark.smoke
@pytest.mark.integration
def test_oneshot_allows_unquantized_smoke_model(tmp_path):
    with skip_weights_download():
        model = oneshot(model=SMOKE_MODEL, recipe=[], output_dir=str(tmp_path))

    assert model is not None


@pytest.mark.smoke
@pytest.mark.integration
def test_oneshot_warns_pre_quantized_smoke_model():
    logs = []
    handler_id = logger.add(logs.append, format="{message}", level="WARNING")

    # TODO: inspect bad interaction between ct decompress and `skip_weights_download`
    # for now, actually load the model weights in this test
    oneshot(model=CT_MODEL, recipe=[])

    assert any("already quantized" in log for log in logs)
    logger.remove(handler_id)


@pytest.mark.skip("Skip downloading large modelopt model")
@pytest.mark.smoke
@pytest.mark.integration
def test_oneshot_rejects_pre_quantized_smoke_model():
    with pytest.raises(ValueError, match="already quantized"):
        with skip_weights_download():
            oneshot(model=NON_CT_MODEL, recipe=[])


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


def test_oneshot_loads_custom_code_model_with_trust_remote_code(tmp_path):
    (tmp_path / "configuration_mycustom.py").write_text(
        "from transformers import LlamaConfig\n"
        "class MyCustomConfig(LlamaConfig):\n"
        "    model_type = 'mycustom'\n"
    )
    (tmp_path / "modeling_mycustom.py").write_text(
        "from transformers import LlamaForCausalLM\n"
        "from .configuration_mycustom import MyCustomConfig\n"
        "class MyCustomForCausalLM(LlamaForCausalLM):\n"
        "    config_class = MyCustomConfig\n"
    )
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    LlamaForCausalLM(config).save_pretrained(tmp_path)
    config_path = tmp_path / "config.json"
    config_dict = json.loads(config_path.read_text())
    config_dict.update(
        model_type="mycustom",
        architectures=["MyCustomForCausalLM"],
        auto_map={
            "AutoConfig": "configuration_mycustom.MyCustomConfig",
            "AutoModelForCausalLM": "modeling_mycustom.MyCustomForCausalLM",
        },
    )
    config_path.write_text(json.dumps(config_dict))

    model = oneshot(model=str(tmp_path), recipe=[], trust_remote_code_model=True)

    assert type(model).__name__ == "MyCustomForCausalLM"
