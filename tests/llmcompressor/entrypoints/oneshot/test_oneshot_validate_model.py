import json

import pytest
from loguru import logger
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, LlamaConfig

from llmcompressor import oneshot


def _make_config() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
    )


def _make_model(source=None, quantization_config=None):
    model = AutoModelForCausalLM.from_config(_make_config())
    model.config.name_or_path = str(source) if source is not None else ""
    if quantization_config is not None:
        model.config.quantization_config = quantization_config
    return model


def _save_source(path, quantization_config=None):
    _make_config().save_pretrained(path)
    if quantization_config is not None:
        config_path = path / "config.json"
        config = json.loads(config_path.read_text())
        config["quantization_config"] = quantization_config
        config_path.write_text(json.dumps(config))
    return path


@pytest.fixture
def warnings_logged():
    logs = []
    handler_id = logger.add(logs.append, format="{message}", level="WARNING")
    yield logs
    logger.remove(handler_id)


@pytest.mark.unit
def test_oneshot_model_created_in_memory():
    model = _make_model()

    assert oneshot(model=model, recipe=[]) is model


@pytest.mark.unit
def test_oneshot_missing_source_config(tmp_path):
    model = _make_model(source=tmp_path / "moved_or_deleted")

    assert oneshot(model=model, recipe=[]) is model


@pytest.mark.unit
@pytest.mark.parametrize(
    "quantization_config",
    [
        {"quant_method": "bitsandbytes"},
        BitsAndBytesConfig(load_in_4bit=True),
    ],
)
def test_oneshot_rejects_in_memory_quantization(tmp_path, quantization_config):
    source = _save_source(tmp_path)
    model = _make_model(source=source, quantization_config=quantization_config)

    with pytest.raises(ValueError, match=r"different format \(bitsandbytes\)"):
        oneshot(model=model, recipe=[])


@pytest.mark.unit
def test_oneshot_rejects_source_quantization(tmp_path):
    source = _save_source(tmp_path, quantization_config={"quant_method": "fp8"})
    model = _make_model(source=source)

    with pytest.raises(ValueError, match=r"different format \(fp8\)"):
        oneshot(model=model, recipe=[])


@pytest.mark.unit
@pytest.mark.parametrize("in_memory", [True, False])
def test_oneshot_warns_compressed_tensors_quantization(
    tmp_path, warnings_logged, in_memory
):
    quantization_config = {"quant_method": "compressed-tensors"}
    if in_memory:
        model = _make_model(quantization_config=quantization_config)
    else:
        # decompressed models only retain the quantization config on disk
        source = _save_source(tmp_path, quantization_config=quantization_config)
        model = _make_model(source=source)

    oneshot(model=model, recipe=[])

    assert any("already quantized" in log for log in warnings_logged)
