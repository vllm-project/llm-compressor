import sys

import pytest
from loguru import logger
from transformers import AutoModelForCausalLM, LlamaConfig

from llmcompressor import oneshot


@pytest.fixture
def make_model(tmp_path):
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    source = tmp_path / "source"
    config.save_pretrained(source)

    def _make_model():
        model = AutoModelForCausalLM.from_config(config)
        model.config.name_or_path = str(source)
        return model

    return _make_model


def _read_log(log_dir):
    (log_file,) = log_dir.glob("*.log")
    return log_file.read_text()


@pytest.mark.unit
def test_log_dir_only_contains_its_own_run(make_model, tmp_path):
    first_dir, second_dir = tmp_path / "first", tmp_path / "second"

    oneshot(model=make_model(), recipe=[], log_dir=str(first_dir))
    logger.warning("logged between runs")
    oneshot(model=make_model(), recipe=[], log_dir=str(second_dir))
    logger.warning("logged after runs")

    first_log, second_log = _read_log(first_dir), _read_log(second_dir)
    assert "logged between runs" not in first_log
    assert "logged after runs" not in first_log
    assert "logged after runs" not in second_log


@pytest.mark.unit
def test_log_file_env_handler_is_removed(make_model, tmp_path, monkeypatch):
    log_file = tmp_path / "oneshot.log"
    monkeypatch.setenv("LLM_COMPRESSOR_LOG_FILE", str(log_file))

    oneshot(model=make_model(), recipe=[])
    logger.warning("logged after run")

    assert log_file.exists()
    assert "logged after run" not in log_file.read_text()


@pytest.mark.unit
def test_log_dir_handler_is_removed_when_init_fails(tmp_path):
    log_dir = tmp_path / "logs"

    with pytest.raises(Exception):
        oneshot(model=str(tmp_path / "missing_model"), recipe=[], log_dir=str(log_dir))
    logger.warning("logged after failed run")

    assert "logged after failed run" not in _read_log(log_dir)


@pytest.mark.unit
def test_log_dir_handler_already_removed(make_model, tmp_path, monkeypatch):
    from llmcompressor.entrypoints.oneshot import Oneshot

    oneshot_cls_call = Oneshot.__call__

    def call_after_removing_handlers(self):
        logger.remove()  # e.g. the application resets all handlers mid-run
        return oneshot_cls_call(self)

    monkeypatch.setattr(Oneshot, "__call__", call_after_removing_handlers)

    oneshot(model=make_model(), recipe=[], log_dir=str(tmp_path / "logs"))


@pytest.mark.unit
def test_log_dir_handler_is_removed_when_call_fails(make_model, tmp_path, monkeypatch):
    # `llmcompressor.entrypoints.oneshot` is shadowed by the `oneshot` function
    oneshot_module = sys.modules["llmcompressor.entrypoints.oneshot"]

    def failing_dataloader(*args, **kwargs):
        raise RuntimeError("calibration failed")

    monkeypatch.setattr(
        oneshot_module, "get_calibration_dataloader", failing_dataloader
    )
    log_dir = tmp_path / "logs"

    with pytest.raises(RuntimeError, match="calibration failed"):
        oneshot(model=make_model(), recipe=[], log_dir=str(log_dir))
    logger.warning("logged after failed run")

    assert "logged after failed run" not in _read_log(log_dir)
