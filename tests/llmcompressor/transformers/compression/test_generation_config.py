# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Invalid generation metadata must fail before costly compression/save."""

import json
import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from transformers import GenerationConfig

from examples.glm52_precision import oneshot_profile as driver
from llmcompressor.transformers.compression import compressed_tensors_utils as saving


def greedy_config(top_p=0.95):
    config = GenerationConfig(do_sample=False)
    config.top_p = top_p
    return config


@pytest.mark.parametrize("sampling", [False, None])
def test_driver_sanitizes_inactive_top_p_and_records_original(tmp_path, sampling):
    config = greedy_config()
    config.do_sample = sampling
    with pytest.raises(ValueError, match="top_p"):
        config.validate(strict=True)
    report = {}
    driver.prepare_generation_config(config, report)
    assert config.do_sample is sampling
    assert config.top_p is None
    config.validate(strict=True)
    config.save_pretrained(tmp_path)  # The real Transformers save validation.
    assert report["generation_config"] == {
        "adjustments": [
            {
                "field": "top_p",
                "original": 0.95,
                "value": None,
                "action": "unset_non_sampling_top_p",
                "do_sample": sampling,
                "phase": "load_validation",
            }
        ],
        "load_validation": "passed",
    }


@pytest.mark.parametrize(
    "sampling,top_p",
    [
        (False, None),
        (False, 1.0),
        (None, None),
        (None, 1.0),
        (True, None),
        (True, 1.0),
        (True, 0.95),
    ],
)
def test_driver_leaves_valid_generation_config_unchanged(sampling, top_p):
    config = GenerationConfig(do_sample=sampling, top_p=top_p)
    before, report = config.to_dict(), {}
    driver.prepare_generation_config(config, report)
    assert config.to_dict() == before
    assert report["generation_config"]["adjustments"] == []
    assert report["generation_config"]["load_validation"] == "passed"


def test_driver_rejects_other_invalid_flags_before_model_load(tmp_path, monkeypatch):
    config = greedy_config()
    config.temperature = 0.5  # Outside the deliberately narrow sanitation.
    source = tmp_path / "generation_config.json"
    source.write_text(json.dumps(config.to_dict()))
    original = source.read_bytes()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "oneshot_profile.py",
            "--model",
            str(tmp_path),
            "--output",
            str(tmp_path / "output"),
            "--device",
            "cpu",
            "--dataset",
            str(tmp_path / "unused.json"),
        ],
    )
    monkeypatch.setattr(driver, "load_context", nullcontext)
    load_weights = Mock(side_effect=AssertionError("Weights must not load"))
    monkeypatch.setattr(driver.AutoModelForCausalLM, "from_pretrained", load_weights)
    tokenizer, oneshot = Mock(), Mock()
    monkeypatch.setattr(driver, "AutoTokenizer", tokenizer)
    monkeypatch.setattr(driver, "oneshot", oneshot)
    with pytest.raises(ValueError, match="temperature"):
        driver.main()
    oneshot.assert_not_called()
    tokenizer.from_pretrained.assert_not_called()
    load_weights.assert_not_called()
    assert source.read_bytes() == original
    assert config.do_sample is False and config.temperature == 0.5
    report = json.loads((tmp_path / "output-profile/rank-0.json").read_text())
    assert report["status"] == "failed"
    assert report["generation_config"]["pre_load_validation"] == "failed"
    assert report["generation_config"]["adjustments"][0]["original"] == 0.95


def test_preload_inputs_precede_weights_and_loaded_config_is_rechecked(
    tmp_path, monkeypatch
):
    from tests.flex_smooth.test_distributed_oneshot import _config, _fixture

    _config(tmp_path)
    model, tokenizer = _fixture(tmp_path)
    tokenizer.save_pretrained(tmp_path / "input")
    source = tmp_path / "input/generation_config.json"
    source.write_text(json.dumps({"do_sample": None, "top_p": 0.95}))
    original = source.read_bytes()
    # Model load returns its own, independently loaded metadata object.
    model.generation_config = GenerationConfig.from_pretrained(tmp_path / "input")
    dataset = tmp_path / "prompts.json"
    dataset.write_text(json.dumps(["w1", "w2 w3"]))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "profile",
            "--model",
            str(tmp_path / "input"),
            "--output",
            str(tmp_path / "out"),
            "--dataset",
            str(dataset),
            "--device",
            "cpu",
        ],
    )
    monkeypatch.setattr(driver, "load_context", nullcontext)
    events = []
    prepare, data = driver.prepare_generation_config, driver.calibration_data

    def record_prepare(config, report, *, phase="load_validation"):
        prepare(config, report, phase=phase)
        events.append(phase)

    def record_data(*args):
        result = data(*args)  # Includes real tokenizer-aware batch construction.
        events.append("batches_validated")
        return result

    def load_weights(*args, **kwargs):
        assert events == ["pre_load_validation", "batches_validated"]
        events.append("model_load")
        return model

    def stop_before_transforms(**kwargs):
        assert model.generation_config.do_sample is None
        assert model.generation_config.top_p is None
        assert len(kwargs["dataset"]) == 2
        assert kwargs["recipe"][1].max_tokens is None
        events.append("oneshot")
        raise RuntimeError("stop before transforms")

    monkeypatch.setattr(driver, "prepare_generation_config", record_prepare)
    monkeypatch.setattr(driver, "calibration_data", record_data)
    monkeypatch.setattr(driver.AutoModelForCausalLM, "from_pretrained", load_weights)
    monkeypatch.setattr(driver, "oneshot", stop_before_transforms)
    with pytest.raises(RuntimeError, match="stop before transforms"):
        driver.main()
    assert events == [
        "pre_load_validation",
        "batches_validated",
        "model_load",
        "load_validation",
        "oneshot",
    ]
    assert source.read_bytes() == original
    report = json.loads((tmp_path / "out-profile/rank-0.json").read_text())
    details = report["generation_config"]
    assert details["pre_load_validation"] == details["load_validation"] == "passed"
    assert [item["phase"] for item in details["adjustments"]] == [
        "pre_load_validation",
        "load_validation",
    ]
    assert all(
        item["original"] == 0.95 and item["do_sample"] is None
        for item in details["adjustments"]
    )
    assert report["tokens_global_valid"] == 3


class SavingModel:
    def __init__(self, config):
        self.generation_config = config
        self.saved = False

    def can_generate(self):
        return True

    def save_pretrained(self, *args, **kwargs):
        self.saved = True


def test_wrapper_rejects_invalid_config_without_mutation_or_compression(monkeypatch):
    model = SavingModel(greedy_config())
    before = model.generation_config.to_dict()
    factory = Mock()
    monkeypatch.setattr(saving.ModelCompressor, "from_pretrained_model", factory)
    saving.modify_save_pretrained(model)
    with pytest.raises(ValueError, match="top_p"):
        model.save_pretrained("unused", save_compressed=True)
    factory.assert_not_called()
    assert not model.saved
    assert model.generation_config.to_dict() == before


def test_wrapper_validates_strictly_before_compression(monkeypatch):
    model = SavingModel(greedy_config(None))
    events = []
    validate = model.generation_config.validate

    def validation(*, strict):
        assert strict is True
        validate(strict=strict)
        events.append("strict_validation")

    def compress(*args, **kwargs):
        events.append("compression")
        raise RuntimeError("stop after observing compression entry")

    monkeypatch.setattr(model.generation_config, "validate", validation)
    compressor = SimpleNamespace(compress_model=compress)
    monkeypatch.setattr(
        saving.ModelCompressor, "from_pretrained_model", lambda *a, **k: compressor
    )
    saving.modify_save_pretrained(model)
    with pytest.raises(RuntimeError, match="stop after"):
        model.save_pretrained("unused", save_compressed=True)
    assert events == ["strict_validation", "compression"]
    assert not model.saved
