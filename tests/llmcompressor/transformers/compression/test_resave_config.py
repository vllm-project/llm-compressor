import json

import pytest

from llmcompressor.transformers.compression import compressed_tensors_utils
from llmcompressor.transformers.compression.compressed_tensors_utils import (
    resave_config,
)


class FakeConfig:
    """
    Minimal stand-in for a ``PretrainedConfig``. ``resave_config`` only relies on
    ``_name_or_path`` (to locate the original config) and ``get_text_config``
    (to source patched field values), so we avoid constructing a real config.
    """

    def __init__(self, name_or_path="", text_config=None, **attrs):
        self._name_or_path = name_or_path
        self._text_config = text_config
        for key, value in attrs.items():
            setattr(self, key, value)

    def get_text_config(self):
        return self._text_config if self._text_config is not None else self


def _write_json(path, data):
    with open(path, "w") as file:
        json.dump(data, file)


def _read_json(path):
    with open(path) as file:
        return json.load(file)


@pytest.fixture
def save_dir(tmp_path):
    """A save directory pre-populated with a transformers-serialized config."""
    save_dir = tmp_path / "save"
    save_dir.mkdir()
    _write_json(save_dir / "config.json", {"serialized_by": "transformers"})
    return save_dir


@pytest.fixture
def original_dir(tmp_path):
    original_dir = tmp_path / "original"
    original_dir.mkdir()
    return original_dir


def test_missing_name_or_path_keeps_serialized_config(save_dir):
    # no _name_or_path -> function returns early and leaves save_dir untouched
    config = FakeConfig(name_or_path="")

    resave_config(config, str(save_dir))

    assert _read_json(save_dir / "config.json") == {"serialized_by": "transformers"}


def test_resave_original_config_without_experts(save_dir, original_dir):
    original = {"architectures": ["Foo"], "hidden_size": 4}
    _write_json(original_dir / "config.json", original)
    config = FakeConfig(name_or_path=str(original_dir))

    resave_config(config, str(save_dir))

    # original config overwrites the transformers-serialized one, unchanged
    assert _read_json(save_dir / "config.json") == original


def test_resave_config_is_sorted_and_indented(save_dir, original_dir):
    _write_json(original_dir / "config.json", {"b": 1, "a": 2})
    config = FakeConfig(name_or_path=str(original_dir))

    resave_config(config, str(save_dir))

    with open(save_dir / "config.json") as file:
        contents = file.read()
    assert contents == json.dumps({"a": 2, "b": 1}, indent=2, sort_keys=True)


def test_experts_field_patched_top_level(save_dir, original_dir):
    # original model had 8 experts; llmcompressor pruned it down to 4
    _write_json(original_dir / "config.json", {"num_experts": 8, "hidden_size": 4})
    config = FakeConfig(name_or_path=str(original_dir), num_experts=4)

    resave_config(config, str(save_dir))

    resaved = _read_json(save_dir / "config.json")
    assert resaved["num_experts"] == 4
    assert resaved["hidden_size"] == 4


def test_experts_field_patched_via_key_fallback(save_dir, original_dir):
    # original uses a different expert key than the config attribute
    _write_json(original_dir / "config.json", {"num_local_experts": 8})
    config = FakeConfig(name_or_path=str(original_dir), num_experts=4)

    resave_config(config, str(save_dir))

    assert _read_json(save_dir / "config.json")["num_local_experts"] == 4


def test_experts_field_patched_in_nested_text_config(save_dir, original_dir):
    _write_json(
        original_dir / "config.json",
        {"text_config": {"num_experts": 8}, "model_type": "multimodal"},
    )
    text_config = FakeConfig(num_experts=4)
    config = FakeConfig(name_or_path=str(original_dir), text_config=text_config)

    resave_config(config, str(save_dir))

    resaved = _read_json(save_dir / "config.json")
    assert resaved["text_config"]["num_experts"] == 4
    assert resaved["model_type"] == "multimodal"


def test_experts_present_but_missing_target_key_keeps_serialized(
    save_dir, original_dir
):
    # experts modified on the config but original has no expert key to patch
    _write_json(original_dir / "config.json", {"hidden_size": 4})
    config = FakeConfig(name_or_path=str(original_dir), num_experts=4)

    resave_config(config, str(save_dir))

    # returns early without overwriting the transformers-serialized config
    assert _read_json(save_dir / "config.json") == {"serialized_by": "transformers"}


def test_invalid_original_json_keeps_serialized_config(save_dir, original_dir):
    with open(original_dir / "config.json", "w") as file:
        file.write("{ not valid json")
    config = FakeConfig(name_or_path=str(original_dir))

    resave_config(config, str(save_dir))

    assert _read_json(save_dir / "config.json") == {"serialized_by": "transformers"}


def test_falls_back_to_hf_hub_download(save_dir, original_dir, tmp_path, monkeypatch):
    # name_or_path has no local config.json, so the hub download path is used
    downloaded = tmp_path / "hub" / "config.json"
    downloaded.parent.mkdir()
    _write_json(downloaded, {"downloaded": True, "num_experts": 8})

    def fake_download(repo_id, filename, **kwargs):
        assert filename == "config.json"
        return str(downloaded)

    monkeypatch.setattr(compressed_tensors_utils, "hf_hub_download", fake_download)
    config = FakeConfig(name_or_path="some/remote-repo", num_experts=4)

    resave_config(config, str(save_dir))

    resaved = _read_json(save_dir / "config.json")
    assert resaved["downloaded"] is True
    assert resaved["num_experts"] == 4


def test_hf_hub_download_failure_keeps_serialized_config(save_dir, monkeypatch):
    def fake_download(*args, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(compressed_tensors_utils, "hf_hub_download", fake_download)
    config = FakeConfig(name_or_path="some/remote-repo")

    resave_config(config, str(save_dir))

    assert _read_json(save_dir / "config.json") == {"serialized_by": "transformers"}
