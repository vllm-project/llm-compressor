from types import SimpleNamespace
from unittest.mock import Mock

from llmcompressor.entrypoints.utils import post_process


def test_post_process_saves_mtp_after_backbone(monkeypatch, tmp_path):
    """Oneshot finalizes MTP tensors after saving the compressed backbone."""
    calls = []
    model = Mock()
    model.save_pretrained.side_effect = lambda *args, **kwargs: calls.append("backbone")
    model_args = SimpleNamespace(
        model=model,
        processor=None,
        save_compressed=True,
        model_revision="revision",
    )
    save_mtp = Mock(side_effect=lambda *args, **kwargs: calls.append("mtp"))
    monkeypatch.setattr(
        "llmcompressor.entrypoints.utils.save_mtp_tensors",
        save_mtp,
    )

    post_process(
        model_args=model_args,
        output_dir=str(tmp_path),
        mtp_scheme="NVFP4A16",
    )

    assert calls == ["backbone", "mtp"]
    model.save_pretrained.assert_called_once_with(str(tmp_path), save_compressed=True)
    save_mtp.assert_called_once_with(
        model,
        str(tmp_path),
        "NVFP4A16",
        revision="revision",
    )
