# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from compressed_tensors.offload import OffloadCache, offload_module
from transformers import LlamaConfig, LlamaForCausalLM

from llmcompressor.transformers.compression import compressed_tensors_utils as saving


@pytest.fixture
def offloaded_model():
    # Construct locally: no model weights, tokenizer or dataset download.
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=32,
            tie_word_embeddings=False,
        )
    ).eval()
    for module in model.modules():
        offload_module(module, "cpu", "cpu")
    _assert_restored(model)
    return model


def _assert_restored(model):
    assert any(
        isinstance(module._parameters, OffloadCache) for module in model.modules()
    )
    assert not any(hasattr(module, "_hf_hook") for module in model.modules())


def _logits(model):
    with torch.no_grad():
        return model(torch.tensor([[1, 2, 3]])).logits.clone()


@pytest.mark.parametrize("stage", ["weights", "recipe", "python_files"])
def test_failed_save_restores_offload(offloaded_model, tmp_path, monkeypatch, stage):
    model = offloaded_model
    expected = _logits(model)
    failure = OSError("injected save failure")

    def fail(*args, **kwargs):
        # Ensure the failure occurs after the real offload conversion.
        assert any(hasattr(module, "_hf_hook") for module in model.modules())
        raise failure

    with monkeypatch.context() as patch:
        if stage == "weights":
            patch.setattr(LlamaForCausalLM, "save_pretrained", fail)
        elif stage == "recipe":
            patch.setattr(saving, "update_and_save_recipe", fail)
        else:
            patch.setattr(saving, "copy_python_files_from_model_cache", fail)
        saving.modify_save_pretrained(model)
        with pytest.raises(OSError) as caught:
            model.save_pretrained(tmp_path / "failed", save_compressed=False)
        assert caught.value is failure

    _assert_restored(model)
    torch.testing.assert_close(_logits(model), expected, rtol=0, atol=0)


def test_invalid_save_directory_restores_offload(offloaded_model, tmp_path):
    model = offloaded_model
    expected = _logits(model)
    target = tmp_path / "not-a-directory"
    target.write_text("sentinel")
    saving.modify_save_pretrained(model)
    with pytest.raises(OSError):
        model.save_pretrained(target, save_compressed=False)
    assert target.read_text() == "sentinel"
    _assert_restored(model)
    torch.testing.assert_close(_logits(model), expected, rtol=0, atol=0)

    # Retry through the same wrapped model, without manually restoring offload.
    model.save_pretrained(tmp_path / "retry", save_compressed=False)
    reloaded = LlamaForCausalLM.from_pretrained(tmp_path / "retry").eval()
    _assert_restored(model)
    torch.testing.assert_close(_logits(reloaded), expected, rtol=0, atol=0)


def test_successful_save_restores_offload(offloaded_model, tmp_path):
    model = offloaded_model
    expected = _logits(model)
    saving.modify_save_pretrained(model)
    model.save_pretrained(tmp_path / "saved", save_compressed=False)
    _assert_restored(model)
    reloaded = LlamaForCausalLM.from_pretrained(tmp_path / "saved").eval()
    torch.testing.assert_close(_logits(reloaded), expected, rtol=0, atol=0)
