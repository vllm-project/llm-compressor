from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from compressed_tensors.offload import disable_offloading, offload_module
from compressed_tensors.offload.cache.cpu import CPUCache
from torch import nn
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

from llmcompressor.core import Event, State
from llmcompressor.modeling.offset_norm import norm_calibration_context
from llmcompressor.modifiers.transform.itersmooth import (
    IterSmoothMapping,
    IterSmoothModifier,
)

pytestmark = pytest.mark.unit


class OffsetNormLinears(nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.config = SimpleNamespace()
        self.norm = Qwen3_5RMSNorm(4).to(dtype)
        self.fc = nn.Linear(4, 3, bias=False).to(dtype)
        self.unobserved_norm = Qwen3_5RMSNorm(4).to(dtype)
        self.unobserved_fc = nn.Linear(4, 3, bias=False).to(dtype)

    def forward(self, inputs):
        return self.fc(self.norm(inputs))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("storage", ["resident", "offloaded", "cached"])
@torch.no_grad()
def test_itersmooth_persists_restored_offset_norm(
    dtype, storage, monkeypatch, tmp_path
):
    model = OffsetNormLinears(dtype).eval()
    model.norm.weight.copy_(torch.tensor([-0.25, 0.0, 0.25, 0.5], dtype=dtype))
    # These small offsets expose round-trip rounding if an unobserved norm is
    # incorrectly included in IterSmooth's writeback.
    model.unobserved_norm.weight.copy_(
        torch.tensor([0.001, 0.01, -0.01, 0.05], dtype=dtype)
    )
    model.fc.weight.copy_(torch.arange(12, dtype=dtype).reshape(3, 4) / 16)
    inputs = torch.tensor([[1.0, -2.0, 3.0, -4.0]], dtype=dtype)
    expected_output = model(inputs)
    original_norm = model.norm
    original_weight = model.norm.weight.clone()
    unobserved_weight = model.unobserved_norm.weight.clone()

    if storage != "resident":
        # Real CPUCache, with copying on access to emulate CPU -> NPU/CUDA
        # transfers using only CPU tensors.
        monkeypatch.setattr(
            CPUCache,
            "onload",
            lambda self, value: None if value is None else value.clone(),
        )
        for module in (model.norm, model.unobserved_norm):
            offload_module(module, onload_device="cpu", offload_device="cpu")

    modifier = IterSmoothModifier(
        mappings=[
            IterSmoothMapping(source="norm", targets=["fc"]),
            IterSmoothMapping(source="unobserved_norm", targets=["unobserved_fc"]),
        ]
    )
    state = State(model=model)
    state.data.calib = [object()]
    cache_context = disable_offloading() if storage == "cached" else nullcontext()
    with cache_context:
        with norm_calibration_context(model):
            modifier.initialize(state)
            modifier.on_calibration_start(state, Event())
            model(inputs)
            modifier.on_sequential_epoch_end(state, Event(), list(model.modules()))
            modifier.on_calibration_end(state, Event())
            smoothed_weight = model.norm.weight.detach().clone()
        # Match the real Oneshot lifecycle: public norm restoration first,
        # IterSmooth finalization second.
        modifier.finalize(state)

    assert model.norm is original_norm
    expected_weight = (smoothed_weight.float() - 1).to(dtype)
    assert not torch.equal(model.norm.weight, original_weight)
    torch.testing.assert_close(model.norm.weight, expected_weight)
    if storage != "resident":
        torch.testing.assert_close(
            original_norm._parameters.offloaded_values["weight"], expected_weight
        )
        # Public restore can round resident norms. IterSmooth must not persist
        # any additional round-trip changes to an unobserved offloaded norm.
        assert torch.equal(model.unobserved_norm.weight, unobserved_weight)
    tolerance = 2e-2 if dtype == torch.bfloat16 else 1e-5
    torch.testing.assert_close(
        model(inputs), expected_output, atol=tolerance, rtol=tolerance
    )
    assert not modifier._resolved
    assert all(not module._forward_pre_hooks for module in model.modules())

    checkpoint = tmp_path / "smoothed.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = OffsetNormLinears(dtype).eval()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
    torch.testing.assert_close(restored.norm.weight, expected_weight)
    torch.testing.assert_close(
        restored(inputs), expected_output, atol=tolerance, rtol=tolerance
    )


@torch.no_grad()
def test_qwen_oneshot_offset_norm_save_reload(monkeypatch, tmp_path):
    from safetensors import safe_open
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from torch.utils.data import DataLoader
    from transformers import (
        PreTrainedTokenizerFast,
        Qwen3_5ForCausalLM,
        Qwen3_5TextConfig,
    )

    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    torch.manual_seed(9)
    config = Qwen3_5TextConfig(
        vocab_size=32,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        partial_rotary_factor=1.0,
        layer_types=["full_attention"],
        max_position_embeddings=64,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0,
            "mrope_section": [2, 3, 3],
        },
    )
    model = Qwen3_5ForCausalLM(config).eval()
    model.config._attn_implementation = "eager"
    original_path = tmp_path / "original"
    model.save_pretrained(original_path)
    model.config._name_or_path = str(original_path)
    processor = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
        unk_token="[UNK]",
    )
    sample = {
        "input_ids": torch.randint(2, 32, (1, 8)),
        "attention_mask": torch.ones(1, 8, dtype=torch.long),
    }
    expected = model(**sample, use_cache=False).logits.clone()
    monkeypatch.setattr(
        CPUCache,
        "onload",
        lambda self, value: None if value is None else value.clone(),
    )
    monkeypatch.setattr(
        "llmcompressor.pipelines.data_free.pipeline.get_main_device",
        lambda: torch.device("cpu"),
    )
    for module in model.modules():
        if isinstance(module, Qwen3_5RMSNorm):
            offload_module(module, onload_device="cpu", offload_device="cpu")

    smoothed_path = tmp_path / "smoothed"
    oneshot(
        model=model,
        processor=processor,
        dataset=DataLoader([sample], batch_size=None),
        recipe=IterSmoothModifier(enable_subgraph_type=["norm-linear"]),
        pipeline="basic",
        output_dir=str(smoothed_path),
    )
    norms = {
        name: module.weight.detach().clone()
        for name, module in model.named_modules()
        if isinstance(module, Qwen3_5RMSNorm)
    }
    assert any(bool(weight.any()) for weight in norms.values())
    torch.testing.assert_close(
        model(**sample, use_cache=False).logits, expected, atol=1e-5, rtol=1e-4
    )
    restored = Qwen3_5ForCausalLM.from_pretrained(
        smoothed_path, attn_implementation="eager"
    ).eval()
    torch.testing.assert_close(
        restored(**sample, use_cache=False).logits, expected, atol=1e-5, rtol=1e-4
    )

    quantized_path = tmp_path / "quantized"
    oneshot(
        model=model,
        processor=processor,
        recipe=QuantizationModifier(
            scheme="W8A8", targets="Linear", ignore=["lm_head"]
        ),
        pipeline="datafree",
        output_dir=str(quantized_path),
    )
    with safe_open(
        quantized_path / "model.safetensors", framework="pt", device="cpu"
    ) as tensors:
        for name, weight in norms.items():
            torch.testing.assert_close(model.get_submodule(name).weight, weight)
            torch.testing.assert_close(tensors.get_tensor(f"{name}.weight"), weight)
