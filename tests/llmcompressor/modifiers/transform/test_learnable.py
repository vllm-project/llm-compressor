"""Tests for SpinQuantModifier learned rotations (learnable=True).

Covers:
  * the full apply -> learn -> fuse lifecycle on a tiny in-memory model (CPU)
  * error handling when no calibration data is provided
  * that ``randomize`` remains unsupported
  * an end-to-end ``oneshot`` run + compressed save on GPU

The learned-rotation path differs from the fixed (QuaRot-style) path: each rotation
is applied, trained over the calibration set, and fused back into the weights one at
a time, because the compressed-tensors factories cannot compose multiple
``requires_grad`` transforms on the same Linear.
"""

import pytest
import torch
from torch.utils.data import DataLoader
from transformers import LlamaConfig, LlamaForCausalLM

from llmcompressor.core import State
from llmcompressor.modifiers.transform import SpinQuantModifier
from tests.testing_utils import requires_gpu

torch.manual_seed(0)

# hidden_size is the default R1/R4 block size, head_dim (hidden/heads) the R2 block
# size; both must be powers of two for the hadamard transform
_TINY_CONFIG = LlamaConfig(
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    vocab_size=100,
    max_position_embeddings=64,
)

_N_SAMPLES = 8
_SEQ_LEN = 16


def _make_model() -> LlamaForCausalLM:
    return LlamaForCausalLM(_TINY_CONFIG)


def _make_dataloader(batch_size: int = 4) -> DataLoader:
    """A dict-yielding DataLoader mimicking llmcompressor's calibration batches."""

    class _DictDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.input_ids = torch.randint(
                0, _TINY_CONFIG.vocab_size, (_N_SAMPLES, _SEQ_LEN)
            )
            self.labels = self.input_ids.clone()
            self.labels[:, :4] = -100  # mask the prompt, like TextGenerationDataset
            self.attention_mask = torch.ones(_N_SAMPLES, _SEQ_LEN, dtype=torch.long)

        def __len__(self):
            return _N_SAMPLES

        def __getitem__(self, index):
            return {
                "input_ids": self.input_ids[index],
                "labels": self.labels[index],
                "attention_mask": self.attention_mask[index],
            }

    return DataLoader(_DictDataset(), batch_size=batch_size)


def _set_calibration_data(state: State) -> State:
    state.data.calib = _make_dataloader()
    return state


def _get_linear_weight(model: torch.nn.Module, key: str) -> torch.Tensor:
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and key in name:
            return module.weight.detach().clone()
    raise AssertionError(f"no Linear matching {key}")


def test_randomize_raises():
    with pytest.raises(NotImplementedError):
        SpinQuantModifier(randomize=True)


def test_learnable_requires_calibration_data():
    model = _make_model()
    modifier = SpinQuantModifier(rotations=["R1"], learnable=True)
    modifier.on_initialize(State(model=model))

    state = State(model=model)  # state.data.calib is None
    with pytest.raises(ValueError, match="calibration data"):
        modifier.on_calibration_start(state, None)


def test_learnable_rotations_train_and_fuse():
    """Full apply -> learn -> fuse cycle; every Linear returns to plain nn.Linear."""
    model = _make_model()
    modifier = SpinQuantModifier(
        rotations=["R1", "R2", "R4"],
        learnable=True,
        learn_steps=6,
        learn_lr=1e-2,
    )

    before = {
        key: _get_linear_weight(model, key) for key in ("q_proj", "v_proj", "down_proj")
    }
    modifier.on_initialize(_set_calibration_data(State(model=model)))
    modifier.on_calibration_start(_set_calibration_data(State(model=model)), None)

    # offline rotations fused: no parametrizations / parametrized linears remain
    parametrized = [
        name
        for name, module in model.named_modules()
        if hasattr(module, "parametrizations") and len(module.parametrizations) > 0
    ]
    assert parametrized == [], f"parametrizations left after finalize: {parametrized}"

    non_plain = [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and type(module) is not torch.nn.Linear
    ]
    assert non_plain == [], f"non-plain Linears left: {non_plain}"

    # rotations were actually learned (weights changed from the base model)
    for key in before:
        assert not torch.allclose(before[key], _get_linear_weight(model, key)), (
            f"{key} weight unchanged after learned rotation"
        )

    # serialized transform config must not request training at runtime
    for name, scheme in modifier.transform_config.config_groups.items():
        assert scheme.requires_grad is False, f"{name} still requires_grad"

    # the model still computes
    out = model(
        input_ids=torch.randint(0, _TINY_CONFIG.vocab_size, (1, _SEQ_LEN)),
        attention_mask=torch.ones(1, _SEQ_LEN, dtype=torch.long),
    )
    assert out.logits.shape == (1, _SEQ_LEN, _TINY_CONFIG.vocab_size)


def test_learnable_serialization_roundtrip():
    modifier = SpinQuantModifier(
        rotations=["R1", "R2"],
        learnable=True,
        learn_lr=1e-4,
        learn_steps=7,
        learn_optimizer="sgd",
    )
    dumped = modifier.model_dump()
    restored = SpinQuantModifier.model_validate(dumped)
    assert restored == modifier
    assert restored.learnable and restored.learn_steps == 7


@requires_gpu
def test_learnable_end_to_end_oneshot(tmp_path):
    """End-to-end oneshot with learned rotations + quantization + compressed save."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    model_id = "nm-testing/tinysmokellama-3.2"
    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    oneshot(
        model=model,
        tokenizer=tokenizer,
        dataset=_make_dataloader(),
        recipe=[
            SpinQuantModifier(
                rotations=["R1", "R2"],
                learnable=True,
                learn_steps=4,
                learn_lr=1e-3,
            ),
            QuantizationModifier(
                targets="Linear", scheme="W4A16", ignore=["lm_head"]
            ),
        ],
    )

    # transforms fused, no parametrizations left after the full pipeline
    parametrized = [
        name
        for name, module in model.named_modules()
        if hasattr(module, "parametrizations") and len(module.parametrizations) > 0
    ]
    assert parametrized == []

    model.save_pretrained(tmp_path, save_compressed=True)

    import json

    with open(tmp_path / "config.json") as f:
        saved_config = json.load(f)
    assert "transform_config" in saved_config
    # no offline rotation should request gradients at runtime
    for group in saved_config["transform_config"]["config_groups"].values():
        assert group["requires_grad"] is False
