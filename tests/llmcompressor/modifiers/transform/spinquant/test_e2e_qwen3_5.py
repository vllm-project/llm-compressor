"""
End-to-end tests for SpinQuant on Qwen3.5 hybrid attention.

Qwen3.5 alternates full self-attention layers with linear (Gated DeltaNet)
layers. These tests exercise the dynamic mapping path end-to-end on a real
(but tiny) Qwen3.5 model, verifying that the R1 residual-stream rotation is a
logits-invariant transform.

Note: only R1 is valid for Qwen3.5. R2 rotates the value/output head space, but
Qwen3.5 full-attention applies an element-wise gate
(``attn_output * sigmoid(gate)``) between the attention output and ``o_proj``
inside that head space; the gate does not commute with R2. The gate's effect is
negligible on randomly-initialized weights (where the gate is ~constant), so R2
only breaks invariance on trained weights and is not covered here.
"""

import pytest
import torch

from llmcompressor.core import State
from llmcompressor.modeling.offset_norm import norm_calibration_context
from llmcompressor.modifiers.transform import SpinQuantModifier

try:
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    _HAS_QWEN3_5 = True
except ImportError:
    _HAS_QWEN3_5 = False

pytestmark = pytest.mark.skipif(
    not _HAS_QWEN3_5, reason="requires transformers with Qwen3.5 support"
)


def _make_tiny_qwen3_5(num_layers: int = 4, full_attention_interval: int = 4):
    """Build a tiny Qwen3.5 causal LM with hybrid full/linear attention."""
    layer_types = [
        (
            "full_attention"
            if i % full_attention_interval == full_attention_interval - 1
            else "linear_attention"
        )
        for i in range(num_layers)
    ]
    config = Qwen3_5TextConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_conv_kernel_dim=4,
        max_position_embeddings=256,
        layer_types=layer_types,
        tie_word_embeddings=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    return Qwen3_5ForCausalLM(config)


def test_r1_logits_invariance():
    """R1 rotation must be a logits-invariant transform on Qwen3.5."""
    torch.manual_seed(0)
    model = _make_tiny_qwen3_5().eval()
    state = State(model=model)
    modifier = SpinQuantModifier(
        rotations=["R1"],
        transform_block_size=64,
        transform_type="hadamard",
    )

    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.no_grad():
        ref_logits = model(input_ids=input_ids).logits.float()

    modifier.on_initialize(state)
    # Qwen3.5 uses offset norms (output * (1 + weight)); convert to standard norm
    # during calibration so fuse_norm_linears folds the correct scale. This mirrors
    # what the oneshot entrypoint does via norm_calibration_context.
    with norm_calibration_context(model):
        modifier.on_calibration_start(state, None)

    with torch.no_grad():
        out_logits = model(input_ids=input_ids).logits.float()

    mse = torch.nn.functional.mse_loss(out_logits, ref_logits).item()
    # The transform is exact up to float64 rounding; 1e-2 is a generous bound
    # (measured ~1e-4). Reference: llmcompressor's Llama SpinQuant test uses 8e-3.
    assert mse < 1e-2, f"R1 not logits-invariant for Qwen3.5, MSE={mse}"


def test_head_dim_reads_from_text_config():
    """on_initialize must read attention head_dim from text_config when present."""
    model = torch.nn.Module()
    # top-level config has no head_dim (hidden_size // num_attention_heads = 8);
    # text_config carries the real attention geometry (head_dim = 16).
    model.config = type(
        "Cfg",
        (),
        {
            "hidden_size": 64,
            "num_attention_heads": 8,
            "text_config": type("TCfg", (), {"head_dim": 16})(),
        },
    )()
    state = State(model=model)
    modifier = SpinQuantModifier(rotations=["R2"])
    modifier.on_initialize(state)

    r2_scheme = modifier.transform_config.config_groups["R2"]
    assert r2_scheme.head_dim == 16
