import pytest
import torch
from compressed_tensors.quantization import apply_quantization_config
from torch.nn import Linear
from torch.testing import assert_close

from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq import (
    AWQ_MAPPING_REGISTRY,
    AWQMapping,
    AWQModifier,
    SlicedSmoothTarget,
)
from llmcompressor.modifiers.transform.awq.base import absorb_sliced_scales

HIDDEN = 8
NUM_CHUNKS = 6
REPEAT = 3
TEMB = 5
TOKENS = 7

SHIFT_CHUNK = 0
SCALE_CHUNK = 1
GATE_CHUNK = 2


class _AdaLNProj(torch.nn.Module):
    """Shared modulation projection, shaped like MiniMax-H3's ``adaln_proj``."""

    def __init__(self):
        super().__init__()
        self.linear = Linear(TEMB, NUM_CHUNKS * HIDDEN * REPEAT)

    def forward(self, temb):
        out = self.linear(torch.nn.functional.silu(temb))
        return out.view(-1, NUM_CHUNKS * HIDDEN).chunk(NUM_CHUNKS, dim=-1)


class _AdaLNBlock(torch.nn.Module):
    """
    Minimal stand-in for an AdaLN-modulated transformer block: the balance layer's
    input is ``norm1(x) * (1 + scale) + shift`` with both terms coming out of one
    shared projection, which is the case ``SlicedSmoothTarget`` exists for.
    """

    def __init__(self):
        super().__init__()
        self.norm1 = torch.nn.RMSNorm(HIDDEN)
        self.attn = torch.nn.ModuleDict({"to_q": Linear(HIDDEN, HIDDEN, bias=False)})
        self.adaln_proj = _AdaLNProj()

    def forward(self, hidden_states, temb, adaln_indices):
        chunks = self.adaln_proj(temb)
        shift, scale, gate = (
            chunks[SHIFT_CHUNK],
            chunks[SCALE_CHUNK],
            chunks[GATE_CHUNK],
        )
        normed = self.norm1(hidden_states) * (
            1.0 + scale.index_select(0, adaln_indices)
        ) + shift.index_select(0, adaln_indices)
        return hidden_states + gate.index_select(0, adaln_indices) * self.attn.to_q(
            normed
        )


def _make_block(dtype=torch.float64):
    torch.manual_seed(0)
    block = _AdaLNBlock().to(dtype)
    with torch.no_grad():
        # RMSNorm starts at 1.0; randomize so an incorrect fold cannot hide
        block.norm1.weight.normal_(1.0, 0.2)
        block.adaln_proj.linear.weight.normal_(0.0, 0.05)
        block.adaln_proj.linear.bias.normal_(0.0, 0.05)
    return block.eval()


def _make_inputs(dtype=torch.float64):
    torch.manual_seed(1234)
    return (
        torch.randn(TOKENS, HIDDEN, dtype=dtype),
        torch.randn(REPEAT, TEMB, dtype=dtype),
        torch.randint(0, REPEAT * REPEAT, (TOKENS,)),
    )


@pytest.mark.unit
def test_absorb_sliced_scales_preserves_output():
    """
    Folding 1/s into norm1 and into the shift rows of the shared projection, while
    scaling the balance layer's input columns by s, must leave the block's output
    unchanged. This is the equivalence that makes AWQ applicable to AdaLN blocks.
    """
    inputs = _make_inputs()
    reference = _make_block()
    with torch.no_grad():
        expected = reference(*inputs)

    block = _make_block()
    scales = torch.rand(HIDDEN, dtype=torch.float64) * 3.0 + 0.25

    with torch.no_grad():
        block.norm1.weight.div_(scales)
        absorb_sliced_scales(
            block.adaln_proj.linear,
            SlicedSmoothTarget(
                "adaln_proj.linear",
                chunk_index=SHIFT_CHUNK,
                num_chunks=NUM_CHUNKS,
                repeat=REPEAT,
            ),
            scales,
        )
        block.attn.to_q.weight.mul_(scales.view(1, -1))
        actual = block(*inputs)

    assert_close(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.unit
def test_absorbing_the_scale_chunk_is_not_equivalent():
    """
    Guards the choice of chunk. The multiplicative ``scale`` term is already
    accounted for by dividing norm1.weight, so folding 1/s into it as well changes
    the output. Without this the test above would pass for the wrong reason.
    """
    inputs = _make_inputs()
    reference = _make_block()
    with torch.no_grad():
        expected = reference(*inputs)

    block = _make_block()
    scales = torch.rand(HIDDEN, dtype=torch.float64) * 3.0 + 0.25

    with torch.no_grad():
        block.norm1.weight.div_(scales)
        absorb_sliced_scales(
            block.adaln_proj.linear,
            SlicedSmoothTarget(
                "adaln_proj.linear",
                chunk_index=SCALE_CHUNK,
                num_chunks=NUM_CHUNKS,
                repeat=REPEAT,
            ),
            scales,
        )
        block.attn.to_q.weight.mul_(scales.view(1, -1))
        actual = block(*inputs)

    assert not torch.allclose(actual, expected, rtol=1e-3, atol=1e-3)


@pytest.mark.unit
def test_absorb_sliced_scales_only_touches_its_own_chunk():
    block = _make_block()
    before = block.adaln_proj.linear.weight.clone()
    scales = torch.full((HIDDEN,), 2.0, dtype=torch.float64)

    absorb_sliced_scales(
        block.adaln_proj.linear,
        SlicedSmoothTarget(
            "adaln_proj.linear",
            chunk_index=SHIFT_CHUNK,
            num_chunks=NUM_CHUNKS,
            repeat=REPEAT,
        ),
        scales,
    )
    after = block.adaln_proj.linear.weight

    group = NUM_CHUNKS * HIDDEN
    changed = torch.zeros(before.size(0), dtype=torch.bool)
    for group_index in range(REPEAT):
        start = group_index * group + SHIFT_CHUNK * HIDDEN
        changed[start : start + HIDDEN] = True

    assert_close(after[changed], before[changed] / 2.0)
    assert_close(after[~changed], before[~changed])


@pytest.mark.unit
def test_absorb_sliced_scales_rejects_shape_mismatch():
    block = _make_block()
    scales = torch.ones(HIDDEN, dtype=torch.float64)
    with pytest.raises(ValueError, match="out_features"):
        absorb_sliced_scales(
            block.adaln_proj.linear,
            SlicedSmoothTarget(
                "adaln_proj.linear",
                chunk_index=SHIFT_CHUNK,
                num_chunks=NUM_CHUNKS,
                repeat=REPEAT + 1,
            ),
            scales,
        )


@pytest.mark.unit
def test_extra_smooth_targets_resolve_against_the_smooth_layer_parent():
    """
    The projection is a sibling of the norm, while the balance layers sit under
    ``attn``, so resolving against the balance layers' ancestor would not find it.
    """
    awq = AWQModifier(
        mappings=[
            AWQMapping(
                "re:.*norm1$",
                ["re:.*attn.to_q$"],
                extra_smooth_targets=[
                    SlicedSmoothTarget(
                        "adaln_proj.linear",
                        chunk_index=SHIFT_CHUNK,
                        num_chunks=NUM_CHUNKS,
                        repeat=REPEAT,
                    )
                ],
            )
        ],
    )
    model = torch.nn.ModuleDict(
        {"transformer_blocks": torch.nn.ModuleList([_AdaLNBlock() for _ in range(2)])}
    )
    apply_quantization_config(
        model,
        config=QuantizationModifier(scheme="W4A16_ASYM").resolve_quantization_config(),
    )

    awq._set_resolved_mappings(model)

    assert len(awq._resolved_mappings) == 2
    for index, mapping in enumerate(awq._resolved_mappings):
        assert len(mapping.extra_smooth_targets) == 1
        module, target = mapping.extra_smooth_targets[0]
        assert module is model.transformer_blocks[index].adaln_proj.linear
        assert target.chunk_index == SHIFT_CHUNK


@pytest.mark.unit
def test_minimax_h3_mappings_are_registered():
    mappings = AWQ_MAPPING_REGISTRY["MiniMaxH3Transformer3DModel"]
    sliced = {
        mapping.smooth_layer: mapping.extra_smooth_targets[0]
        for mapping in mappings
        if mapping.extra_smooth_targets
    }
    # shift_msa and shift_mlp of the shift/scale/gate x msa/mlp layout
    assert sliced["re:.*norm1$"].chunk_index == 0
    assert sliced["re:.*norm2$"].chunk_index == 3
    for target in sliced.values():
        assert (target.num_chunks, target.repeat) == (6, 3)
        assert target.layer == "adaln_proj.linear"
