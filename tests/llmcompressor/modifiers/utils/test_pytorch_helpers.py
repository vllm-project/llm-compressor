import pytest
import torch

from llmcompressor.core import active_session, reset_session
from llmcompressor.modifiers.utils.pytorch_helpers import (
    apply_loss_mask,
    get_current_loss_mask,
)


def test_apply_loss_mask_without_mask_flattens():
    activations = torch.randn(2, 5, 8)
    assert torch.equal(apply_loss_mask(activations, None), activations.reshape(10, 8))


def test_apply_loss_mask_matches_awq_formula():
    activations = torch.randn(3, 4, 6)
    mask = (torch.rand(3, 4) > 0.5).long()

    # the formula which used to be inlined in AWQModifier's activation hook
    flat_mask = mask.unsqueeze(-1).flatten(0, -2).squeeze(-1)
    expected = activations.flatten(0, -2)[flat_mask.bool()]

    assert torch.equal(apply_loss_mask(activations, mask), expected)


def test_apply_loss_mask_keeps_nonzero_tokens():
    activations = torch.randn(2, 5, 8)
    mask = torch.tensor([[0, 1, 1, 0, 1], [1, 0, 0, 0, 1]])

    kept = apply_loss_mask(activations, mask)

    assert torch.equal(kept, activations.reshape(10, 8)[[1, 2, 4, 5, 9]])


def test_apply_loss_mask_accepts_flat_activations():
    activations = torch.randn(6, 4)
    mask = torch.tensor([[1, 0, 1], [0, 0, 1]])

    assert torch.equal(apply_loss_mask(activations, mask), activations[[0, 2, 5]])


def test_apply_loss_mask_mismatch_raises():
    # e.g. a routed MoE expert which only sees a subset of the batch's tokens
    with pytest.raises(ValueError, match="cannot be aligned"):
        apply_loss_mask(torch.randn(7, 4), torch.ones(2, 5))


def test_get_current_loss_mask():
    reset_session()
    state = active_session().state
    masks = [torch.ones(1, 3), torch.zeros(1, 3)]
    try:
        assert get_current_loss_mask() is None  # masking disabled

        state.loss_masks = masks
        assert get_current_loss_mask() is None  # no batch active yet

        state.current_batch_idx = 1
        assert get_current_loss_mask() is masks[1]

        state.current_batch_idx = 2
        assert get_current_loss_mask() is None  # out of range
    finally:
        reset_session()
