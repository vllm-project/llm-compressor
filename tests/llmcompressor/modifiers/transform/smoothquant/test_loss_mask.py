from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from datasets import Dataset
from transformers import AutoModelForCausalLM

from llmcompressor import oneshot
from llmcompressor.core import create_session, reset_session
from llmcompressor.modifiers.transform.smoothquant.base import (
    SmoothQuantMapping,
    SmoothQuantModifier,
)
from tests.testing_utils import torchrun


def _modifier(hidden=8):
    norm = torch.nn.LayerNorm(hidden)
    linear = torch.nn.Linear(hidden, 4)
    modifier = SmoothQuantModifier()
    modifier.resolved_mappings_ = [SmoothQuantMapping("norm", norm, [linear])]
    modifier.scales_ = {}
    modifier._setup_scale_hooks()
    return modifier, norm


@torch.no_grad()
def _calibrate(batches, masks):
    modifier, norm = _modifier()
    with create_session() as session:
        session.state.loss_masks = masks
        for batch_idx, batch in enumerate(batches):
            session.state.current_batch_idx = batch_idx
            norm(batch)
    modifier.remove_hooks()
    return modifier, norm


@torch.no_grad()
def test_mask_excludes_tokens():
    batches = [torch.randn(2, 6, 8), torch.randn(2, 6, 8)]
    masks = [torch.ones(2, 6, dtype=torch.long), torch.ones(2, 6, dtype=torch.long)]
    masks[0][:, :2] = 0
    masks[1][0] = 0
    batches[0][:, :2] *= 100  # outliers which only masked tokens carry

    modifier, norm = _calibrate(batches, masks)

    kept = torch.cat([norm(b)[m.bool()] for b, m in zip(batches, masks)])
    scale = modifier.scales_["norm"]
    torch.testing.assert_close(scale.min_channel_vals, kept.min(dim=0)[0])
    torch.testing.assert_close(scale.max_channel_vals, kept.max(dim=0)[0])


@torch.no_grad()
def test_all_ones_mask_matches_unmasked():
    batches = [torch.randn(2, 6, 8), torch.randn(2, 6, 8)]
    ones = [torch.ones(2, 6, dtype=torch.long)] * 2

    masked, _ = _calibrate(batches, ones)
    unmasked, _ = _calibrate(batches, None)

    for name in ("min_channel_vals", "max_channel_vals"):
        assert torch.equal(
            getattr(masked.scales_["norm"], name),
            getattr(unmasked.scales_["norm"], name),
        )


@torch.no_grad()
def test_fully_masked_batch_is_skipped():
    batches = [torch.randn(2, 6, 8), torch.randn(2, 6, 8)]
    masks = [torch.zeros(2, 6, dtype=torch.long), torch.ones(2, 6, dtype=torch.long)]

    modifier, norm = _calibrate(batches[:1], masks[:1])
    assert "norm" not in modifier.scales_

    modifier, norm = _calibrate(batches, masks)
    out = norm(batches[1]).flatten(0, 1)
    torch.testing.assert_close(modifier.scales_["norm"].max_channel_vals, out.max(0)[0])


@torchrun(world_size=2)
def test_reduce_with_a_fully_masked_rank():
    """
    With DDP, rank 1's tokens are all masked, so it holds no statistics. Both ranks
    must still issue the same all-reduces and end with rank 0's statistics.
    """
    dist.init_process_group(
        backend="gloo", init_method="env://", timeout=timedelta(seconds=30)
    )
    try:
        rank = dist.get_rank()
        torch.manual_seed(0)
        batch = torch.randn(1, 4, 8)
        modifier, norm = _calibrate([batch], [torch.full((1, 4), 1 - rank)])
        assert ("norm" in modifier.scales_) == (rank == 0)

        modifier._reduce_activation_scales()

        expected = norm(batch).flatten(0, 1)
        scale = modifier.scales_["norm"]
        torch.testing.assert_close(scale.min_channel_vals, expected.min(0)[0])
        torch.testing.assert_close(scale.max_channel_vals, expected.max(0)[0])
    finally:
        dist.destroy_process_group()


@torchrun(world_size=2)
def test_reduce_drops_layers_no_rank_saw():
    dist.init_process_group(
        backend="gloo", init_method="env://", timeout=timedelta(seconds=30)
    )
    try:
        modifier, _ = _calibrate([torch.randn(1, 4, 8)], [torch.zeros(1, 4)])
        modifier._reduce_activation_scales()
        assert modifier.scales_ == {}
    finally:
        dist.destroy_process_group()


def _calibration_dataset(num_samples, seq_len, num_masked):
    generator = torch.Generator().manual_seed(0)
    input_ids = torch.randint(0, 1000, (num_samples, seq_len), generator=generator)
    loss_mask = torch.ones(num_samples, seq_len, dtype=torch.long)
    loss_mask[:, :num_masked] = 0
    return Dataset.from_dict(
        {
            "input_ids": input_ids.tolist(),
            "attention_mask": torch.ones_like(input_ids).tolist(),
            "loss_mask": loss_mask.tolist(),
        }
    )


def _smoothed_norms(num_masked, use_loss_mask):
    reset_session()
    model = AutoModelForCausalLM.from_pretrained("nm-testing/tinysmokellama-3.2")
    oneshot(
        model=model,
        dataset=_calibration_dataset(4, 32, num_masked),
        recipe=SmoothQuantModifier(smoothing_strength=0.8),
        num_calibration_samples=4,
        max_seq_length=32,
        use_loss_mask=use_loss_mask,
    )
    weights = {
        name: module.weight.detach().clone()
        for name, module in model.named_modules()
        if name.endswith("layernorm")
    }
    reset_session()
    return weights


@pytest.mark.integration
def test_oneshot_loss_mask():
    unmasked = _smoothed_norms(num_masked=0, use_loss_mask=False)
    all_ones = _smoothed_norms(num_masked=0, use_loss_mask=True)
    prefix_masked = _smoothed_norms(num_masked=8, use_loss_mask=True)

    for name, weight in unmasked.items():
        assert torch.equal(weight, all_ones[name]), name
    assert any(
        not torch.equal(weight, prefix_masked[name])
        for name, weight in unmasked.items()
    )
