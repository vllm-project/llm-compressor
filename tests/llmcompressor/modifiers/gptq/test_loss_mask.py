import pytest
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM

from llmcompressor import oneshot
from llmcompressor.core import active_session, reset_session
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.gptq.helpers import accumulate_hessian, make_empty_hessian


def _accumulate(inp, module, mask=None):
    if isinstance(module, torch.nn.Linear):
        hessian = make_empty_hessian(module)
    else:
        hessian = torch.zeros(inp.shape[-1], inp.shape[-1])
    num_samples = torch.zeros(())
    return accumulate_hessian(inp, module, hessian, num_samples, mask=mask)


@torch.no_grad()
def test_all_ones_mask_matches_unmasked():
    module = torch.nn.Linear(16, 8)
    inp = torch.randn(3, 10, 16)

    hessian, num_samples = _accumulate(inp, module)
    masked_hessian, masked_num_samples = _accumulate(
        inp, module, mask=torch.ones(3, 10, dtype=torch.long)
    )

    assert torch.equal(hessian, masked_hessian)
    assert num_samples == masked_num_samples == 3


@torch.no_grad()
def test_mask_excludes_tokens():
    module = torch.nn.Linear(16, 8)
    inp = torch.randn(2, 10, 16)
    mask = torch.ones(2, 10, dtype=torch.long)
    mask[:, :4] = 0
    mask[1, 7] = 0

    hessian, num_samples = _accumulate(inp, module, mask=mask)

    kept = inp.reshape(-1, 16)[mask.reshape(-1).bool()]
    expected, _ = _accumulate(kept, module)
    torch.testing.assert_close(hessian, expected)
    # the sample counter still counts sequences, not unmasked tokens
    assert num_samples == 2


@torch.no_grad()
def test_conv1d_mask():
    from transformers import Conv1D

    module = Conv1D(8, 16)
    inp = torch.randn(2, 5, 16)
    mask = torch.tensor([[0, 1, 1, 1, 1], [0, 0, 1, 1, 1]])

    hessian, _ = _accumulate(inp, module, mask=mask)

    expected, _ = _accumulate(inp[mask.bool()], module)
    torch.testing.assert_close(hessian, expected)


def test_mask_shape_mismatch_raises():
    module = torch.nn.Linear(16, 8)
    # e.g. a routed MoE expert which only sees a subset of the batch's tokens
    inp = torch.randn(7, 16)
    with pytest.raises(ValueError, match="cannot be aligned"):
        _accumulate(inp, module, mask=torch.ones(2, 10))


def test_conv2d_mask_raises():
    module = torch.nn.Conv2d(3, 4, kernel_size=3)
    inp = torch.randn(2, 3, 8, 8)
    with pytest.raises(ValueError, match="Conv2d"):
        _accumulate(inp, module, mask=torch.ones(2, 3, 8))


@torch.no_grad()
def test_calibrate_module_reads_mask_from_state():
    reset_session()
    module = torch.nn.Linear(16, 8)
    modifier = GPTQModifier()
    batches = [torch.randn(2, 6, 16), torch.randn(2, 6, 16)]
    masks = [torch.ones(2, 6, dtype=torch.long), torch.ones(2, 6, dtype=torch.long)]
    masks[0][:, :2] = 0
    masks[1][0] = 0

    state = active_session().state
    state.loss_masks = masks
    try:
        for batch_idx, batch in enumerate(batches):
            state.current_batch_idx = batch_idx
            modifier.calibrate_module(module, (batch,), None)
    finally:
        reset_session()

    kept = torch.cat([b[m.bool()] for b, m in zip(batches, masks)])
    expected, _ = _accumulate(kept, module)
    torch.testing.assert_close(modifier._hessians[module], expected)
    assert modifier._num_samples[module] == 4


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


def _quantized_weights(num_masked, use_loss_mask):
    reset_session()
    model = AutoModelForCausalLM.from_pretrained("nm-testing/tinysmokellama-3.2")
    oneshot(
        model=model,
        dataset=_calibration_dataset(4, 32, num_masked),
        recipe=GPTQModifier(
            config_groups={
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 4,
                        "type": "int",
                        "symmetric": True,
                        "strategy": "group",
                        "group_size": 32,
                    },
                }
            },
            ignore=["lm_head"],
            block_size=16,
        ),
        num_calibration_samples=4,
        max_seq_length=32,
        use_loss_mask=use_loss_mask,
    )
    weights = {
        name: module.weight.detach().clone()
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and name != "lm_head"
    }
    reset_session()
    return weights


@pytest.mark.integration
def test_oneshot_loss_mask():
    unmasked = _quantized_weights(num_masked=0, use_loss_mask=False)
    all_ones = _quantized_weights(num_masked=0, use_loss_mask=True)
    prefix_masked = _quantized_weights(num_masked=8, use_loss_mask=True)

    for name, weight in unmasked.items():
        assert torch.equal(weight, all_ones[name]), name
    assert any(
        not torch.equal(weight, prefix_masked[name])
        for name, weight in unmasked.items()
    )
