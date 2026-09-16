from unittest.mock import patch

import pytest
import torch

from llmcompressor.modifiers.qad import QADModifier
from llmcompressor.modifiers.qad.base import _masked_mse
from llmcompressor.utils.dev import get_main_device


def test_fp16_long_sequence_gradients_match_fp32():
    device = get_main_device()
    shape = (1, 2048, 4096)
    initial = torch.tensor(0.001, dtype=torch.float16).item()
    results = {}
    for dtype, enabled in [
        (torch.float16, False),
        (torch.float16, True),
        (torch.float32, False),
    ]:
        parameter = torch.nn.Parameter(
            torch.tensor(initial, dtype=dtype, device=device)
        )
        master = torch.nn.Parameter(parameter.detach().float().clone())
        target = torch.zeros(shape, dtype=dtype, device=device)
        qad = QADModifier(gradient_accumulation_steps=4, max_grad_norm=0.001)
        qad._batches = [None] * 5
        optimizer = torch.optim.SGD([master], lr=0.1)
        scaler = torch.amp.GradScaler(device.type, enabled=enabled)
        with patch.object(
            qad,
            "_batch_loss",
            side_effect=lambda _: _masked_mse(parameter.expand(shape), target, None),
        ):
            steps = qad._train_epoch(
                optimizer, [parameter], [master], list(range(5)), scaler
            )
        assert steps == 2
        results[dtype, enabled] = master.detach().cpu()
    # The unscaled FP16 backward loses every output gradient at this shape.
    assert results[torch.float16, False].item() == initial
    torch.testing.assert_close(
        results[torch.float16, True], results[torch.float32, False], rtol=0.01, atol=0
    )
    assert results[torch.float16, True].item() < initial * 0.9


def test_fp16_overflow_retries_accumulation_group():
    device = get_main_device()
    parameter = torch.nn.Parameter(
        torch.tensor([1.0], dtype=torch.float16, device=device)
    )
    master = torch.nn.Parameter(parameter.detach().float().clone())
    qad = QADModifier(gradient_accumulation_steps=2, max_grad_norm=None)
    qad._batches = [1.0, 3.0, 5.0]
    optimizer = torch.optim.SGD([master], lr=0.1)
    scaler = torch.amp.GradScaler(device.type)
    with (
        patch.object(
            qad,
            "_batch_loss",
            side_effect=lambda target: (parameter.float() - target).square().sum(),
        ),
        patch.object(optimizer, "step", wraps=optimizer.step) as step,
    ):
        steps = qad._train_epoch(optimizer, [parameter], [master], [0, 1, 2], scaler)
    assert steps == step.call_count == 2
    assert scaler.get_scale() < 65536
    # Overflow retries must not omit either the complete or partial group.
    torch.testing.assert_close(master.cpu(), torch.tensor([1.96]), rtol=0.001, atol=0)
    torch.testing.assert_close(parameter.float(), master, rtol=0.001, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_nonfinite_gradients_fail_without_updating_weights(dtype):
    device = get_main_device()
    parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=dtype, device=device))
    master = torch.nn.Parameter(parameter.detach().float().clone())
    parameter.register_hook(lambda grad: torch.full_like(grad, float("nan")))
    qad = QADModifier()
    qad._batches = [None]
    optimizer = torch.optim.AdamW([master])
    scaler = torch.amp.GradScaler(device.type, enabled=dtype == torch.float16)
    with patch.object(
        qad, "_batch_loss", side_effect=lambda _: parameter.float().sum()
    ):
        with pytest.raises(ValueError, match="Nonfinite QAD gradient"):
            qad._train_epoch(optimizer, [parameter], [master], [0], scaler)
    assert parameter.item() == master.item() == 1.0
    assert not optimizer.state
