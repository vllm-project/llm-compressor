"""
Unit and integration tests for distributed SmoothQuantModifier.

Unit tests (no GPU): mock torch.distributed to verify the all_reduce
call contract without any real hardware.

Multi-GPU integration test: uses nm-testing/tinysmokellama-3.2 (tiny
model safe for CI) to verify weight equivalence between single-GPU and
distributed SmoothQuant runs.

Run unit tests only:
    pytest tests/.../test_smoothquant_distributed.py -m unit -v

Run integration test (requires 2 GPUs):
    pytest tests/.../test_smoothquant_distributed.py -m multi_gpu -v
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier
from tests.testing_utils import requires_gpu

# ---------------------------------------------------------------------------
# Unit tests — mock-based, no GPU required
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_reduce_activation_scales_noop_single_gpu():
    """_reduce_activation_scales must be a no-op when is_distributed() is False."""
    from llmcompressor.modifiers.transform.smoothquant.base import (
        SmoothQuantModifier,
        SmoothQuantScale,
    )

    modifier = SmoothQuantModifier()
    modifier.resolved_mappings_ = []
    modifier.scales_ = {
        "layer0": SmoothQuantScale(
            min_channel_vals=torch.tensor([-1.0, -2.0]),
            max_channel_vals=torch.tensor([1.0, 2.0]),
        )
    }

    with (
        patch(
            "llmcompressor.modifiers.transform.smoothquant.base.is_distributed",
            return_value=False,
        ),
        patch("llmcompressor.modifiers.transform.smoothquant.base.dist") as mock_dist,
    ):
        modifier._reduce_activation_scales()
        mock_dist.all_reduce.assert_not_called()


@pytest.mark.unit
def test_reduce_activation_scales_2n_calls_for_n_layers():
    """For N layers, exactly 2*N async all_reduce calls must be batched."""
    from llmcompressor.modifiers.transform.smoothquant.base import (
        SmoothQuantModifier,
        SmoothQuantScale,
    )

    n = 4
    modifier = SmoothQuantModifier()
    modifier.resolved_mappings_ = []
    modifier.scales_ = {
        f"layer{i}": SmoothQuantScale(
            min_channel_vals=torch.zeros(16),
            max_channel_vals=torch.ones(16),
        )
        for i in range(n)
    }

    collected = []

    with (
        patch(
            "llmcompressor.modifiers.transform.smoothquant.base.is_distributed",
            return_value=True,
        ),
        patch("llmcompressor.modifiers.transform.smoothquant.base.dist") as mock_dist,
        patch(
            "llmcompressor.modifiers.transform.smoothquant.base.wait_for_comms",
            side_effect=lambda h: collected.extend(h),
        ),
    ):
        mock_dist.all_reduce.return_value = MagicMock()
        mock_dist.ReduceOp.MIN = torch.distributed.ReduceOp.MIN
        mock_dist.ReduceOp.MAX = torch.distributed.ReduceOp.MAX

        modifier._reduce_activation_scales()

        assert mock_dist.all_reduce.call_count == 2 * n
        assert len(collected) == 2 * n


@pytest.mark.unit
def test_apply_smoothing_calls_reduce_only_when_distributed():
    """
    _reduce_activation_scales() must be called inside _apply_smoothing()
    only when is_distributed() is True, and skipped otherwise.
    """
    from llmcompressor.modifiers.transform.smoothquant.base import SmoothQuantModifier

    modifier = SmoothQuantModifier()
    modifier.scales_ = {}
    modifier.resolved_mappings_ = []

    # Distributed case: reduce should be called
    order = []
    with patch.object(
        modifier,
        "_reduce_activation_scales",
        side_effect=lambda: order.append("reduce"),
    ):
        with patch(
            "llmcompressor.modifiers.transform.smoothquant.base.is_distributed",
            return_value=True,
        ):
            modifier._apply_smoothing(model=MagicMock())
    assert order == ["reduce"]

    # Single-GPU case: reduce should NOT be called
    order_single = []
    with patch.object(
        modifier,
        "_reduce_activation_scales",
        side_effect=lambda: order_single.append("reduce"),
    ):
        with patch(
            "llmcompressor.modifiers.transform.smoothquant.base.is_distributed",
            return_value=False,
        ):
            modifier._apply_smoothing(model=MagicMock())
    assert order_single == []


# ---------------------------------------------------------------------------
# Multi-GPU integration test
# ---------------------------------------------------------------------------


def _prepare_dataset(model_id: str, num_samples: int):
    """Prepare calibration dataset for SmoothQuant."""
    tok = AutoTokenizer.from_pretrained(model_id)

    # Use a simple default chat template for models without one
    if tok.chat_template is None:
        tok.chat_template = (
            "{% for message in messages %}{{ message['content'] }}{% endfor %}"
        )

    ds = load_dataset(
        "HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{num_samples}]"
    )
    ds = ds.map(
        lambda ex: {"text": tok.apply_chat_template(ex["messages"], tokenize=False)}
    )
    ds = ds.map(
        lambda s: tok(s["text"], padding=False, max_length=512, truncation=True),
        remove_columns=ds.column_names,
    )
    return ds


def _run_single_gpu_smoothquant(
    model_id: str, num_samples: int, device: str = "cuda:0"
):
    """Run SmoothQuant on a single GPU and return smoothed weights."""
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, device_map=device
    )
    ds = _prepare_dataset(model_id, num_samples)

    oneshot(
        model=model,
        dataset=ds,
        recipe=SmoothQuantModifier(smoothing_strength=0.8),
        num_calibration_samples=num_samples,
        max_seq_length=512,
    )

    # Extract smoothed weights (input_layernorm and q_proj affected by SmoothQuant)
    weights = {
        name: param.clone().cpu()
        for name, param in model.named_parameters()
        if "input_layernorm" in name or "q_proj" in name
    }

    del model
    torch.cuda.empty_cache()

    return weights


@pytest.mark.integration
@pytest.mark.multi_gpu
@requires_gpu(2)
def test_smoothquant_distributed_weights_match_single_gpu(tmp_path):
    """
    Verify that distributed SmoothQuant produces the same smoothed weights
    as single-GPU SmoothQuant on the same calibration data.

    Uses nm-testing/tinysmokellama-3.2 (tiny model, safe for CI).
    Each distributed rank gets half the calibration samples.
    After all_reduce, both ranks should agree on the same smoothing scales,
    producing weights identical to the single-GPU reference (atol=1e-4).
    """
    MODEL = "nm-testing/tinysmokellama-3.2"
    NUM_SAMPLES = 32

    # ------------------------------------------------------------------
    # Single-GPU reference
    # ------------------------------------------------------------------
    single_out = tmp_path / "single_weights.pt"
    ref = _run_single_gpu_smoothquant(MODEL, NUM_SAMPLES)
    torch.save(ref, single_out)

    # ------------------------------------------------------------------
    # Distributed run (2 ranks, each gets half via get_rank_partition)
    # ------------------------------------------------------------------
    ddp_out = tmp_path / "ddp_weights.pt"
    ddp_runner = Path(__file__).parent / "_smoothquant_ddp_runner.py"

    r = subprocess.run(
        [
            "torchrun",
            "--standalone",
            "--nproc_per_node=2",
            str(ddp_runner),
            MODEL,
            str(NUM_SAMPLES),
            str(ddp_out),
        ],
        capture_output=True,
        text=True,
    )
    assert (
        r.returncode == 0
    ), f"DDP run failed:\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}"

    ddp = torch.load(ddp_out, weights_only=True)

    # ------------------------------------------------------------------
    # Compare
    # ------------------------------------------------------------------
    assert set(ref.keys()) == set(ddp.keys())
    mismatches = []
    for name in ref:
        try:
            torch.testing.assert_close(ref[name], ddp[name], atol=1e-4, rtol=1e-4)
        except AssertionError as e:
            mismatches.append(f"  {name}: {e}")

    assert not mismatches, (
        "Distributed SmoothQuant weights differ from single-GPU:\n"
        + "\n".join(mismatches)
    )
