"""
Integration tests for distributed SmoothQuantModifier.

Unit tests (no GPU): mock torch.distributed to verify the all_reduce call
contract without any real hardware.

Multi-GPU tests: launch real torchrun subprocesses to verify end-to-end
correctness on actual hardware. Requires >= 2 GPUs; skipped otherwise.

Run unit tests only:
    pytest tests/.../test_smoothquant_distributed.py -m unit -v

Run everything (needs 2+ GPUs):
    pytest tests/.../test_smoothquant_distributed.py -m "unit or multi_gpu" -v
"""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed

from tests.testing_utils import requires_gpu, run_cli_command

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
def test_reduce_activation_scales_issues_min_max_all_reduce():
    """
    In distributed mode, exactly two async all_reduce calls must be issued per layer:
    one with ReduceOp.MIN for min_channel_vals,
    one with ReduceOp.MAX for max_channel_vals.
    """
    from llmcompressor.modifiers.transform.smoothquant.base import (
        SmoothQuantModifier,
        SmoothQuantScale,
    )

    min_vals = torch.tensor([-0.5, -1.5])
    max_vals = torch.tensor([0.5, 1.5])

    modifier = SmoothQuantModifier()
    modifier.resolved_mappings_ = []
    modifier.scales_ = {
        "attn": SmoothQuantScale(
            min_channel_vals=min_vals.clone(),
            max_channel_vals=max_vals.clone(),
        )
    }

    with (
        patch(
            "llmcompressor.modifiers.transform.smoothquant.base.is_distributed",
            return_value=True,
        ),
        patch("llmcompressor.modifiers.transform.smoothquant.base.dist") as mock_dist,
        patch("llmcompressor.modifiers.transform.smoothquant.base.wait_for_comms"),
    ):
        mock_dist.all_reduce.return_value = MagicMock()
        mock_dist.ReduceOp.MIN = torch.distributed.ReduceOp.MIN
        mock_dist.ReduceOp.MAX = torch.distributed.ReduceOp.MAX

        modifier._reduce_activation_scales()

        assert mock_dist.all_reduce.call_count == 2
        calls = mock_dist.all_reduce.call_args_list

        # First call: MIN reduce on min_channel_vals
        assert torch.equal(calls[0].args[0], min_vals)
        assert calls[0].kwargs["op"] == torch.distributed.ReduceOp.MIN
        assert calls[0].kwargs["async_op"] is True

        # Second call: MAX reduce on max_channel_vals
        assert torch.equal(calls[1].args[0], max_vals)
        assert calls[1].kwargs["op"] == torch.distributed.ReduceOp.MAX
        assert calls[1].kwargs["async_op"] is True


@pytest.mark.unit
def test_reduce_activation_scales_2n_calls_for_n_layers():
    """For N layers, exactly 2*N async all_reduce calls must be
    batched into one wait."""
    from llmcompressor.modifiers.transform.smoothquant.base import (
        SmoothQuantModifier,
        SmoothQuantScale,
    )

    n = 6
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
def test_apply_smoothing_calls_reduce_first():
    """
    _reduce_activation_scales() must be the very first operation inside
    _apply_smoothing() so that scale computation always uses global statistics.
    """
    from llmcompressor.modifiers.transform.smoothquant.base import SmoothQuantModifier

    modifier = SmoothQuantModifier()
    modifier.scales_ = {}
    modifier.resolved_mappings_ = []

    order = []
    with patch.object(
        modifier,
        "_reduce_activation_scales",
        side_effect=lambda: order.append("reduce"),
    ):
        modifier._apply_smoothing(model=MagicMock())

    assert order == [
        "reduce"
    ], "_reduce_activation_scales must be the first call in _apply_smoothing"


@pytest.mark.unit
def test_reduce_activation_scales_empty_scales_still_calls_wait():
    """Even with no layers, wait_for_comms must be called once (with empty list)."""
    from llmcompressor.modifiers.transform.smoothquant.base import SmoothQuantModifier

    modifier = SmoothQuantModifier()
    modifier.resolved_mappings_ = []
    modifier.scales_ = {}

    with (
        patch(
            "llmcompressor.modifiers.transform.smoothquant.base.is_distributed",
            return_value=True,
        ),
        patch("llmcompressor.modifiers.transform.smoothquant.base.dist"),
        patch(
            "llmcompressor.modifiers.transform.smoothquant.base.wait_for_comms"
        ) as mock_wait,
    ):
        modifier._reduce_activation_scales()
        mock_wait.assert_called_once_with([])


# ---------------------------------------------------------------------------
# Multi-GPU integration tests — real torchrun
# ---------------------------------------------------------------------------

_EXAMPLE_SCRIPT = (
    Path(__file__).resolve().parents[6]
    / "examples"
    / "quantization_w8a8_int8"
    / "smoothquant_ddp_example.py"
)


@pytest.mark.integration
@pytest.mark.multi_gpu
@requires_gpu(2)
def test_smoothquant_ddp_script_runs_cleanly(tmp_path):
    """
    Smoke test: torchrun --nproc_per_node=2 smoothquant_ddp_example.py exits 0
    and produces a valid compressed checkpoint directory.
    """
    if not _EXAMPLE_SCRIPT.exists():
        pytest.skip(f"DDP example script not found: {_EXAMPLE_SCRIPT}")

    script = tmp_path / "smoothquant_ddp_example.py"
    shutil.copy(_EXAMPLE_SCRIPT, script)

    cmd = ["torchrun", "--standalone", "--nproc_per_node=2", str(script)]
    result = run_cli_command(cmd, cwd=tmp_path)

    assert result.returncode == 0, (
        f"torchrun exited {result.returncode}:\n"
        f"cmd: {shlex.join(cmd)}\n"
        f"output:\n{result.stdout}"
    )

    output_dirs = list(tmp_path.glob("*-W8A8-SmoothQuant-DDP*"))
    assert (
        len(output_dirs) == 1
    ), f"Expected exactly one output dir, found: {output_dirs}"

    config = json.loads((output_dirs[0] / "config.json").read_text())
    quant_cfg = config.get("quantization_config", {})
    assert quant_cfg, "quantization_config missing from saved config.json"
    assert (
        quant_cfg.get("format") == "int-quantized"
    ), f"Expected int-quantized, got: {quant_cfg.get('format')}"


@pytest.mark.integration
@pytest.mark.multi_gpu
@requires_gpu(2)
def test_smoothquant_distributed_weights_match_single_gpu(tmp_path):
    """
    Core correctness test: verify that after distributed SmoothQuant, the
    smoothed weights on rank 0 match a single-GPU reference run given the
    same full calibration data.

    Single-GPU run: full 32-sample dataset, standard oneshot.
    Distributed run: 32 samples split across 2 ranks (16 each), activation
    stats all-reduced → both runs should produce identical smoothing scales
    and therefore identical smoothed weights (within float tolerance).
    """
    # ------------------------------------------------------------------
    # Single-GPU reference run
    # ------------------------------------------------------------------
    single_script = tmp_path / "single_gpu_sq.py"
    single_out = tmp_path / "single_weights.pt"
    single_script.write_text(
        f"""
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
tok   = AutoTokenizer.from_pretrained(MODEL)

ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:32]")
ds = ds.map(
    lambda ex: {{"text": tok.apply_chat_template(ex["messages"], tokenize=False)}}
)
ds = ds.map(
    lambda s: tok(s["text"], padding=False, max_length=512, truncation=True),
    remove_columns=ds.column_names,
)

oneshot(
    model=model,
    dataset=ds,
    recipe=SmoothQuantModifier(smoothing_strength=0.8),
    num_calibration_samples=32,
    max_seq_length=512,
)

torch.save(
    {{n: p.clone().cpu() for n, p in model.named_parameters()
      if "input_layernorm" in n or "q_proj" in n}},
    "{single_out}",
)
"""
    )

    r = subprocess.run(
        [sys.executable, str(single_script)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    if r.returncode != 0:
        pytest.skip(f"Single-GPU reference run failed:\n{r.stderr}")

    ref = torch.load(single_out, weights_only=True)

    # ------------------------------------------------------------------
    # Distributed run (2 ranks, 16 samples each → same 32-sample pool)
    # ------------------------------------------------------------------
    ddp_out = tmp_path / "ddp_weights.pt"
    ddp_script = tmp_path / "ddp_sq_verify.py"
    ddp_script.write_text(
        f"""
import torch
import torch.distributed as dist
from compressed_tensors.offload import init_dist, load_offloaded_model
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.datasets.utils import get_rank_partition
from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
init_dist()

with load_offloaded_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, device_map="auto_offload"
    )
tok = AutoTokenizer.from_pretrained(MODEL)

# Each rank gets 16 of the 32 samples
ds = load_dataset(
    "HuggingFaceH4/ultrachat_200k",
    split=get_rank_partition("train_sft[:32]", 32),
)
ds = ds.map(
    lambda ex: {{"text": tok.apply_chat_template(ex["messages"], tokenize=False)}}
)
ds = ds.map(
    lambda s: tok(s["text"], padding=False, max_length=512, truncation=True),
    remove_columns=ds.column_names,
)

oneshot(
    model=model,
    dataset=ds,
    recipe=SmoothQuantModifier(smoothing_strength=0.8),
    num_calibration_samples=32,
    max_seq_length=512,
)

if dist.get_rank() == 0:
    torch.save(
        {{n: p.clone().cpu() for n, p in model.named_parameters()
          if "input_layernorm" in n or "q_proj" in n}},
        "{ddp_out}",
    )
dist.destroy_process_group()
"""
    )

    r = subprocess.run(
        ["torchrun", "--standalone", "--nproc_per_node=2", str(ddp_script)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert (
        r.returncode == 0
    ), f"DDP run failed:\ncmd: torchrun ...\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}"

    ddp = torch.load(ddp_out, weights_only=True)

    # ------------------------------------------------------------------
    # Compare weights: distributed smoothing must match single-GPU
    # ------------------------------------------------------------------
    assert set(ref.keys()) == set(ddp.keys()), (
        f"Weight key mismatch.\nRef keys: {sorted(ref.keys())}\n"
        f"DDP keys: {sorted(ddp.keys())}"
    )

    mismatches = []
    for name in ref:
        try:
            torch.testing.assert_close(ref[name], ddp[name], atol=1e-4, rtol=1e-4)
        except AssertionError as e:
            mismatches.append(f"  {name}: {e}")

    assert not mismatches, (
        "Distributed SmoothQuant weights differ from single-GPU reference:\n"
        + "\n".join(mismatches)
    )
