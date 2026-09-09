"""
Multi-GPU tests for REAP expert pruning with DDP.

Verifies that running REAP with DDP (data partitioned across ranks) produces
the same pruning decisions as running on a single GPU (full dataset).

Run with:
    pytest tests/llmcompressor/modifiers/pruning/reap/test_ddp.py \
        -m multi_gpu -v
"""

from __future__ import annotations

import json
import tempfile

import pytest
import torch
import torch.distributed as dist
from compressed_tensors.offload import init_dist, load_offloaded_model
from transformers import AutoModelForCausalLM

from llmcompressor import oneshot
from llmcompressor.modifiers.pruning.reap import REAPPruningModifier
from tests.testing_utils import requires_gpu, torchrun

QWEN_MODEL = "inference-optimization/Qwen3.8-1.0B-A0.6B"
NUM_SAMPLES = 16
MAX_SEQ_LENGTH = 512


def _load_report(path):
    with open(path) as f:
        return json.load(f)


@pytest.mark.integration
@pytest.mark.multi_gpu
@requires_gpu(2)
@torchrun(world_size=2)
def test_reap_ddp_qwen3():
    """REAP with DDP on Qwen3.8-1.0B-A0.6B retains the same experts as single-GPU."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ref_report = f"{tmpdir}/ref_report.json"

        torch.manual_seed(42)
        torch.get_device_module().manual_seed_all(42)

        # Single-GPU reference (before init_dist)
        with load_offloaded_model():
            model_ref = AutoModelForCausalLM.from_pretrained(
                QWEN_MODEL, dtype=torch.bfloat16, device_map="auto_offload"
            )

        oneshot(
            model=model_ref,
            dataset="perfectblend",
            splits="train[:512]",
            recipe=REAPPruningModifier(sparsity=0.25, report_path=ref_report),
            num_calibration_samples=NUM_SAMPLES,
            max_seq_length=MAX_SEQ_LENGTH,
            shuffle_calibration_samples=False,
            pipeline="sequential",
        )

        ref_retained = _load_report(ref_report)
        del model_ref
        torch.accelerator.empty_cache()

        # DDP run
        init_dist()
        rank = dist.get_rank()

        ddp_report = f"{tmpdir}/ddp_report.json"

        torch.manual_seed(42)
        torch.get_device_module().manual_seed_all(42)

        with load_offloaded_model():
            model_ddp = AutoModelForCausalLM.from_pretrained(
                QWEN_MODEL, dtype=torch.bfloat16, device_map="auto_offload"
            )

        oneshot(
            model=model_ddp,
            dataset="perfectblend",
            splits="train[:512]",
            recipe=REAPPruningModifier(sparsity=0.25, report_path=ddp_report),
            num_calibration_samples=NUM_SAMPLES,
            max_seq_length=MAX_SEQ_LENGTH,
            shuffle_calibration_samples=False,
            pipeline="sequential",
        )

        if rank == 0:
            ddp_retained = _load_report(ddp_report)
            assert ref_retained == ddp_retained, (
                f"Retained experts differ between single-GPU and DDP.\n"
                f"  ref: {ref_retained}\n"
                f"  ddp: {ddp_retained}"
            )

        del model_ddp
        torch.accelerator.empty_cache()
        dist.barrier()
