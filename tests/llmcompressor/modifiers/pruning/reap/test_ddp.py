"""
Multi-GPU tests for REAP expert pruning with DDP.

Verifies that running REAP with DDP (data partitioned across ranks) produces
the same pruning decisions as running on a single GPU (full dataset).

Run with:
    pytest tests/llmcompressor/modifiers/pruning/reap/test_ddp.py \
        -m multi_gpu -v
"""

from __future__ import annotations

import pickle
import tempfile

import pytest
import torch
import torch.distributed as dist
from compressed_tensors.offload import init_dist
from transformers import AutoModelForCausalLM

from llmcompressor import oneshot
from llmcompressor.datasets.utils import get_rank_partition
from llmcompressor.modifiers.pruning.reap import REAPPruningModifier
from llmcompressor.utils import load_context
from tests.testing_utils import requires_gpu, torchrun

QWEN_MODEL = "inference-optimization/Qwen3.8-1.0B-A0.6B"
NUM_SAMPLES = 16
MAX_SEQ_LENGTH = 512


def _load_report(path):
    with open(path, "rb") as f:
        return pickle.load(f)


@pytest.mark.integration
@pytest.mark.multi_gpu
@requires_gpu(2)
@torchrun(world_size=2)
def test_reap_ddp_qwen3():
    """REAP with DDP on Qwen3.8-1.0B-A0.6B scores the same experts as single-GPU."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ref_report = f"{tmpdir}/ref_report.pkl"

        torch.manual_seed(42)
        torch.get_device_module().manual_seed_all(42)

        # Single-GPU reference (before init_dist)
        with load_context():
            model_ref = AutoModelForCausalLM.from_pretrained(
                QWEN_MODEL, dtype=torch.bfloat16, device_map="auto_offload"
            )

        oneshot(
            model=model_ref,
            dataset="perfectblend",
            splits=f"train[:{NUM_SAMPLES}]",
            recipe=REAPPruningModifier(sparsity=0.25, report_path=ref_report),
            num_calibration_samples=NUM_SAMPLES,
            max_seq_length=MAX_SEQ_LENGTH,
            shuffle_calibration_samples=False,
            pipeline="sequential",
        )

        ref_saliency = _load_report(ref_report)
        del model_ref
        torch.accelerator.empty_cache()

        # DDP run
        init_dist()
        rank = dist.get_rank()

        ddp_report = f"{tmpdir}/ddp_report.pkl"

        torch.manual_seed(42)
        torch.get_device_module().manual_seed_all(42)

        with load_context():
            model_ddp = AutoModelForCausalLM.from_pretrained(
                QWEN_MODEL, dtype=torch.bfloat16, device_map="auto_offload"
            )

        oneshot(
            model=model_ddp,
            dataset="perfectblend",
            splits=get_rank_partition("train", NUM_SAMPLES),
            recipe=REAPPruningModifier(sparsity=0.25, report_path=ddp_report),
            num_calibration_samples=NUM_SAMPLES,
            max_seq_length=MAX_SEQ_LENGTH,
            shuffle_calibration_samples=False,
            pipeline="sequential",
        )

        if rank == 0:
            ddp_saliency = _load_report(ddp_report)
            assert len(ref_saliency) == len(ddp_saliency), (
                f"Number of MoE layers differ between single-GPU and DDP.\n"
                f"  ref: {len(ref_saliency)}\n"
                f"  ddp: {len(ddp_saliency)}"
            )
            for layer_idx, (ref_layer, ddp_layer) in enumerate(
                zip(ref_saliency, ddp_saliency)
            ):
                torch.testing.assert_close(
                    torch.tensor(ref_layer),
                    torch.tensor(ddp_layer),
                    msg=lambda m, i=layer_idx: (
                        f"Saliency scores differ between single-GPU and DDP "
                        f"at layer {i}.\n{m}"
                    ),
                )

        del model_ddp
        torch.accelerator.empty_cache()
        dist.barrier()
