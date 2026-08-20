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
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.datasets.utils import get_rank_partition
from llmcompressor.modifiers.pruning.reap import REAPPruningModifier
from tests.testing_utils import requires_gpu, torchrun

QWEN_MODEL = "inference-optimization/Qwen3.8-1.0B-A0.6B"
NUM_SAMPLES = 16
MAX_SEQ_LENGTH = 512


def _load_report(path):
    with open(path) as f:
        return json.load(f)


def _prepare_dataset(model_id, num_samples, max_seq_length):
    """Prepare calibration dataset with rank-aware partitioning."""
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.chat_template is None:
        tok.chat_template = (
            "{% for message in messages %}{{ message['content'] }}{% endfor %}"
        )
    split = get_rank_partition("train_sft", num_samples)
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=split)
    ds = ds.map(
        lambda ex: {"text": tok.apply_chat_template(ex["messages"], tokenize=False)}
    )
    ds = ds.map(
        lambda s: tok(
            s["text"], padding=False, max_length=max_seq_length, truncation=True
        ),
        remove_columns=ds.column_names,
    )
    ds.set_format("torch")
    return ds


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
        ds_ref = _prepare_dataset(QWEN_MODEL, NUM_SAMPLES, MAX_SEQ_LENGTH)

        oneshot(
            model=model_ref,
            dataset=ds_ref,
            recipe=REAPPruningModifier(sparsity=0.25, report_path=ref_report),
            num_calibration_samples=NUM_SAMPLES,
            max_seq_length=MAX_SEQ_LENGTH,
            pipeline="sequential",
        )

        ref_retained = _load_report(ref_report)
        del model_ref, ds_ref
        torch.accelerator.empty_cache()

        # DDP run
        init_dist()
        rank = dist.get_rank()

        ddp_report = f"{tmpdir}/ddp_report_rank{rank}.json"

        torch.manual_seed(42)
        torch.get_device_module().manual_seed_all(42)

        with load_offloaded_model():
            model_ddp = AutoModelForCausalLM.from_pretrained(
                QWEN_MODEL, dtype=torch.bfloat16, device_map="auto_offload"
            )
        ds_ddp = _prepare_dataset(QWEN_MODEL, NUM_SAMPLES, MAX_SEQ_LENGTH)

        oneshot(
            model=model_ddp,
            dataset=ds_ddp,
            recipe=REAPPruningModifier(sparsity=0.25, report_path=ddp_report),
            num_calibration_samples=NUM_SAMPLES,
            max_seq_length=MAX_SEQ_LENGTH,
            pipeline="sequential",
        )

        ddp_retained = _load_report(ddp_report)

        if rank == 0:
            assert ref_retained == ddp_retained, (
                f"Retained experts differ between single-GPU and DDP.\n"
                f"  ref: {ref_retained}\n"
                f"  ddp: {ddp_retained}"
            )

        del model_ddp, ds_ddp
        torch.accelerator.empty_cache()
        dist.barrier()
