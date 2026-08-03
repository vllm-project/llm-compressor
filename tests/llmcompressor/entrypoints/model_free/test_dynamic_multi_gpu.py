"""
Multi-GPU tests for the dynamic shard scheduler.

Saves a tiny model with many small safetensors shards, then verifies that
quantizing across N GPUs produces the same result as a single-GPU run.
"""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import model_free_ptq
from tests.testing_utils import requires_gpu


def _save_sharded(model_id, out_dir, max_shard_size="1MB"):
    """Re-save a HF model with a tiny shard size so we get lots of files."""
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
    tok = AutoTokenizer.from_pretrained(model_id)
    model.save_pretrained(out_dir, max_shard_size=max_shard_size)
    tok.save_pretrained(out_dir)


@pytest.mark.multi_gpu
@requires_gpu(2)
def test_multi_gpu_matches_single_gpu(tmp_path):
    """Dynamic scheduling across N GPUs should give identical output to
    a single-GPU run."""
    model_id = "Qwen/Qwen3-0.6B"
    scheme = "FP8_dynamic"
    ignore = ["model.embed_tokens", "lm_head"]

    sharded_dir = tmp_path / "sharded_model"
    _save_sharded(model_id, sharded_dir, max_shard_size="1MB")

    single_out = tmp_path / "single_gpu"
    multi_out = tmp_path / "multi_gpu"

    model_free_ptq(
        str(sharded_dir),
        single_out,
        scheme=scheme,
        max_workers=1,
        device="cuda:0",
        ignore=ignore,
    )

    n = torch.accelerator.device_count()
    model_free_ptq(
        str(sharded_dir),
        multi_out,
        scheme=scheme,
        max_workers=n,
        device=[f"cuda:{i}" for i in range(n)],
        ignore=ignore,
    )

    _assert_outputs_equal(single_out, multi_out)


@pytest.mark.multi_gpu
@requires_gpu(2)
def test_multi_gpu_more_workers_than_shards(tmp_path):
    """Scheduler should be fine when there are more threads than shards."""
    model_id = "Qwen/Qwen3-0.6B"
    scheme = "FP8_dynamic"
    ignore = ["model.embed_tokens", "lm_head"]

    out = tmp_path / "output"
    n = torch.accelerator.device_count()

    model_free_ptq(
        model_id,
        out,
        scheme=scheme,
        max_workers=n * 2,
        device=[f"cuda:{i}" for i in range(n)],
        ignore=ignore,
    )

    st_files = list(out.glob("*.safetensors"))
    assert st_files, "No safetensors files produced"


# ── helpers ──────────────────────────────────────────────────────────────


def _assert_outputs_equal(dir_a, dir_b):
    """Load every tensor from both dirs and compare names, values, dtypes."""
    from safetensors.torch import load_file

    a = {}
    b = {}
    for f in sorted(dir_a.glob("*.safetensors")):
        a.update(load_file(f))
    for f in sorted(dir_b.glob("*.safetensors")):
        b.update(load_file(f))

    assert a, "No tensors loaded from dir_a"
    assert set(a) == set(b), (
        f"Keys differ — only in a: {a.keys() - b.keys()}, "
        f"only in b: {b.keys() - a.keys()}"
    )

    for k in a:
        assert torch.equal(a[k], b[k]), f"{k}: values differ"
        assert a[k].dtype == b[k].dtype, f"{k}: dtype {a[k].dtype} vs {b[k].dtype}"
