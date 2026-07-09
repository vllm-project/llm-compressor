"""
Multi-GPU smoke tests for compression modifiers with distributed data parallel.

These tests verify that running compression with DDP produces the same final
weights as running on a single GPU. This ensures that cross-rank synchronization
of statistics (observers, Hessians, activation scales) is working correctly.

Uses nm-testing/tinysmokeqwen3moe (tiny MoE model safe for CI) with minimal
calibration data to keep test runtime low while still exercising the DDP paths.

These tests use the @torchrun decorator which automatically spawns torchrun
when run with regular pytest.

Run with:
    pytest tests/llmcompressor/transformers/compression/test_compression_ddp.py \
        -m multi_gpu -v
"""

from __future__ import annotations

import os
import tempfile

import pytest
import torch
import torch.distributed
from compressed_tensors.offload import init_dist, load_offloaded_model
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.datasets.utils import get_rank_partition
from llmcompressor.modifiers.autoround import AutoRoundModifier
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq import AWQModifier
from llmcompressor.modifiers.transform.imatrix import IMatrixGatherer
from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier
from tests.testing_utils import requires_gpu, torchrun

# Test configuration
MODEL = "nm-testing/tinysmokeqwen3moe"
NUM_SAMPLES = 16  # Small number for smoke test
MAX_SEQ_LENGTH = 512


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def _prepare_dataset(model_id: str, num_samples: int):
    """Prepare calibration dataset with minimal preprocessing."""
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
            s["text"], padding=False, max_length=MAX_SEQ_LENGTH, truncation=True
        ),
        remove_columns=ds.column_names,
    )
    ds.set_format("torch")
    return ds


def _run_single_gpu(
    model_id: str,
    recipe,
    num_samples: int,
    device: str = "cuda:0",
    return_model: bool = False,
):
    """
    Run oneshot compression on a single GPU and return weights (control/reference).

    Args:
        model_id: HuggingFace model ID
        recipe: Compression recipe
        num_samples: Number of calibration samples
        device: Device to run on
        return_model: If True, return (weights, model, dataset) instead of just weights

    Returns:
        weights dict if return_model=False, else (weights, model, dataset)
    """
    with load_offloaded_model():
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="auto_offload"
        )
    ds = _prepare_dataset(model_id, num_samples)

    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        num_calibration_samples=num_samples,
        max_seq_length=MAX_SEQ_LENGTH,
    )

    # Extract quantized weights (exclude common ignored parameters)
    weights = {
        name: param.clone().cpu()
        for name, param in model.named_parameters()
        if "weight" in name and "lm_head" not in name
    }

    if return_model:
        return weights, model, ds

    del model
    torch.cuda.empty_cache()

    return weights


def _compare_weights(ref_weights: dict, ddp_weights: dict, atol: float = 1e-4):
    """Compare weights from single-GPU and DDP runs."""
    assert set(ref_weights.keys()) == set(ddp_weights.keys()), (
        f"Weight keys mismatch: "
        f"ref has {len(ref_weights)} weights, "
        f"ddp has {len(ddp_weights)} weights"
    )

    mismatches = []
    for name in ref_weights:
        try:
            torch.testing.assert_close(
                ref_weights[name], ddp_weights[name], atol=atol, rtol=1e-4
            )
        except AssertionError as e:
            mismatches.append(f"  {name}: {e}")

    return mismatches


def _compare_outputs(ref_model, ddp_model, dataset, num_samples: int = 5):
    """
    Compare model outputs between single-GPU and DDP runs.

    Returns tuple of (kl_div, top1_match_rate)
    """
    ref_model.eval()
    ddp_model.eval()

    kl_divs = []
    top1_matches = 0
    top1_total = 0

    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            sample = dataset[i]
            inputs = {
                k: v.unsqueeze(0).to("cuda:0")
                for k, v in sample.items()
                if k == "input_ids"
            }

            ref_out = ref_model(**inputs).logits[0].float().cpu()
            ddp_out = ddp_model(**inputs).logits[0].float().cpu()

            ref_log_probs = torch.nn.functional.log_softmax(ref_out, dim=-1)
            ddp_log_probs = torch.nn.functional.log_softmax(ddp_out, dim=-1)
            kl = torch.nn.functional.kl_div(
                ddp_log_probs, ref_log_probs, log_target=True, reduction="batchmean"
            )
            kl_divs.append(kl.item())

            top1_matches += (
                (ref_out.argmax(dim=-1) == ddp_out.argmax(dim=-1)).sum().item()
            )
            top1_total += ref_out.shape[0]

    kl_div = sum(kl_divs) / len(kl_divs)
    top1_match_rate = top1_matches / top1_total

    return kl_div, top1_match_rate


# ---------------------------------------------------------------------------
# Shared Test Implementation
# ---------------------------------------------------------------------------


def _test_ddp_modifier(
    test_id,
    recipe_factory,
    pipeline,
    offload_device,
    weight_atol=1e-4,
    min_top1_match=0.95,
    max_kl_div=0.1,
    offload_folder=None,
):
    """
    Shared implementation for DDP smoke tests.

    Verifies that running compression with DDP produces the same final outputs
    and weights as running on a single GPU. This ensures proper cross-rank
    synchronization of statistics (observers, Hessians, activation scales, etc.).

    Each distributed rank gets a different partition of the calibration samples.
    After synchronization, all ranks should produce identical (or very similar) results

    Args:
        test_id: Identifier for the test configuration
        recipe_factory: Callable that returns a fresh modifier instance
        pipeline: Pipeline type ("independent" or "sequential")
        offload_device: Device to offload model weights (None, "cpu", or "disk")
        weight_atol: Absolute tolerance for weight comparison (default 1e-4)
        min_top1_match: Minimum top-1 prediction match rate (default 0.95 = 95%)
        max_kl_div: Maximum allowed mean KL divergence (default 0.1)
        offload_folder: Optional folder for disk offload (default None)
    """
    # Set deterministic seed for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # Single-GPU reference (run BEFORE init_dist)
    ref_weights, ref_model, ref_dataset = _run_single_gpu(
        MODEL, recipe_factory(), NUM_SAMPLES, return_model=True
    )

    # Initialize distributed
    init_dist()
    rank = torch.distributed.get_rank()

    # Set deterministic seed again for DDP run
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # Load model with offloading (standard pattern for DDP)
    load_kwargs = {
        "dtype": torch.bfloat16,
        "device_map": "auto_offload",
    }
    if offload_device == "cpu":
        # Force model weights to CPU by making CPU attractive
        load_kwargs["max_memory"] = {"cpu": "1000GB"}
    elif offload_device == "disk":
        # Force model weights to disk by restricting CPU
        load_kwargs["max_memory"] = {"cpu": "1GB"}
        if offload_folder:
            load_kwargs["offload_folder"] = offload_folder

    with load_offloaded_model():
        model = AutoModelForCausalLM.from_pretrained(MODEL, **load_kwargs)

    # Prepare dataset with rank partitioning
    ds = _prepare_dataset(MODEL, NUM_SAMPLES)

    # Run oneshot with DDP
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe_factory(),
        num_calibration_samples=NUM_SAMPLES,
        max_seq_length=MAX_SEQ_LENGTH,
        pipeline=pipeline,
    )

    # Extract DDP weights (exclude common ignored parameters)
    ddp_weights = {
        name: param.clone().cpu()
        for name, param in model.named_parameters()
        if "weight" in name and "lm_head" not in name
    }

    torch.distributed.barrier()

    # Compare (only rank 0)
    if rank == 0:
        # Compare outputs first (primary correctness test)
        kl_div, top1_match_rate = _compare_outputs(
            ref_model, model, ref_dataset, num_samples=5
        )

        print(
            f"[Rank {rank}] Results: top1_match={100 * top1_match_rate:.1f}%, "
            f"kl_div={kl_div:.6f}",
            flush=True,
        )

        # Primary test: outputs should be very similar
        errors = []
        if top1_match_rate < min_top1_match:
            errors.append(
                f"  Top-1 match rate: {100 * top1_match_rate:.1f}% "
                f"(expected >= {100 * min_top1_match:.0f}%)"
            )
        if kl_div > max_kl_div:
            errors.append(
                f"  Mean KL divergence: {kl_div:.6f} (expected <= {max_kl_div})"
            )

        assert not errors, (
            f"Distributed {test_id} outputs differ significantly:\n" + "\n".join(errors)
        )

        # Clean up reference model
        del ref_model
        torch.cuda.empty_cache()

        # Secondary test: weights should be similar (with configurable tolerance)
        mismatches = _compare_weights(ref_weights, ddp_weights, atol=weight_atol)
        assert not mismatches, (
            f"Distributed {test_id} weights differ from single-GPU:\n"
            + "\n".join(mismatches)
        )

    # Clean up DDP model on all ranks
    del model
    torch.cuda.empty_cache()
    torch.distributed.barrier()


w8a8_static_rtn = {
    "group_0": QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=8,
            type="int",
            symmetric=True,
            strategy="channel",
        ),
        input_activations=QuantizationArgs(
            num_bits=8,
            type="int",
            symmetric=True,
            strategy="tensor",
            observer="static_minmax",
        ),
    )
}

w8a8_static_mse = {
    "group_0": QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=8,
            type="int",
            symmetric=True,
            strategy="channel",
            observer="mse",
        ),
        input_activations=QuantizationArgs(
            num_bits=8,
            type="int",
            symmetric=True,
            strategy="tensor",
            observer="static_minmax",
        ),
    )
}

# ---------------------------------------------------------------------------
# Individual Test Functions (each gets its own torchrun subprocess)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.multi_gpu
@requires_gpu(2)
@torchrun(world_size=2)
def test_ddp_smoke_smoothquant():
    _test_ddp_modifier(
        "smoothquant",
        lambda: SmoothQuantModifier(smoothing_strength=0.8),
        "independent",
        None,
        weight_atol=1e-4,
    )


@pytest.mark.integration
@pytest.mark.multi_gpu
@requires_gpu(2)
@torchrun(world_size=2)
def test_ddp_smoke_imatrix():
    _test_ddp_modifier(
        "imatrix",
        lambda: [
            IMatrixGatherer(ignore=["lm_head"]),
            QuantizationModifier(
                scheme={"W4A16": ["Linear"]},
                ignore=["lm_head"],
            ),
        ],
        "independent",
        None,
        weight_atol=5e-3,
        min_top1_match=1.00,
        max_kl_div=0.00,
    )


@pytest.mark.integration
@pytest.mark.multi_gpu
@requires_gpu(2)
@torchrun(world_size=2)
def test_ddp_smoke_mse_cpu_offload():
    _test_ddp_modifier(
        "mse_cpu_offload",
        lambda: QuantizationModifier(
            config_groups=w8a8_static_mse,
            ignore=["lm_head"],
        ),
        "independent",
        "cpu",
        weight_atol=1e-5,
        min_top1_match=1.00,
        max_kl_div=0.00,
    )


@pytest.mark.integration
@pytest.mark.multi_gpu
@requires_gpu(2)
@torchrun(world_size=2)
def test_ddp_smoke_rtn_disk_offload():
    # Use path under pytest's temp directory (excluded from conftest size check)
    offload_folder = os.path.join(
        tempfile.gettempdir(),
        f"pytest-of-{os.getenv('USER', 'user')}",
        "ddp_disk_offload",
    )
    _test_ddp_modifier(
        "rtn_disk_offload",
        lambda: QuantizationModifier(
            config_groups=w8a8_static_rtn,
            ignore=["lm_head"],
        ),
        "independent",
        "disk",
        weight_atol=1e-5,
        min_top1_match=1.00,
        max_kl_div=0.00,
        offload_folder=offload_folder,
    )


@pytest.mark.integration
@pytest.mark.multi_gpu
@requires_gpu(2)
@torchrun(world_size=2)
def test_ddp_smoke_awq():
    _test_ddp_modifier(
        "awq",
        lambda: [
            AWQModifier(duo_scaling=True),
            QuantizationModifier(
                scheme={"W4A16": ["Linear"]},
                ignore=["lm_head"],
            ),
        ],
        "independent",
        None,
        weight_atol=5e-2,
        min_top1_match=0.85,
        max_kl_div=0.001,
    )


@pytest.mark.integration
@pytest.mark.multi_gpu
@requires_gpu(2)
@torchrun(world_size=2)
def test_ddp_smoke_gptq():
    _test_ddp_modifier(
        "gptq",
        lambda: GPTQModifier(
            scheme={"W4A16": ["Linear"]},
            ignore=["lm_head"],
            block_size=128,
        ),
        "independent",
        None,
        weight_atol=1e-1,
        min_top1_match=0.95,
        max_kl_div=0.001,
    )


@pytest.mark.integration
@pytest.mark.multi_gpu
@requires_gpu(2)
@torchrun(world_size=2)
def test_ddp_smoke_autoround():
    _test_ddp_modifier(
        "autoround",
        lambda: AutoRoundModifier(
            ignore=["lm_head"],
            iters=10,  # Small number for smoke test
            scheme="W4A16",
        ),
        "independent",
        None,
        weight_atol=1e-1,
        min_top1_match=0.85,
        max_kl_div=0.01,
    )
