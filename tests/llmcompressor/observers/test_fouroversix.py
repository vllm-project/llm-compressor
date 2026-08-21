from unittest.mock import patch

import pytest
import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy

from llmcompressor.observers import Observer


def test_fouroversix_searches_only_1x_and_1_5x():
    """Verify the fouroversix observer evaluates exactly two shrink factors:
    1.0x and 1.5x of the observed per-group range."""
    args = QuantizationArgs(
        num_bits=4,
        type="float",
        strategy=QuantizationStrategy.TENSOR_GROUP,
        group_size=16,
        observer="fouroversix",
    )

    observer = Observer.load_from_registry("fouroversix", base_name="weight", args=args)

    torch.manual_seed(0)
    weight = torch.randn(1, 32, 16, dtype=torch.bfloat16)

    shrink_factors = []
    orig_eager = (
        __import__("llmcompressor.observers.mse_quant", fromlist=["_grid_search_eager"])
    )._grid_search_eager

    def patched_eager(
        observed,
        args,
        token_args,
        min_val,
        max_val,
        best_error,
        best_min_val,
        best_max_val,
        total_steps,
        patience,
        grid,
        norm,
    ):
        for i in range(total_steps):
            p = 1 - i / grid
            shrink_factors.append(round(p * observer.expand, 6))
        return orig_eager(
            observed,
            args,
            token_args,
            min_val,
            max_val,
            best_error,
            best_min_val,
            best_max_val,
            total_steps,
            patience,
            grid,
            norm,
        )

    with patch("llmcompressor.observers.mse_quant._grid_search_eager", patched_eager):
        observer(weight)

    assert (
        len(shrink_factors) == 2
    ), f"Expected 2 search points, got {len(shrink_factors)}"
    assert shrink_factors[0] == pytest.approx(
        1.5, abs=1e-4
    ), f"First search point should be 1.5x, got {shrink_factors[0]}"
    assert shrink_factors[1] == pytest.approx(
        1.0, abs=1e-4
    ), f"Second search point should be 1.0x, got {shrink_factors[1]}"


def test_fouroversix_is_nvfp4_default():
    """NVFP4 (FP4 + TENSOR_GROUP) should default to fouroversix observer."""
    args = QuantizationArgs(
        num_bits=4,
        type="float",
        strategy=QuantizationStrategy.TENSOR_GROUP,
        group_size=16,
    )
    assert args.observer == "fouroversix"


def test_non_nvfp4_does_not_default_to_fouroversix():
    """Non-NVFP4 quantization should not use fouroversix."""
    # MXFP4 (GROUP strategy)
    mxfp4 = QuantizationArgs(
        num_bits=4,
        type="float",
        strategy=QuantizationStrategy.GROUP,
        group_size=32,
    )
    assert mxfp4.observer != "fouroversix"

    # INT8
    int8 = QuantizationArgs(num_bits=8, type="int")
    assert int8.observer != "fouroversix"
