# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compare public LLMC observers/CT QDQ against the executable external oracle."""

import torch
from compressed_tensors.quantization import QuantizationScheme
from compressed_tensors.quantization.lifecycle.forward import dequantize, quantize
from compressed_tensors.quantization.quant_scheme import MXFP4, MXFP8
from compressed_tensors.quantization.utils import compute_dynamic_scales_and_zp

from llmcompressor.core import EventType
from llmcompressor.core.lifecycle import CompressionLifecycle
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.observers import MinMaxObserver
from llmcompressor.utils.helpers import DisableQuantization


def llmc_tensor(value, bits, role):
    scheme = QuantizationScheme(targets=["Linear"], **(MXFP4 if bits == 4 else MXFP8))
    args = scheme.weights if role == "weight" else scheme.input_activations
    if role == "weight":
        scale = MinMaxObserver("weight", args)(value).get_qparams()["scale"]
    else:
        scale, _ = compute_dynamic_scales_and_zp(value, args, torch.nn.Identity())
    quantized = quantize(value, scale, None, args, dtype=value.dtype)
    dq = dequantize(quantized, scale, None, args, dtype=value.dtype)
    exponent = scale.float().log2()
    return {
        "exponent": exponent,
        "scale": scale.float(),
        "e8m0": (exponent + 127).to(torch.uint8),
        "quantized": quantized,
        "dequantized": dq,
    }


def llmc_linear(x, weight, bits):
    model = torch.nn.Sequential(
        torch.nn.Linear(
            weight.shape[1], weight.shape[0], bias=False, dtype=weight.dtype
        )
    ).eval()
    model[0].weight.data.copy_(weight)
    lifecycle = CompressionLifecycle()
    lifecycle.initialize(model=model, recipe=QuantizationModifier(scheme=f"MXFP{bits}"))
    lifecycle.event(EventType.CALIBRATION_START)
    with DisableQuantization(model):
        model(x)
    lifecycle.event(EventType.SEQUENTIAL_EPOCH_END, modules=list(model.modules()))
    lifecycle.event(EventType.CALIBRATION_END)
    lifecycle.finalize()
    return model(x)


def compare(reference, actual):
    assert reference.shape == actual.shape
    ref, out = reference.float(), actual.float()
    finite = torch.isfinite(ref) & torch.isfinite(out)
    equal = ref == out
    mismatch = ~equal
    indexes = mismatch.nonzero()
    first = indexes[0].tolist() if indexes.numel() else None
    return {
        "elements": ref.numel(),
        "mismatches": mismatch.sum().item(),
        "reference_nonfinite": (~torch.isfinite(ref)).sum().item(),
        "llmc_nonfinite": (~torch.isfinite(out)).sum().item(),
        "max_abs_error": (ref[finite] - out[finite]).abs().max().item()
        if finite.any()
        else None,
        "first_index": first,
        "first_reference": ref[tuple(first)].item()
        if first is not None and finite[tuple(first)]
        else None,
        "first_llmc": out[tuple(first)].item()
        if first is not None and finite[tuple(first)]
        else None,
    }


def cases(bits, dtype):
    for seed in (0, 42, 1234):
        generator = torch.Generator().manual_seed(seed)
        yield f"random-{seed}", torch.randn(8, 64, generator=generator).to(dtype)
    yield "zeros", torch.zeros(2, 64, dtype=dtype)
    # Keep each block maximum fixed to isolate the rounding rule from scale selection.
    ties = (
        [0.25, -0.25, 1.25, -1.25, 2.5, -2.5, 5, -5]
        if bits == 4
        else [1.0625, -1.0625, 1.1875, -1.1875, 1.3125, -1.3125]
    )
    value = torch.zeros(1, 32, dtype=dtype)
    value[0, : len(ties)] = torch.tensor(ties, dtype=dtype)
    value[0, -1] = 6 if bits == 4 else 256
    yield "halfway", value
    for maximum in (6.5, 6.96875, 7.0, 7.5, 8.0):
        value = torch.linspace(-maximum, maximum, 32, dtype=dtype).unsqueeze(0)
        yield f"scale-boundary-{maximum}", value
    for magnitude in (1e-8, 1e-30):
        yield f"tiny-{magnitude}", torch.full((2, 32), magnitude, dtype=dtype)
    # Representable values with a fixed unit scale must agree on both sides.
    exact = (
        torch.tensor([0, 0.5, -0.5, 1, -1, 2, -2, 4], dtype=dtype)
        .repeat(4)
        .unsqueeze(0)
    )
    exact[0, -1] = 4 if bits == 4 else 256
    yield "exact-control", exact


@torch.no_grad()
def run_matrix(oracle):
    tensors, linears = [], []
    for bits in (4, 8):
        for dtype in (torch.float32, torch.bfloat16):
            for name, value in cases(bits, dtype):
                reference = oracle.tensor(value, bits)
                for role in ("weight", "activation"):
                    actual = llmc_tensor(value, bits, role)
                    tensors.append(
                        {
                            "bits": bits,
                            "dtype": str(dtype),
                            "role": role,
                            "case": name,
                            "shape": list(value.shape),
                            "comparisons": {
                                key: compare(reference[key], actual[key])
                                for key in reference
                            },
                        }
                    )
            generator = torch.Generator().manual_seed(42)
            weight = torch.randn(16, 64, generator=generator).to(dtype)
            x = torch.randn(2, 3, 64, generator=generator).to(dtype)
            linears.append(
                {
                    "bits": bits,
                    "dtype": str(dtype),
                    "seed": 42,
                    "comparison": compare(
                        oracle.linear(x, weight, bits), llmc_linear(x, weight, bits)
                    ),
                }
            )
    tensor_failures = sum(
        any(row["mismatches"] for row in case["comparisons"].values())
        for case in tensors
    )
    linear_failures = sum(row["comparison"]["mismatches"] > 0 for row in linears)
    return {
        "parity_passed": tensor_failures == linear_failures == 0,
        "tensor_cases": len(tensors),
        "tensor_cases_with_differences": tensor_failures,
        "linear_cases": len(linears),
        "linear_cases_with_differences": linear_failures,
        "tensors": tensors,
        "linears": linears,
    }
