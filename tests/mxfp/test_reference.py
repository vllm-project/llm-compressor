# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Harness checks and explicit counterexamples; passing does NOT mean MX parity."""

import os
from pathlib import Path

import pytest
import torch

from .differential import cases, llmc_linear, llmc_tensor
from .oracle import MxOracle


@pytest.fixture(scope="module")
def oracle():
    root = Path(
        os.environ.get(
            "MODELSLIM_SOURCE",
            Path(__file__).resolve().parents[4] / "reference/msmodelslim",
        )
    )
    if not (root / "msmodelslim/ir/api/impl/mx_quantization.py").is_file():
        if os.environ.get("MXFP_REQUIRE_REFERENCE") == "1":
            pytest.fail("MXFP oracle requires an external ModelSlim checkout")
        pytest.skip("Set MODELSLIM_SOURCE to the external reference checkout")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield MxOracle(root)
    torch.set_num_threads(previous)


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("role", ["weight", "activation"])
def test_exact_control(oracle, bits, dtype, role):
    value = dict(cases(bits, dtype))["exact-control"]
    expected, actual = oracle.tensor(value, bits), llmc_tensor(value, bits, role)
    for key in expected:
        torch.testing.assert_close(
            actual[key].float(), expected[key].float(), atol=0, rtol=0
        )


@pytest.mark.parametrize("bits,ms_value,llmc_value", [(4, 0.5, 0), (8, 1.125, 1)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("role", ["weight", "activation"])
def test_halfway_counterexample(oracle, bits, ms_value, llmc_value, dtype, role):
    value = dict(cases(bits, dtype))["halfway"]
    expected, actual = oracle.tensor(value, bits), llmc_tensor(value, bits, role)
    torch.testing.assert_close(
        expected["exponent"].float(), actual["exponent"], atol=0, rtol=0
    )
    assert expected["dequantized"][0, 0].item() == ms_value
    assert actual["dequantized"][0, 0].item() == llmc_value
    assert expected["dequantized"][0, 1].item() == -ms_value
    assert actual["dequantized"][0, 1].item() == -llmc_value


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mxfp8_scale_counterexample(oracle, dtype):
    value = dict(cases(8, dtype))["scale-boundary-7.5"]
    expected, actual = oracle.tensor(value, 8), llmc_tensor(value, 8, "weight")
    assert expected["exponent"].item() == -6
    assert actual["exponent"].item() == -5
    assert expected["dequantized"][0, -1].item() == 7
    assert actual["dequantized"][0, -1].item() == 7.5


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mxfp4_zero_and_small_value_policy(oracle, dtype):
    for case in ("zeros", "tiny-1e-08"):
        value = dict(cases(4, dtype))[case]
        expected, actual = oracle.tensor(value, 4), llmc_tensor(value, 4, "activation")
        assert (expected["exponent"] == -22).all()
        assert not torch.equal(expected["exponent"].float(), actual["exponent"])
        assert torch.count_nonzero(expected["dequantized"]) == 0
        assert bool(torch.count_nonzero(actual["dequantized"])) == (case != "zeros")


def test_mxfp4_bf16_scale_boundary(oracle):
    value = dict(cases(4, torch.bfloat16))["scale-boundary-6.96875"]
    assert oracle.tensor(value, 4)["exponent"].item() == 1
    assert llmc_tensor(value, 4, "weight")["exponent"].item() == 0


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.no_grad()
def test_real_linear_drivers_match_their_own_components(oracle, bits, dtype):
    generator = torch.Generator().manual_seed(42)
    x = torch.randn(2, 3, 64, generator=generator).to(dtype)
    w = torch.randn(16, 64, generator=generator).to(dtype)
    original = oracle.linear(x, w, bits)
    expected = torch.nn.functional.linear(
        oracle.tensor(x, bits)["dequantized"], oracle.tensor(w, bits)["dequantized"]
    )
    torch.testing.assert_close(original, expected, atol=0, rtol=0)
    actual = llmc_linear(x, w, bits)
    components = torch.nn.functional.linear(
        llmc_tensor(x, bits, "activation")["dequantized"],
        llmc_tensor(w, bits, "weight")["dequantized"],
    )
    torch.testing.assert_close(actual, components, atol=0, rtol=0)


@pytest.mark.parametrize("bits", [4, 8])
def test_last_axis_groups_and_noncontiguous_activation(oracle, bits):
    maximum = 4 if bits == 4 else 256
    value = torch.full((2, 3, 128), float(maximum))[..., ::2]
    value[..., 32:] *= 2
    assert not value.is_contiguous()
    reference, actual = (
        oracle.tensor(value, bits),
        llmc_tensor(value, bits, "activation"),
    )
    expected = torch.tensor([0, 1]).expand(2, 3, 2).float()
    torch.testing.assert_close(reference["exponent"], expected, atol=0, rtol=0)
    torch.testing.assert_close(actual["exponent"], expected, atol=0, rtol=0)
    torch.testing.assert_close(
        reference["dequantized"], actual["dequantized"], atol=0, rtol=0
    )
