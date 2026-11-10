"""Tests for the ceil_x observer."""

import pytest
import torch
from compressed_tensors.quantization import fake_quantize
from compressed_tensors.quantization.quant_args import QuantizationArgs

from llmcompressor.observers import Observer
from llmcompressor.observers.ceil_x import (
    CeilXObserver,
    MXFP4CeilXObserver,
    _ceil_x_adjust,
)


class TestCeilXAdjust:
    """Tests for the core _ceil_x_adjust function."""

    def test_symmetric_basic(self):
        """Adjusted absmax = 2^(ceil(log2(max/c + eps)) + EMAX), with hardcoded values.

        Hand-computed (c=7.25, EMAX=2):
          absmax=3.0  -> log2(0.4138)=-1.27 -> ceil=-1 -> 2^(-1+2)=2.0
          absmax=10.0 -> log2(1.3793)=0.46  -> ceil= 1 -> 2^( 1+2)=8.0
          absmax=1.0  -> log2(0.1379)=-2.86 -> ceil=-2 -> 2^(-2+2)=1.0

        These values are independent of the implementation's own formula, so
        they catch bugs in the exponent/EMAX arithmetic rather than restating it.
        """
        cases = [
            (3.0, 2.0),
            (10.0, 8.0),
            (1.0, 1.0),
        ]
        for absmax_in, expected_absmax in cases:
            min_vals = torch.tensor([-absmax_in])
            max_vals = torch.tensor([absmax_in])
            adj_min, adj_max = _ceil_x_adjust(min_vals, max_vals, 7.25)
            assert torch.allclose(
                adj_max, torch.tensor([expected_absmax]), rtol=1e-5
            ), f"absmax={absmax_in}: got {adj_max.item()}, expected {expected_absmax}"
            assert torch.allclose(adj_min, torch.tensor([-expected_absmax]), rtol=1e-5)

    def test_different_c_values(self):
        """Larger c crosses the ceil discontinuity to a smaller exponent.

        With absmax=10.0 the ceil(log2(10/c)) boundary at log2=0 (i.e. c=10)
        separates ceil=1 (c<10) from ceil=0 (c>=10), so adj jumps from
        2^(1+2)=8.0 to 2^(0+2)=4.0.
        """
        min_vals = torch.tensor([-10.0])
        max_vals = torch.tensor([10.0])

        # c in [6, 7.25, 9] -> ceil=1 -> 8.0 ; c=12 -> ceil=0 -> 4.0
        expected = {6.0: 8.0, 7.25: 8.0, 9.0: 8.0, 12.0: 4.0}
        for c, exp_adj in expected.items():
            _, adj_max = _ceil_x_adjust(min_vals, max_vals, c)
            assert torch.allclose(
                adj_max, torch.tensor([exp_adj]), rtol=1e-5
            ), f"c={c}: got {adj_max.item()}, expected {exp_adj}"

    def test_extreme_inputs_finite(self):
        """Zero and very large inputs should not produce NaN or inf."""
        for absmax in [0.0, 1e30]:
            min_vals = torch.tensor([-absmax])
            max_vals = torch.tensor([absmax])
            adj_min, adj_max = _ceil_x_adjust(min_vals, max_vals, 7.25)
            assert torch.isfinite(adj_min).all()
            assert torch.isfinite(adj_max).all()


class TestCeilXObserverRegistration:
    """Tests for observer registration, construction, and validation."""

    def test_registered_as_ceil_x(self):
        """The observer should be loadable from the registry as 'ceil_x'."""
        args = QuantizationArgs(
            num_bits=4,
            type="float",
            symmetric=True,
            strategy="tensor_group",
            group_size=32,
            observer="ceil_x",
        )
        obs = Observer.load_from_registry("ceil_x", base_name="weight", args=args)
        assert isinstance(obs, CeilXObserver)

    def test_mxfp4_ceil_x_alias_produces_identical_qparams(self):
        """``mxfp4_ceil_x`` is a pure ``pass`` subclass with identical defaults,
        so it must produce bit-identical qparams. Guards against future
        divergence between the alias and the base class.
        """
        torch.manual_seed(2024)
        weight = torch.randn(8, 32)
        args = QuantizationArgs(
            num_bits=4,
            type="float",
            symmetric=True,
            strategy="tensor_group",
            group_size=32,
        )
        args_base = args.model_copy(update={"observer": "ceil_x"})
        args_alias = args.model_copy(update={"observer": "mxfp4_ceil_x"})

        obs_base = Observer.load_from_registry(
            "ceil_x", base_name="weight", args=args_base
        )
        obs_alias = Observer.load_from_registry(
            "mxfp4_ceil_x", base_name="weight", args=args_alias
        )
        qp_base = obs_base(weight).get_qparams()
        qp_alias = obs_alias(weight).get_qparams()

        assert isinstance(obs_alias, MXFP4CeilXObserver)
        assert torch.equal(qp_base["scale"], qp_alias["scale"])
        assert torch.equal(qp_base["zero_point"], qp_alias["zero_point"])
        assert torch.equal(qp_base["global_scale"], qp_alias["global_scale"])

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"ceil_x_value": 5.0}, "ceil_x_value must be in"),
            ({"ceil_x_value": 13.0}, "ceil_x_value must be in"),
            (
                {"enable_search": True, "search_min": 8.0, "search_max": 8.0},
                "search_max.*must be greater than",
            ),
        ],
    )
    def test_rejects_invalid_kwargs(self, kwargs, match):
        """Invalid ceil_x_value or search range should raise ValueError."""
        args = QuantizationArgs(
            num_bits=4, type="float", symmetric=True, observer="ceil_x"
        )
        with pytest.raises(ValueError, match=match):
            Observer.load_from_registry(
                "ceil_x", base_name="weight", args=args, **kwargs
            )

    def test_rejects_int_type_and_asymmetric(self):
        """INT type and asymmetric quantization should raise ValueError."""
        with pytest.raises(ValueError, match="ceil_x only supports FLOAT type"):
            args = QuantizationArgs(
                num_bits=8, type="int", symmetric=True, observer="ceil_x"
            )
            Observer.load_from_registry("ceil_x", base_name="weight", args=args)
        with pytest.raises(ValueError, match="ceil_x only supports symmetric"):
            args = QuantizationArgs(
                num_bits=4, type="float", symmetric=False, observer="ceil_x"
            )
            Observer.load_from_registry("ceil_x", base_name="weight", args=args)


class TestCeilXObserverFP4:
    """Tests with FLOAT type (MXFP4) quantization."""

    def test_mxfp4_produces_valid_qparams(self):
        """The observer should produce valid qparams for MXFP4."""
        module = torch.nn.Linear(64, 32)
        args = QuantizationArgs(
            num_bits=4,
            type="float",
            symmetric=True,
            strategy="tensor_group",
            group_size=32,
            observer="ceil_x",
        )
        obs = Observer.load_from_registry("ceil_x", base_name="weight", args=args)
        qparams = obs(module.weight).get_qparams()
        scale, zero_point, global_scale = (
            qparams["scale"],
            qparams["zero_point"],
            qparams["global_scale"],
        )

        assert torch.isfinite(scale).all()
        assert (scale > 0).all()
        assert torch.isfinite(zero_point).all()
        assert global_scale is not None
        assert torch.isfinite(global_scale).all()

    def test_mxfp4_global_scale_at_least_minmax(self):
        """ceil_x shared exponent >= minmax shared exponent.

        This is the core algorithmic claim: ceil(log2(m/c)) rounds the
        per-block exponent *up*, while floor(log2(m)) (minmax) rounds it
        *down*. A larger-or-equal shared exponent means ceil_x never
        truncates large values more aggressively than minmax, so the
        global_scale (which encodes the tensor-wide max shared exponent)
        must be >= the minmax-derived global_scale.
        """
        torch.manual_seed(123)
        module = torch.nn.Linear(128, 64)
        module.weight.data = torch.randn_like(module.weight) * 0.1

        args = QuantizationArgs(
            num_bits=4,
            type="float",
            symmetric=True,
            strategy="tensor_group",
            group_size=32,
        )

        obs_minmax = Observer.load_from_registry(
            "memoryless_minmax", base_name="weight", args=args
        )
        gs_minmax = obs_minmax(module.weight).get_qparams()["global_scale"]

        args_ceilx = args.model_copy(update={"observer": "ceil_x"})
        obs_ceilx = Observer.load_from_registry(
            "ceil_x", base_name="weight", args=args_ceilx
        )
        gs_ceilx = obs_ceilx(module.weight).get_qparams()["global_scale"]

        assert gs_ceilx >= gs_minmax, (
            f"ceil_x global_scale ({gs_ceilx.item()}) < minmax ({gs_minmax.item()}); "
            "ceil_x must round the shared exponent up, never below minmax"
        )


class TestCeilXSearch:
    """Tests for the MSE search functionality."""

    def test_search_no_worse_than_default(self):
        """Search over a grid that includes the default must be no worse.

        With search_step=0.25 and search_min=6.0, the candidate grid is
        {6.0, 6.25, ..., 12.0}, which includes the default 7.25. Since the
        search evaluates MSE at every candidate and picks the minimum, the
        result must be <= the default's MSE — no tolerance buffer needed.
        A regression that makes search pick a non-optimal candidate would
        fail this exactly.
        """
        torch.manual_seed(99)
        tensor = torch.randn(16, 128) * 10.0
        args = QuantizationArgs(
            num_bits=4,
            type="float",
            symmetric=True,
            strategy="tensor_group",
            group_size=32,
        )

        # Default (no search)
        args_default = args.model_copy(update={"observer": "ceil_x"})
        obs_default = Observer.load_from_registry(
            "ceil_x", base_name="weight", args=args_default, ceil_x_value=7.25
        )
        qp_default = obs_default(tensor).get_qparams()
        qdq_default = fake_quantize(
            tensor,
            qp_default["scale"],
            qp_default["zero_point"],
            args_default,
            global_scale=qp_default["global_scale"],
        )
        mse_default = ((qdq_default - tensor) ** 2).mean().item()

        # With search (grid includes 7.25)
        obs_search = Observer.load_from_registry(
            "ceil_x",
            base_name="weight",
            args=args_default,
            enable_search=True,
            search_min=6.0,
            search_max=12.0,
            search_step=0.25,
        )
        qp_search = obs_search(tensor).get_qparams()
        qdq_search = fake_quantize(
            tensor,
            qp_search["scale"],
            qp_search["zero_point"],
            args_default,
            global_scale=qp_search["global_scale"],
        )
        mse_search = ((qdq_search - tensor) ** 2).mean().item()

        # Search includes the default as a candidate, so it must be <= it.
        # Use a tiny rtol to absorb floating-point summation order differences.
        assert mse_search <= mse_default * (1 + 1e-6), (
            f"Search MSE ({mse_search:.8f}) > default ({mse_default:.8f}); "
            "search grid includes the default so this is a regression"
        )


class TestCeilXWeightObserver:
    """Tests for weight-specific observer behavior."""

    def test_weight_observed_once(self):
        """Weight observer should not update statistics on second observation."""
        tensor = torch.randn(16, 64)
        args = QuantizationArgs(
            num_bits=4,
            type="float",
            symmetric=True,
            strategy="tensor_group",
            group_size=32,
            observer="ceil_x",
        )
        obs = Observer.load_from_registry("ceil_x", base_name="weight", args=args)
        obs(tensor)
        first_max = obs.max_vals.clone()
        obs(tensor * 100)  # Very different values
        assert torch.equal(obs.max_vals, first_max)