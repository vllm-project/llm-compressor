"""Tests for the MXFP4 ceil_x observer."""

import pytest
import torch
from compressed_tensors.quantization import (
    fake_quantize,
    preset_name_to_scheme,
)
from compressed_tensors.quantization.quant_args import QuantizationArgs
from compressed_tensors.quantization.utils import calculate_qparams
from compressed_tensors.registry import standardize_lookup_name

from llmcompressor.observers import Observer
from llmcompressor.observers.ceil_x import (
    MXFP4CeilXObserver,
    _mxfp4_ceil_x_adjust,
)
from llmcompressor.observers.helpers import flatten_for_calibration
from llmcompressor.observers.mse import MemorylessMSEObserver


def _mxfp4_args(observer: str | None = None, **overrides) -> QuantizationArgs:
    """Default MXFP4 weight quantization args for observer tests.

    Mirrors the weights args of the ``MXFP4A16`` preset: GROUP strategy
    with 32-value blocks and uint8 (E8M0) shared scales.
    """
    kwargs = dict(
        num_bits=4,
        type="float",
        symmetric=True,
        strategy="group",
        group_size=32,
        scale_dtype=torch.uint8,
    )
    if observer is not None:
        kwargs["observer"] = observer
    kwargs.update(overrides)
    return QuantizationArgs(**kwargs)


class TestCeilXAdjust:
    """Tests for the core _mxfp4_ceil_x_adjust function."""

    def test_default_divisor_basic(self):
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
            adj_min, adj_max = _mxfp4_ceil_x_adjust(min_vals, max_vals)
            assert torch.allclose(
                adj_max, torch.tensor([expected_absmax]), rtol=1e-5
            ), f"absmax={absmax_in}: got {adj_max.item()}, expected {expected_absmax}"
            assert torch.allclose(adj_min, torch.tensor([-expected_absmax]), rtol=1e-5)

    def test_default_divisor_is_7_25(self):
        """The default divisor must be the tuned constant 7.25."""
        min_vals = torch.tensor([-10.0])
        max_vals = torch.tensor([10.0])
        adj_default_min, adj_default_max = _mxfp4_ceil_x_adjust(min_vals, max_vals)
        adj_explicit_min, adj_explicit_max = _mxfp4_ceil_x_adjust(
            min_vals, max_vals, 7.25
        )
        assert torch.equal(adj_default_min, adj_explicit_min)
        assert torch.equal(adj_default_max, adj_explicit_max)

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
            _, adj_max = _mxfp4_ceil_x_adjust(min_vals, max_vals, c)
            assert torch.allclose(
                adj_max, torch.tensor([exp_adj]), rtol=1e-5
            ), f"c={c}: got {adj_max.item()}, expected {exp_adj}"

    def test_extreme_inputs_finite(self):
        """Zero and very large inputs should not produce NaN or inf."""
        for absmax in [0.0, 1e30]:
            min_vals = torch.tensor([-absmax])
            max_vals = torch.tensor([absmax])
            adj_min, adj_max = _mxfp4_ceil_x_adjust(min_vals, max_vals)
            assert torch.isfinite(adj_min).all()
            assert torch.isfinite(adj_max).all()


class TestMXFP4CeilXObserver:
    """Registration, validation, and qparams for the mxfp4_ceil_x observer."""

    def test_registered_as_mxfp4_ceil_x(self):
        """The observer loads as a MemorylessMSEObserver subclass.

        Only the mxfp4-specific name exists: the generic ``ceil_x`` name and
        the separate ``mxfp4_ceil_x_mse`` variant are intentionally absent.
        """
        args = _mxfp4_args(observer="mxfp4_ceil_x")
        obs = Observer.load_from_registry("mxfp4_ceil_x", base_name="weight", args=args)
        assert isinstance(obs, MXFP4CeilXObserver)
        assert isinstance(obs, MemorylessMSEObserver)

        # The registry standardizes names for lookup: underscores and spaces
        # become hyphens (e.g. ``memoryless_minmax`` -> ``memoryless-minmax``),
        # so compare against the standardized spellings.
        names = Observer.registered_names()
        assert standardize_lookup_name("mxfp4_ceil_x") in names
        assert standardize_lookup_name("ceil_x") not in names
        assert standardize_lookup_name("mxfp4_ceil_x_mse") not in names

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"type": "int", "num_bits": 8}, "only supports FLOAT type"),
            ({"num_bits": 8}, "only supports 4-bit"),
            ({"num_bits": 6}, "only supports 4-bit"),
            ({"symmetric": False}, "only supports symmetric"),
            ({"strategy": "tensor_group"}, "only supports the GROUP strategy"),
            ({"group_size": 64}, "requires MXFP4 scale generation"),
            ({"scale_dtype": None}, "requires MXFP4 scale generation"),
        ],
    )
    def test_rejects_non_mxfp4_args(self, overrides, match):
        """Args that do not describe MXFP4 (GROUP, 32-value, uint8) must raise."""
        args = _mxfp4_args(observer="mxfp4_ceil_x", **overrides)
        with pytest.raises(ValueError, match=match):
            Observer.load_from_registry("mxfp4_ceil_x", base_name="weight", args=args)

    def test_mxfp4a16_preset_produces_valid_qparams(self):
        """The documented recipe path must construct and quantize cleanly.

        Real MXFP4 presets (``MXFP4A16``) use GROUP strategy with 32-value
        blocks and uint8 E8M0 scales; the observer must accept them. This is
        a regression test for a validation that used to require TENSOR_GROUP
        and rejected every real MXFP4 recipe.
        """
        args = preset_name_to_scheme("MXFP4A16", ["Linear"]).weights
        assert args.strategy == "group"
        assert args.group_size == 32
        assert args.scale_dtype == torch.uint8
        args = args.model_copy(update={"observer": "mxfp4_ceil_x"})

        module = torch.nn.Linear(64, 32)
        obs = Observer.load_from_registry("mxfp4_ceil_x", base_name="weight", args=args)
        qparams = obs(module.weight).get_qparams()
        scale, zero_point = qparams["scale"], qparams["zero_point"]

        # MX scales are positive powers of two produced from E8M0 exponents
        assert torch.isfinite(scale).all()
        assert (scale > 0).all()
        log2_scales = torch.log2(scale)
        assert torch.allclose(log2_scales, torch.round(log2_scales), atol=1e-6)

        # Symmetric MX quantization has no zero points and no global scale
        assert torch.equal(zero_point, torch.zeros_like(zero_point))
        assert qparams["global_scale"] is None

    def test_maxshrink_zero_reproduces_fixed_ceil_x(self):
        """An empty search grid returns the ceil_x adjusted range unchanged.

        ``maxshrink=0`` disables the grid search, so the observer must be
        equivalent to applying ``_mxfp4_ceil_x_adjust`` and deriving qparams
        from the adjusted range directly.
        """
        torch.manual_seed(3)
        module = torch.nn.Linear(64, 32)
        args = _mxfp4_args(observer="mxfp4_ceil_x")
        obs = Observer.load_from_registry(
            "mxfp4_ceil_x", base_name="weight", args=args, maxshrink=0.0
        )
        qparams = obs(module.weight).get_qparams()

        observed = flatten_for_calibration(module.weight, "weight", args)
        raw_min = torch.amin(observed, dim=(0, -1))
        raw_max = torch.amax(observed, dim=(0, -1))
        adj_min, adj_max = _mxfp4_ceil_x_adjust(raw_min, raw_max)
        expected_scale, expected_zero_point = calculate_qparams(
            adj_min, adj_max, args, global_scale=None
        )

        assert torch.equal(qparams["scale"], expected_scale)
        assert torch.equal(qparams["zero_point"], expected_zero_point)

    def test_search_no_worse_than_fixed_ceil_x(self):
        """The grid includes the ceil_x range itself (multiplier 1), so the
        per-group search result must be no worse than fixed ceil_x.

        The search selects per-group candidate ranges minimizing the
        powered QDQ error with norm=2 (squared error), so the summed
        end-to-end MSE must be <= the fixed ceil_x MSE — no tolerance
        buffer needed beyond floating-point summation order.
        """
        torch.manual_seed(99)
        tensor = torch.randn(16, 128) * 10.0
        args = _mxfp4_args(observer="mxfp4_ceil_x")

        # Fixed ceil_x (search disabled)
        obs_fixed = Observer.load_from_registry(
            "mxfp4_ceil_x", base_name="weight", args=args, maxshrink=0.0
        )
        qp_fixed = obs_fixed(tensor).get_qparams()
        qdq_fixed = fake_quantize(
            tensor, qp_fixed["scale"], qp_fixed["zero_point"], args
        )
        mse_fixed = ((qdq_fixed - tensor) ** 2).mean().item()

        # MSE search from the ceil_x range
        obs_search = Observer.load_from_registry(
            "mxfp4_ceil_x", base_name="weight", args=args, norm=2.0
        )
        qp_search = obs_search(tensor).get_qparams()
        qdq_search = fake_quantize(
            tensor, qp_search["scale"], qp_search["zero_point"], args
        )
        mse_search = ((qdq_search - tensor) ** 2).mean().item()

        # The ceil_x range is the first grid candidate, so the per-group
        # selection must be no worse than it under the same error norm.
        assert mse_search <= mse_fixed * (1 + 1e-5), (
            f"MSE-search MSE ({mse_search:.8f}) > fixed ceil_x MSE "
            f"({mse_fixed:.8f}); the search grid includes the ceil_x range "
            "so this is a regression"
        )

    def test_search_refines_outlier_block(self):
        """A block dominated by values below the outlier chooses a finer exponent.

        Hand-computed for the block ``[3.9, 0.76 * 31]``:
        - minmax rounds 3.9 up (round_to_power_2 at mantissa >= 1.75) to an
          E8M0 of 127, i.e. a scale of 1.0 and a representable max of 6.0.
        - the ceil_x base range is [-4, 4]; shrinking it by 1/2 yields an
          E8M0 of 126 (scale 0.5), which clamps only 3.9 (to 3.0) while
          halving the step for the remaining values, so the search selects
          it. In squared error (norm=2, used only for this hand-count; the
          observer's default norm=2.4 ranks the candidates identically):
          0.81 + 31 * 0.0001 versus 0.01 + 31 * 0.0576.
        """
        block = torch.tensor([[3.9] + [0.76] * 31] * 2)
        args = _mxfp4_args()

        obs = Observer.load_from_registry("mxfp4_ceil_x", base_name="weight", args=args)
        scale = obs(block).get_qparams()["scale"]
        assert torch.allclose(scale, torch.full_like(scale, 0.5))

        obs_minmax = Observer.load_from_registry(
            "memoryless_minmax", base_name="weight", args=args
        )
        scale_minmax = obs_minmax(block).get_qparams()["scale"]
        assert torch.allclose(scale_minmax, torch.full_like(scale_minmax, 1.0))

    def test_weight_observed_once(self):
        """Weight observers should not update statistics on re-observation."""
        tensor = torch.randn(16, 64)
        args = _mxfp4_args(observer="mxfp4_ceil_x")
        obs = Observer.load_from_registry("mxfp4_ceil_x", base_name="weight", args=args)
        obs(tensor)
        first_max = obs.max_vals.clone()
        obs(tensor * 100)  # Very different values
        assert torch.equal(obs.max_vals, first_max)
