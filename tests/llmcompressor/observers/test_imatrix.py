import pytest
import torch
from compressed_tensors.quantization import QuantizationStrategy, preset_name_to_scheme
from compressed_tensors.quantization.lifecycle import fake_quantize
from compressed_tensors.quantization.quant_args import QuantizationArgs
from compressed_tensors.quantization.utils import calculate_qparams
from compressed_tensors.utils.impl_backend import ImplBackend

from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.observers.base import Observer
from llmcompressor.observers.helpers import flatten_for_calibration

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _set_importance(observer, importance):
    """Set raw imatrix accumulators so that importance = sum / count."""
    observer._imatrix_sum = importance.clone()
    observer._imatrix_count = torch.tensor(1, dtype=torch.int64)


def _make_linear(in_features=8, out_features=4, seed=42):
    """Create a Linear module."""
    torch.manual_seed(seed)
    return torch.nn.Linear(in_features, out_features)


def _make_importance(in_features=8):
    """Create non-uniform imatrix importance."""
    importance = torch.ones(in_features)
    importance[: in_features // 2] = 10.0
    return importance


def _make_observer(
    module, strategy="channel", group_size=None, importance=None, **kwargs
):
    """Create an imatrix_mse observer and attach it to the module."""
    args = QuantizationArgs(
        num_bits=4,
        symmetric=True,
        strategy=strategy,
        group_size=group_size,
        observer="imatrix_mse",
        observer_kwargs=kwargs,
    )
    observer = Observer.load_from_registry("imatrix_mse", base_name="weight", args=args)
    observer.attach(module)
    if importance is not None:
        _set_importance(observer, importance)
    return observer


# ---------------------------------------------------------------------------
# Activation collection lifecycle
# ---------------------------------------------------------------------------


class TestActivationCollection:
    def test_attach_collects_importance_on_observer(self):
        module = _make_linear(in_features=8, out_features=4)
        observer = _make_observer(module, strategy="channel")

        x = torch.ones(2, 3, 8)
        x[..., 0] = 2.0
        module(x)

        assert observer._imatrix_count.item() == 6
        assert torch.allclose(
            observer._imatrix_sum,
            torch.tensor([24.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0]),
        )
        assert not hasattr(module, "_imatrix_sum")
        assert not hasattr(module, "_imatrix_count")
        assert not hasattr(module, "_imatrix_hook")

    def test_attach_initializes_importance_on_module_device(self):
        module = torch.nn.Linear(8, 4, device="meta")
        observer = _make_observer(module, strategy="channel")

        assert observer._imatrix_sum.device == module.weight.device
        assert observer._imatrix_count.device == module.weight.device

    def test_attach_to_unsupported_module_removes_existing_hook(self):
        module = _make_linear(in_features=8, out_features=4)
        observer = _make_observer(module, strategy="channel")
        old_hook = observer._imatrix_hook

        observer.attach(torch.nn.ReLU())

        assert observer._imatrix_hook is None
        assert old_hook.id not in module._forward_pre_hooks

    def test_detach_stops_collection(self):
        module = _make_linear(in_features=8, out_features=4)
        observer = _make_observer(module, strategy="channel")

        module(torch.ones(2, 3, 8))
        before = observer._imatrix_sum.clone()

        observer.detach(module)
        module(torch.full((2, 3, 8), 100.0))

        assert torch.equal(observer._imatrix_sum, before)
        assert observer._imatrix_hook is None


# ---------------------------------------------------------------------------
# Bug 1: get_global_min_max must not crash on TENSOR_GROUP
# ---------------------------------------------------------------------------


class TestGlobalMinMaxTensorGroup:
    """Regression test for bug #1: global_scale path with TENSOR_GROUP."""

    def test_global_scale_tensor_group_does_not_crash(self):
        """get_global_scale must complete without error on TENSOR_GROUP."""
        module = _make_linear(in_features=8, out_features=4)
        args = QuantizationArgs(
            num_bits=4,
            symmetric=True,
            strategy="tensor_group",
            group_size=4,
            observer="imatrix_mse",
        )
        observer = Observer.load_from_registry(
            "imatrix_mse", base_name="weight", args=args
        )
        observer.attach(module)
        _set_importance(observer, _make_importance(in_features=8))

        global_scale = observer(module.weight).get_qparams()["global_scale"]
        assert global_scale is not None
        assert global_scale.shape == (1,)
        assert torch.isfinite(global_scale).all()

    def test_global_scale_then_forward_tensor_group(self):
        """Full flow: global_scale -> forward must produce valid qparams."""
        module = _make_linear(in_features=8, out_features=4)
        args = QuantizationArgs(
            num_bits=4,
            symmetric=True,
            strategy="tensor_group",
            group_size=4,
            observer="imatrix_mse",
        )
        observer = Observer.load_from_registry(
            "imatrix_mse", base_name="weight", args=args
        )
        observer.attach(module)
        _set_importance(observer, _make_importance(in_features=8))

        qparams = observer(module.weight).get_qparams()
        assert torch.isfinite(qparams["scale"]).all()


# ---------------------------------------------------------------------------
# Basic functionality (sanity checks)
# ---------------------------------------------------------------------------


class TestBasicFunctionality:
    """Sanity checks for the happy path."""

    def test_channel_strategy(self):
        module = _make_linear(in_features=8, out_features=4)
        observer = _make_observer(
            module, strategy="channel", importance=_make_importance(in_features=8)
        )
        qparams = observer(module.weight).get_qparams()
        assert qparams["scale"].shape == (4, 1)
        assert torch.isfinite(qparams["scale"]).all()

    def test_group_strategy(self):
        module = _make_linear(in_features=8, out_features=4)
        observer = _make_observer(
            module,
            strategy="group",
            group_size=4,
            importance=_make_importance(in_features=8),
        )
        qparams = observer(module.weight).get_qparams()
        assert torch.isfinite(qparams["scale"]).all()

    def test_tensor_group_strategy(self):
        module = _make_linear(in_features=8, out_features=4)
        observer = _make_observer(
            module,
            strategy="tensor_group",
            group_size=4,
            importance=_make_importance(in_features=8),
        )
        qparams = observer(module.weight).get_qparams()
        assert torch.isfinite(qparams["scale"]).all()

    def test_block_strategy(self):
        module = _make_linear(in_features=8, out_features=4)
        args = QuantizationArgs(
            num_bits=4,
            symmetric=True,
            strategy="block",
            block_structure=[2, 4],
            observer="imatrix_mse",
        )
        observer = Observer.load_from_registry(
            "imatrix_mse", base_name="weight", args=args
        )
        observer.attach(module)
        _set_importance(observer, _make_importance(in_features=8))
        qparams = observer(module.weight).get_qparams()
        assert torch.isfinite(qparams["scale"]).all()

    def test_no_importance_falls_back(self):
        """Observer without importance data must fall back gracefully."""
        module = torch.nn.Linear(8, 4)
        observer = _make_observer(module, strategy="channel")
        qparams = observer(module.weight).get_qparams()
        assert torch.isfinite(qparams["scale"]).all()

    def test_importance_changes_result(self):
        """Non-uniform importance must produce different scales than uniform."""
        torch.manual_seed(123)
        module_weighted = torch.nn.Linear(8, 4)
        module_uniform = torch.nn.Linear(8, 4)
        module_uniform.weight.data = module_weighted.weight.data.clone()

        # Very skewed importance
        importance = torch.tensor(
            [1000.0, 1000.0, 1000.0, 1000.0, 0.01, 0.01, 0.01, 0.01]
        )
        obs_w = _make_observer(module_weighted, strategy="channel")
        _set_importance(obs_w, importance)
        obs_u = _make_observer(module_uniform, strategy="channel")

        scale_w = obs_w(module_weighted.weight).get_qparams()["scale"]
        scale_u = obs_u(module_uniform.weight).get_qparams()["scale"]

        assert not torch.allclose(
            scale_w, scale_u
        ), "Extreme importance weighting should produce different scales"

    def test_uniform_importance_matches_memoryless_mse(self):
        """All-ones importance must match the uniform MSE observer."""
        torch.manual_seed(123)
        module_imatrix = torch.nn.Linear(8, 4)
        module_mse = torch.nn.Linear(8, 4)
        module_mse.weight.data.copy_(module_imatrix.weight.data)
        module_mse.bias.data.copy_(module_imatrix.bias.data)

        args_uniform = QuantizationArgs(
            num_bits=4,
            symmetric=True,
            strategy="channel",
            observer="memoryless_mse",
            observer_kwargs={"grid": 20},
        )
        obs_imatrix = _make_observer(
            module_imatrix,
            strategy="channel",
            importance=torch.ones(8),
            norm=2.4,
            maxshrink=0.20,
        )
        obs_uniform = Observer.load_from_registry(
            "memoryless_mse", base_name="weight", args=args_uniform
        )

        qparams_i = obs_imatrix(module_imatrix.weight).get_qparams()
        qparams_u = obs_uniform(module_mse.weight).get_qparams()

        assert torch.allclose(qparams_i["scale"], qparams_u["scale"])
        assert torch.equal(qparams_i["zero_point"], qparams_u["zero_point"])


# ---------------------------------------------------------------------------
# Weight-only guard
# ---------------------------------------------------------------------------


class TestWeightOnlyGuard:
    """Regression test: base_name != 'weight' must be rejected."""

    def test_non_weight_base_name_strict_raises(self):
        """strict=True must raise NotImplementedError for non-weight."""
        args = QuantizationArgs(
            num_bits=8,
            symmetric=True,
            strategy="tensor",
            observer="imatrix_mse",
            observer_kwargs={"strict": True},
        )
        observer = Observer.load_from_registry(
            "imatrix_mse", base_name="input", args=args
        )
        observed = torch.randn(2, 1, 8)
        with pytest.raises(NotImplementedError, match="weight observers"):
            observer(observed)

    def test_non_weight_base_name_non_strict_falls_back(self):
        """strict=False must fall back to uniform MSE (no crash)."""
        args = QuantizationArgs(
            num_bits=8,
            symmetric=True,
            strategy="tensor",
            observer="imatrix_mse",
            observer_kwargs={"strict": False},
        )
        observer = Observer.load_from_registry(
            "imatrix_mse", base_name="input", args=args
        )
        observed = torch.randn(2, 1, 8)
        observer(observed)
        qparams = observer.get_qparams()
        scale, zero_point = qparams["scale"], qparams["zero_point"]
        assert torch.isfinite(scale).all()
        assert torch.isfinite(zero_point).all()


# ---------------------------------------------------------------------------
# Validation edge cases
# ---------------------------------------------------------------------------


class TestValidation:
    def test_strict_raises_on_missing_importance(self):
        module = torch.nn.Linear(8, 4)
        observer = _make_observer(module, strategy="channel", strict=True)
        with pytest.raises(ValueError, match="importance"):
            observer(module.weight).get_qparams()

    def test_strict_raises_on_wrong_size(self):
        module = torch.nn.Linear(8, 4)
        observer = _make_observer(
            module, strategy="channel", importance=torch.ones(5), strict=True
        )
        with pytest.raises(ValueError, match="size mismatch"):
            observer(module.weight).get_qparams()

    @pytest.mark.parametrize(
        ("importance", "match"),
        [
            (
                torch.tensor([1.0, float("nan"), 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                "non-finite",
            ),
            (torch.tensor([1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), "negative"),
            (torch.zeros(8), "all zeros"),
        ],
    )
    def test_strict_raises_on_invalid_importance_values(self, importance, match):
        module = torch.nn.Linear(8, 4)
        observer = _make_observer(
            module, strategy="channel", importance=importance, strict=True
        )
        with pytest.raises(ValueError, match=match):
            observer(module.weight).get_qparams()

    @pytest.mark.parametrize(
        "importance",
        [
            torch.tensor([1.0, float("nan"), 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            torch.tensor([1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            torch.zeros(8),
        ],
    )
    def test_non_strict_invalid_importance_falls_back_to_uniform_mse(self, importance):
        module_imatrix = torch.nn.Linear(8, 4)
        module_mse = torch.nn.Linear(8, 4)
        module_mse.weight.data.copy_(module_imatrix.weight.data)
        module_mse.bias.data.copy_(module_imatrix.bias.data)

        args_uniform = QuantizationArgs(
            num_bits=4,
            symmetric=True,
            strategy="channel",
            observer="memoryless_mse",
            observer_kwargs={"grid": 20},
        )
        obs_imatrix = _make_observer(
            module_imatrix, strategy="channel", importance=importance, strict=False
        )
        obs_uniform = Observer.load_from_registry(
            "memoryless_mse", base_name="weight", args=args_uniform
        )

        qparams_i = obs_imatrix(module_imatrix.weight).get_qparams()
        qparams_u = obs_uniform(module_mse.weight).get_qparams()

        assert torch.allclose(qparams_i["scale"], qparams_u["scale"])
        assert torch.equal(qparams_i["zero_point"], qparams_u["zero_point"])

    @pytest.mark.parametrize("norm", [0, -1, float("inf"), float("nan")])
    def test_invalid_norm_raises(self, norm):
        module = _make_linear()
        with pytest.raises(ValueError, match="norm must be a finite positive number"):
            _make_observer(module, strategy="channel", norm=norm)

    def test_strict_raises_on_tensor_strategy(self):
        module = _make_linear()
        args = QuantizationArgs(
            num_bits=4,
            symmetric=True,
            strategy="tensor",
            observer="imatrix_mse",
            observer_kwargs={"strict": True},
        )
        observer = Observer.load_from_registry(
            "imatrix_mse", base_name="weight", args=args
        )
        observer.attach(module)
        with pytest.raises(NotImplementedError, match="TENSOR strategy"):
            observer(module.weight).get_qparams()


# ---------------------------------------------------------------------------
# Hook disabled during HooksMixin.disable_hooks()
# ---------------------------------------------------------------------------


class TestStatisticsCleanup:
    """Verify that imatrix stats are cleaned up after get_qparams."""

    def test_get_qparams_deletes_imatrix_stats(self):
        module = _make_linear(in_features=8, out_features=4)
        observer = _make_observer(
            module, strategy="channel", importance=_make_importance(in_features=8)
        )
        observer(module.weight)

        assert hasattr(observer, "_imatrix_sum")
        assert hasattr(observer, "_imatrix_count")
        assert hasattr(observer, "min_vals")
        assert hasattr(observer, "max_vals")

        qparams = observer.get_qparams()
        assert torch.isfinite(qparams["scale"]).all()

        assert not hasattr(observer, "_imatrix_sum")
        assert not hasattr(observer, "_imatrix_count")
        assert not hasattr(observer, "min_vals")
        assert not hasattr(observer, "max_vals")

    def test_get_qparams_deletes_imatrix_stats_group_strategy(self):
        module = _make_linear(in_features=8, out_features=4)
        observer = _make_observer(
            module,
            strategy="group",
            group_size=4,
            importance=_make_importance(in_features=8),
        )
        observer(module.weight)
        observer.get_qparams()

        assert not hasattr(observer, "_imatrix_sum")
        assert not hasattr(observer, "_imatrix_count")
        assert not hasattr(observer, "min_vals")
        assert not hasattr(observer, "max_vals")

    def test_has_statistics_false_after_cleanup(self):
        module = _make_linear(in_features=8, out_features=4)
        observer = _make_observer(
            module, strategy="channel", importance=_make_importance(in_features=8)
        )
        observer(module.weight)
        assert observer.has_statistics

        observer.get_qparams()
        assert not observer.has_statistics


# ---------------------------------------------------------------------------
# Hook disabled during HooksMixin.disable_hooks()
# ---------------------------------------------------------------------------


class TestHookDisabling:
    """
    The iMatrix hook must not accumulate when HooksMixin.disable_hooks() is active.
    """

    def test_hook_skipped_under_disable_hooks(self):
        module = torch.nn.Linear(8, 4)
        observer = _make_observer(module, strategy="channel")

        x = torch.randn(2, 8)
        module(x)
        assert observer._imatrix_count.item() > 0
        count_before = observer._imatrix_count.item()

        with HooksMixin.disable_hooks():
            module(torch.randn(2, 8))

        assert observer._imatrix_count.item() == count_before

    def test_hook_resumes_after_disable_hooks(self):
        module = torch.nn.Linear(8, 4)
        observer = _make_observer(module, strategy="channel")

        with HooksMixin.disable_hooks():
            module(torch.randn(2, 8))
        assert observer._imatrix_count.item() == 0

        module(torch.randn(2, 8))
        assert observer._imatrix_count.item() > 0


@pytest.mark.parametrize(
    "quant_type,strategy",
    [
        ("int", QuantizationStrategy.GROUP),
        ("int", QuantizationStrategy.TENSOR_GROUP),
        ("float", QuantizationStrategy.GROUP),
        ("float", QuantizationStrategy.TENSOR_GROUP),
    ],
)
@pytest.mark.skipif(not torch.accelerator.is_available(), reason="requires CUDA Triton")
def test_imatrix_triton_matches_eager_when_tile_fits_group(quant_type, strategy):
    """A full buffer preserves eager choices when an imatrix group fits one tile."""
    args = QuantizationArgs(
        num_bits=4,
        type=quant_type,
        symmetric=True,
        strategy=strategy,
        group_size=512,
    )
    torch.manual_seed(0)
    observed = flatten_for_calibration(
        torch.randn(8, 1024, device="cuda"), "weight", args
    )
    importance = flatten_for_calibration(
        torch.linspace(0.2, 2.0, 1024, device="cuda").unsqueeze(0).expand(8, -1),
        "weight",
        args,
    )
    search_args = (observed, args, 0.5, 100, 100.0, 2.4)
    search_kwargs = {
        "importance_weights": importance,
        "triton_error_buffer": 1.0,
    }

    eager = ImplBackend.call("_grid_search_observer", *search_args, **search_kwargs)
    triton = ImplBackend.call(
        "_grid_search_observer_triton", *search_args, **search_kwargs
    )

    assert torch.equal(eager[0], triton[0])
    assert torch.equal(eager[1], triton[1])


@pytest.mark.skipif(not torch.accelerator.is_available(), reason="requires CUDA Triton")
def test_imatrix_bfloat16_triton_matches_eager_for_packed_group():
    """BF16 weighted errors compile and preserve choices in the packed kernel."""
    args = QuantizationArgs(
        num_bits=4,
        type="int",
        symmetric=True,
        strategy=QuantizationStrategy.GROUP,
        group_size=128,
    )
    torch.manual_seed(0)
    observed = flatten_for_calibration(
        torch.randn(16, 128, device="cuda", dtype=torch.bfloat16), "weight", args
    )
    importance = flatten_for_calibration(
        torch.linspace(0.2, 2.0, 128, device="cuda").expand(16, -1),
        "weight",
        args,
    )
    search_args = (observed, args, 0.95, 5, 20.0, 3.0)
    search_kwargs = {
        "importance_weights": importance,
        "triton_error_buffer": 1.0,
    }

    eager = ImplBackend.call("_grid_search_observer", *search_args, **search_kwargs)
    triton = ImplBackend.call(
        "_grid_search_observer_triton", *search_args, **search_kwargs
    )

    assert torch.equal(eager[0], triton[0])
    assert torch.equal(eager[1], triton[1])


@pytest.mark.skipif(not torch.accelerator.is_available(), reason="requires CUDA Triton")
def test_imatrix_triton_matches_eager_for_packed_nvfp4_groups():
    """NVFP4A16 expanded iMatrix search preserves eager ranges and QDQ exactly."""
    args = preset_name_to_scheme("NVFP4A16", ["Linear"]).weights
    torch.manual_seed(0)
    observed = flatten_for_calibration(
        torch.randn(8, 1024, device="cuda", dtype=torch.bfloat16), "weight", args
    )
    importance = flatten_for_calibration(
        torch.linspace(0.2, 2.0, 1024, device="cuda").unsqueeze(0).expand(8, -1),
        "weight",
        args,
    )
    search_args = (
        observed,
        args,
        1.0 - 0.8 / 1.8,
        1000,
        200.0,
        3.0,
    )
    search_kwargs = {
        "expand": 1.8,
        "importance_weights": importance,
        "triton_error_buffer": 1.0,
    }

    eager = ImplBackend.call("_grid_search_observer", *search_args, **search_kwargs)
    triton = ImplBackend.call(
        "_grid_search_observer_triton", *search_args, **search_kwargs
    )

    assert torch.equal(eager[0], triton[0])
    assert torch.equal(eager[1], triton[1])

    scales_and_zps = [
        calculate_qparams(
            min_vals=bounds[0],
            max_vals=bounds[1],
            quantization_args=args,
            global_scale=None,
        )
        for bounds in (eager, triton)
    ]
    assert torch.equal(scales_and_zps[0][0], scales_and_zps[1][0])
    assert torch.equal(scales_and_zps[0][1], scales_and_zps[1][1])

    token_args = args.model_copy(update={"strategy": QuantizationStrategy.TOKEN})
    qdq_weights = [
        fake_quantize(
            observed,
            scales.unsqueeze(-1),
            zero_points.unsqueeze(-1),
            token_args,
        ).to(observed.dtype)
        for scales, zero_points in scales_and_zps
    ]
    assert torch.equal(qdq_weights[0], qdq_weights[1])


@pytest.mark.skipif(not torch.accelerator.is_available(), reason="requires CUDA Triton")
def test_imatrix_nvfp4_bf16_reduction_tie_matches_eager():
    """A Llama-derived BF16 near-tie chooses the same NVFP4 grid point."""
    args = preset_name_to_scheme("NVFP4A16", ["Linear"]).weights
    observed = torch.tensor(
        [
            0.0078125,
            0.00439453125,
            -0.00092315673828125,
            0.006439208984375,
            0.0078125,
            0.0130615234375,
            0.0067138671875,
            0.004974365234375,
            -0.00567626953125,
            -0.0147705078125,
            -0.00185394287109375,
            0.00130462646484375,
            -0.0057373046875,
            0.01251220703125,
            0.0133056640625,
            -0.00174713134765625,
        ],
        device="cuda",
        dtype=torch.bfloat16,
    ).reshape(1, 1, 1, 16)
    importance = torch.tensor(
        [
            0.0008253254927694798,
            0.002562405075877905,
            0.0028008718509227037,
            0.001142465160228312,
            0.001915410510264337,
            0.0009209896088577807,
            0.0016524026868864894,
            0.002158285351470113,
            0.0010275698732584715,
            0.0035809692926704884,
            0.00206969166174531,
            0.0017336469609290361,
            0.0008845608099363744,
            0.0036817658692598343,
            0.007513321004807949,
            0.0026003867387771606,
        ],
        device="cuda",
        dtype=torch.float32,
    ).reshape(1, 1, 1, 16)
    search_args = (observed, args, 1.0 - 0.8 / 1.8, 1000, 200.0, 3.0)
    search_kwargs = {
        "expand": 1.8,
        "importance_weights": importance,
        "triton_error_buffer": 1.0,
    }

    eager = ImplBackend.call("_grid_search_observer", *search_args, **search_kwargs)
    triton = ImplBackend.call(
        "_grid_search_observer_triton", *search_args, **search_kwargs
    )
    assert torch.equal(eager[0], triton[0])
    assert torch.equal(eager[1], triton[1])

    qparams = [
        calculate_qparams(
            min_vals=bounds[0],
            max_vals=bounds[1],
            quantization_args=args,
            global_scale=None,
        )
        for bounds in (eager, triton)
    ]
    assert torch.equal(qparams[0][0], qparams[1][0])
    assert torch.equal(qparams[0][1], qparams[1][1])

    token_args = args.model_copy(update={"strategy": QuantizationStrategy.TOKEN})
    qdq_weights = [
        fake_quantize(
            observed,
            scales.unsqueeze(-1),
            zero_points.unsqueeze(-1),
            token_args,
        ).to(observed.dtype)
        for scales, zero_points in qparams
    ]
    assert torch.equal(qdq_weights[0], qdq_weights[1])


@pytest.mark.skipif(not torch.accelerator.is_available(), reason="requires CUDA Triton")
def test_imatrix_nvfp4_bf16_power_rounding_tie_matches_eager():
    """Match eager when a weighted cubic error lands on a BF16 rounding tie."""
    args = preset_name_to_scheme("NVFP4A16", ["Linear"]).weights
    observed = torch.tensor(
        [
            -0.013671875,
            0.01007080078125,
            -0.001068115234375,
            -0.0050048828125,
            -0.01275634765625,
            -0.005645751953125,
            -0.0169677734375,
            0.000766754150390625,
            -0.0045166015625,
            0.012939453125,
            0.004547119140625,
            -5.340576171875e-05,
            -0.0036163330078125,
            -0.002349853515625,
            0.006927490234375,
            0.01318359375,
        ],
        device="cuda",
        dtype=torch.bfloat16,
    ).reshape(1, 1, 1, 16)
    importance = torch.tensor(
        [
            0.002571985125541687,
            0.0014283129712566733,
            0.0010594911873340607,
            0.005643679294735193,
            0.002667697612196207,
            0.0023365672677755356,
            0.0015463161980733275,
            0.010683584958314896,
            0.0014458026271313429,
            0.0011119716800749302,
            0.0010083864908665419,
            14119.0341796875,
            0.0009728227159939706,
            0.00145068415440619,
            0.010342647321522236,
            0.0010723379673436284,
        ],
        device="cuda",
        dtype=torch.float32,
    ).reshape(1, 1, 1, 16)
    search_args = (observed, args, 1.0 - 0.8 / 1.8, 1000, 200.0, 3.0)
    search_kwargs = {
        "expand": 1.8,
        "importance_weights": importance,
        "triton_error_buffer": 1.0,
    }

    eager = ImplBackend.call("_grid_search_observer", *search_args, **search_kwargs)
    triton = ImplBackend.call(
        "_grid_search_observer_triton", *search_args, **search_kwargs
    )
    assert torch.equal(eager[0], triton[0])
    assert torch.equal(eager[1], triton[1])

    qparams = [
        calculate_qparams(
            min_vals=bounds[0],
            max_vals=bounds[1],
            quantization_args=args,
            global_scale=None,
        )
        for bounds in (eager, triton)
    ]
    assert torch.equal(qparams[0][0], qparams[1][0])
    assert torch.equal(qparams[0][1], qparams[1][1])

    token_args = args.model_copy(update={"strategy": QuantizationStrategy.TOKEN})
    qdq_weights = [
        fake_quantize(
            observed,
            scales.unsqueeze(-1),
            zero_points.unsqueeze(-1),
            token_args,
        ).to(observed.dtype)
        for scales, zero_points in qparams
    ]
    assert torch.equal(qdq_weights[0], qdq_weights[1])
