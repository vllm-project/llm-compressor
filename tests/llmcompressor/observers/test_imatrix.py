import pytest
import torch
from compressed_tensors.quantization.quant_args import QuantizationArgs

from llmcompressor.observers.base import Observer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _set_importance(module, importance):
    """Set raw imatrix accumulators so that importance = sum / count."""
    module._imatrix_sum = importance.clone()
    module._imatrix_count = torch.tensor(1, dtype=torch.int64)


def _make_linear_with_importance(in_features=8, out_features=4, seed=42):
    """Create a Linear module with non-uniform imatrix importance."""
    torch.manual_seed(seed)
    module = torch.nn.Linear(in_features, out_features)
    importance = torch.ones(in_features)
    importance[: in_features // 2] = 10.0
    _set_importance(module, importance)
    return module


def _make_observer(module, strategy="channel", group_size=None, **kwargs):
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
    return observer


# ---------------------------------------------------------------------------
# Bug 1: get_global_min_max must not crash on TENSOR_GROUP
# ---------------------------------------------------------------------------


class TestGlobalMinMaxTensorGroup:
    """Regression test for bug #1: global_scale path with TENSOR_GROUP."""

    def test_global_scale_tensor_group_does_not_crash(self):
        """get_global_scale must complete without error on TENSOR_GROUP."""
        module = _make_linear_with_importance(in_features=8, out_features=4)
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

        global_scale = observer(module.weight).get_qparams()["global_scale"]
        assert global_scale is not None
        assert global_scale.shape == (1,)
        assert torch.isfinite(global_scale).all()

    def test_global_scale_then_forward_tensor_group(self):
        """Full flow: global_scale -> forward must produce valid qparams."""
        module = _make_linear_with_importance(in_features=8, out_features=4)
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

        qparams = observer(module.weight).get_qparams()
        assert torch.isfinite(qparams["scale"]).all()


# ---------------------------------------------------------------------------
# Basic functionality (sanity checks)
# ---------------------------------------------------------------------------


class TestBasicFunctionality:
    """Sanity checks for the happy path."""

    def test_channel_strategy(self):
        module = _make_linear_with_importance(in_features=8, out_features=4)
        observer = _make_observer(module, strategy="channel")
        qparams = observer(module.weight).get_qparams()
        assert qparams["scale"].shape == (4, 1)
        assert torch.isfinite(qparams["scale"]).all()

    def test_group_strategy(self):
        module = _make_linear_with_importance(in_features=8, out_features=4)
        observer = _make_observer(module, strategy="group", group_size=4)
        qparams = observer(module.weight).get_qparams()
        assert torch.isfinite(qparams["scale"]).all()

    def test_tensor_group_strategy(self):
        module = _make_linear_with_importance(in_features=8, out_features=4)
        observer = _make_observer(module, strategy="tensor_group", group_size=4)
        qparams = observer(module.weight).get_qparams()
        assert torch.isfinite(qparams["scale"]).all()

    def test_block_strategy(self):
        module = _make_linear_with_importance(in_features=8, out_features=4)
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
        _set_importance(module_weighted, importance)

        obs_w = _make_observer(module_weighted, strategy="channel")
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
        _set_importance(module_imatrix, torch.ones(8))

        args_uniform = QuantizationArgs(
            num_bits=4,
            symmetric=True,
            strategy="channel",
            observer="memoryless_mse",
            observer_kwargs={"grid": 20},
        )
        obs_imatrix = _make_observer(
            module_imatrix, strategy="channel", norm=2.4, maxshrink=0.20
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
        _set_importance(module, torch.ones(5))  # wrong size
        observer = _make_observer(module, strategy="channel", strict=True)
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
        _set_importance(module, importance)
        observer = _make_observer(module, strategy="channel", strict=True)
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
        _set_importance(module_imatrix, importance)

        args_uniform = QuantizationArgs(
            num_bits=4,
            symmetric=True,
            strategy="channel",
            observer="memoryless_mse",
            observer_kwargs={"grid": 20},
        )
        obs_imatrix = _make_observer(module_imatrix, strategy="channel", strict=False)
        obs_uniform = Observer.load_from_registry(
            "memoryless_mse", base_name="weight", args=args_uniform
        )

        qparams_i = obs_imatrix(module_imatrix.weight).get_qparams()
        qparams_u = obs_uniform(module_mse.weight).get_qparams()

        assert torch.allclose(qparams_i["scale"], qparams_u["scale"])
        assert torch.equal(qparams_i["zero_point"], qparams_u["zero_point"])

    @pytest.mark.parametrize("norm", [0, -1, float("inf"), float("nan")])
    def test_invalid_norm_raises(self, norm):
        module = _make_linear_with_importance()
        with pytest.raises(ValueError, match="norm must be a finite positive number"):
            _make_observer(module, strategy="channel", norm=norm)

    def test_strict_raises_on_tensor_strategy(self):
        module = _make_linear_with_importance()
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
