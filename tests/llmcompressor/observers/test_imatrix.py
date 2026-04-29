import pytest
import torch
from compressed_tensors.quantization.quant_args import QuantizationArgs

from llmcompressor.observers.base import Observer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_linear_with_importance(in_features=8, out_features=4, seed=42):
    """Create a Linear module with non-uniform _imatrix_importance."""
    torch.manual_seed(seed)
    module = torch.nn.Linear(in_features, out_features)
    # Non-uniform importance: channels 0-3 are 10x more important
    importance = torch.ones(in_features)
    importance[: in_features // 2] = 10.0
    module._imatrix_importance = importance
    return module


def _make_observer(module, strategy="channel", group_size=None, **kwargs):
    """Create an imatrix_mse observer for a given module."""
    args = QuantizationArgs(
        num_bits=4,
        symmetric=True,
        strategy=strategy,
        group_size=group_size,
        observer="imatrix_mse",
        observer_kwargs=kwargs,
    )
    return Observer.load_from_registry(
        "imatrix_mse", base_name="weight", args=args, module=module
    )


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
            "imatrix_mse", base_name="weight", args=args, module=module
        )

        # This used to crash because _get_global_scale_with_minmax reshapes
        # to (1, 1, -1), breaking importance broadcasting.
        global_scale = observer.get_global_scale(module.weight)
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
            "imatrix_mse", base_name="weight", args=args, module=module
        )

        global_scale = observer.get_global_scale(module.weight)
        module.weight_global_scale = global_scale
        scale, zp = observer(module.weight)
        assert torch.isfinite(scale).all()


# ---------------------------------------------------------------------------
# Bug 2: importance must be reordered with g_idx (actorder)
# ---------------------------------------------------------------------------


class TestActorderReordering:
    """Regression test for bug #2: g_idx alignment."""

    def test_reorder_importance_with_g_idx(self):
        """With g_idx set, importance must be reordered by argsort(g_idx)."""
        in_features = 8
        module = _make_linear_with_importance(in_features=in_features)

        # Simulate actorder: g_idx assigns columns to groups in non-trivial order
        g_idx = torch.tensor([1, 1, 0, 0, 1, 1, 0, 0])
        module.weight_g_idx = g_idx

        observer = _make_observer(module, strategy="group", group_size=4)

        # Call through the observer — if reordering works, it should not crash
        # and should produce valid scales
        scale, zp = observer(module.weight)
        assert torch.isfinite(scale).all()

    def test_no_g_idx_still_works(self):
        """Without g_idx, observer must work normally."""
        module = _make_linear_with_importance(in_features=8)
        observer = _make_observer(module, strategy="group", group_size=4)
        scale, zp = observer(module.weight)
        assert torch.isfinite(scale).all()


# ---------------------------------------------------------------------------
# Bug 3: weight-only guard
# ---------------------------------------------------------------------------


class TestWeightOnlyGuard:
    """Regression test for bug #3: base_name != 'weight' must be rejected."""

    def test_non_weight_base_name_strict_raises(self):
        """strict=True must raise NotImplementedError for non-weight."""
        module = _make_linear_with_importance()
        args = QuantizationArgs(
            num_bits=8,
            symmetric=True,
            strategy="channel",
            observer="imatrix_mse",
            observer_kwargs={"strict": True},
        )
        observer = Observer.load_from_registry(
            "imatrix_mse", base_name="input", args=args, module=module
        )
        observed = torch.randn(2, 1, 1, 8)
        with pytest.raises(NotImplementedError, match="weight observers"):
            observer.get_min_max(observed)

    def test_non_weight_base_name_non_strict_falls_back(self):
        """strict=False must fall back to uniform MSE (no crash)."""
        module = _make_linear_with_importance()
        args = QuantizationArgs(
            num_bits=8,
            symmetric=True,
            strategy="channel",
            observer="imatrix_mse",
            observer_kwargs={"strict": False},
        )
        observer = Observer.load_from_registry(
            "imatrix_mse", base_name="input", args=args, module=module
        )
        observed = torch.randn(2, 1, 1, 8)
        min_val, max_val = observer.get_min_max(observed)
        assert torch.isfinite(min_val).all()
        assert torch.isfinite(max_val).all()


# ---------------------------------------------------------------------------
# Basic functionality (sanity checks)
# ---------------------------------------------------------------------------


class TestBasicFunctionality:
    """Sanity checks for the happy path."""

    def test_channel_strategy(self):
        module = _make_linear_with_importance(in_features=8, out_features=4)
        observer = _make_observer(module, strategy="channel")
        scale, zp = observer(module.weight)
        assert scale.shape == (4, 1)
        assert torch.isfinite(scale).all()

    def test_group_strategy(self):
        module = _make_linear_with_importance(in_features=8, out_features=4)
        observer = _make_observer(module, strategy="group", group_size=4)
        scale, zp = observer(module.weight)
        assert torch.isfinite(scale).all()

    def test_tensor_group_strategy(self):
        module = _make_linear_with_importance(in_features=8, out_features=4)
        observer = _make_observer(module, strategy="tensor_group", group_size=4)
        global_scale = observer.get_global_scale(module.weight)
        module.weight_global_scale = global_scale
        scale, zp = observer(module.weight)
        assert torch.isfinite(scale).all()

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
            "imatrix_mse", base_name="weight", args=args, module=module
        )
        scale, zp = observer(module.weight)
        assert torch.isfinite(scale).all()

    def test_no_importance_falls_back(self):
        """Observer without _imatrix_importance must fall back gracefully."""
        module = torch.nn.Linear(8, 4)
        observer = _make_observer(module, strategy="channel")
        scale, zp = observer(module.weight)
        assert torch.isfinite(scale).all()

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
        module_weighted._imatrix_importance = importance

        obs_w = _make_observer(module_weighted, strategy="channel")
        obs_u = _make_observer(module_uniform, strategy="channel")

        scale_w, _ = obs_w(module_weighted.weight)
        scale_u, _ = obs_u(module_uniform.weight)

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
        module_imatrix._imatrix_importance = torch.ones(8)

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
            "memoryless_mse", base_name="weight", args=args_uniform, module=module_mse
        )

        scale_i, zp_i = obs_imatrix(module_imatrix.weight)
        scale_u, zp_u = obs_uniform(module_mse.weight)

        assert torch.allclose(scale_i, scale_u)
        assert torch.equal(zp_i, zp_u)


# ---------------------------------------------------------------------------
# Validation edge cases
# ---------------------------------------------------------------------------


class TestValidation:
    def test_strict_raises_on_missing_importance(self):
        module = torch.nn.Linear(8, 4)
        observer = _make_observer(module, strategy="channel", strict=True)
        with pytest.raises(ValueError, match="imatrix_importance"):
            observer(module.weight)

    def test_strict_raises_on_wrong_size(self):
        module = torch.nn.Linear(8, 4)
        module._imatrix_importance = torch.ones(5)  # wrong size
        observer = _make_observer(module, strategy="channel", strict=True)
        with pytest.raises(ValueError, match="size mismatch"):
            observer(module.weight)

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
        module._imatrix_importance = importance
        observer = _make_observer(module, strategy="channel", strict=True)
        with pytest.raises(ValueError, match=match):
            observer(module.weight)

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
        module_imatrix._imatrix_importance = importance

        args_uniform = QuantizationArgs(
            num_bits=4,
            symmetric=True,
            strategy="channel",
            observer="memoryless_mse",
            observer_kwargs={"grid": 20},
        )
        obs_imatrix = _make_observer(module_imatrix, strategy="channel", strict=False)
        obs_uniform = Observer.load_from_registry(
            "memoryless_mse", base_name="weight", args=args_uniform, module=module_mse
        )

        scale_i, zp_i = obs_imatrix(module_imatrix.weight)
        scale_u, zp_u = obs_uniform(module_mse.weight)

        assert torch.allclose(scale_i, scale_u)
        assert torch.equal(zp_i, zp_u)

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
            "imatrix_mse", base_name="weight", args=args, module=module
        )
        with pytest.raises(NotImplementedError, match="TENSOR strategy"):
            observer(module.weight)
