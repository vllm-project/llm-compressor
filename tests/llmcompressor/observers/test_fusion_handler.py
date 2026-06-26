import pytest
import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy
from torch import nn

from llmcompressor.observers import Observer
from llmcompressor.observers.fusion import FusionHandler


def _make_observer(strategy=QuantizationStrategy.TENSOR_GROUP, group_size=16):
    args = QuantizationArgs(
        num_bits=4,
        strategy=strategy,
        group_size=group_size,
    )
    return Observer.load_from_registry(
        "memoryless_minmax", base_name="weight", args=args
    )


def _make_linear(in_f=64, out_f=64):
    return nn.Linear(in_f, out_f, bias=False)


def _make_fused_group(n=3):
    observers = [_make_observer() for _ in range(n)]
    modules = [_make_linear() for _ in range(n)]
    pairs = list(zip(observers, modules))
    FusionHandler.fuse(pairs)
    return observers, modules


@pytest.mark.unit
class TestFusionHandlerInit:
    def test_unfused_handler_not_fused(self):
        obs = _make_observer()
        assert not obs.fusion_handler.is_fused

    def test_unfused_handler_module_is_none(self):
        obs = _make_observer()
        assert obs.fusion_handler.module is None


@pytest.mark.unit
class TestFusionHandlerFuse:
    def test_fuse_links_handlers(self):
        observers, modules = _make_fused_group(3)
        for obs in observers:
            assert obs.fusion_handler.is_fused
            assert len(obs.fusion_handler._group) == 2

    def test_fuse_sets_module_refs(self):
        observers, modules = _make_fused_group(3)
        for obs, mod in zip(observers, modules):
            assert obs.fusion_handler.module is mod

    def test_fuse_via_classmethod(self):
        observers = [_make_observer() for _ in range(2)]
        modules = [_make_linear() for _ in range(2)]
        FusionHandler.fuse(zip(observers, modules))
        for obs in observers:
            assert obs.fusion_handler.is_fused


@pytest.mark.unit
class TestGetStatistics:
    def test_get_statistics_returns_dict(self):
        obs = _make_observer()
        mod = _make_linear()
        obs(mod.weight)
        stats = obs.get_statistics()
        assert "min_vals" in stats
        assert "max_vals" in stats
        assert isinstance(stats["min_vals"], torch.Tensor)
        assert isinstance(stats["max_vals"], torch.Tensor)

    def test_get_statistics_raises_without_observation(self):
        obs = _make_observer()
        with pytest.raises(AssertionError, match="No statistics"):
            obs.get_statistics()


@pytest.mark.unit
class TestGetFusedStatistics:
    def test_all_observed(self):
        observers, modules = _make_fused_group(3)
        for obs, mod in zip(observers, modules):
            obs(mod.weight)
        stats = observers[0].fusion_handler.get_fused_statistics()
        assert len(stats) == 3
        for s in stats:
            assert "min_vals" in s
            assert "max_vals" in s

    def test_auto_observes_missing(self):
        observers, modules = _make_fused_group(3)
        observers[0](modules[0].weight)
        stats = observers[0].fusion_handler.get_fused_statistics()
        assert len(stats) == 3
        for obs in observers:
            assert obs.has_statistics

    def test_unfused_returns_self_only(self):
        obs = _make_observer()
        mod = _make_linear()
        obs(mod.weight)
        stats = obs.fusion_handler.get_fused_statistics()
        assert len(stats) == 1


@pytest.mark.unit
class TestMaybeDeleteStatistics:
    def test_unfused_deletes_immediately(self):
        obs = _make_observer()
        mod = _make_linear()
        obs(mod.weight)
        assert obs.has_statistics
        obs.fusion_handler.maybe_delete_statistics()
        assert not obs.has_statistics

    def test_fused_partial_does_not_delete(self):
        observers, modules = _make_fused_group(3)
        for obs, mod in zip(observers, modules):
            obs(mod.weight)
        observers[0].fusion_handler.maybe_delete_statistics()
        for obs in observers:
            assert obs.has_statistics

    def test_fused_all_called_deletes(self):
        observers, modules = _make_fused_group(3)
        for obs, mod in zip(observers, modules):
            obs(mod.weight)
        for obs in observers:
            obs.fusion_handler.maybe_delete_statistics()
        for obs in observers:
            assert not obs.has_statistics

    def test_resets_after_deletion(self):
        observers, modules = _make_fused_group(3)
        for obs, mod in zip(observers, modules):
            obs(mod.weight)
        for obs in observers:
            obs.fusion_handler.maybe_delete_statistics()
        # re-observe and verify cycle works again
        for obs, mod in zip(observers, modules):
            obs(mod.weight)
        observers[0].fusion_handler.maybe_delete_statistics()
        for obs in observers:
            assert obs.has_statistics


@pytest.mark.unit
class TestCalculateGlobalScale:
    def test_non_tensor_group_returns_none(self):
        args = QuantizationArgs(
            num_bits=8,
            strategy=QuantizationStrategy.TENSOR,
        )
        obs = Observer.load_from_registry(
            "memoryless_minmax", base_name="weight", args=args
        )
        mod = _make_linear()
        obs(mod.weight)
        assert obs.get_global_scale() is None

    def test_tensor_group_returns_scale(self):
        observers, modules = _make_fused_group(3)
        for obs, mod in zip(observers, modules):
            obs(mod.weight)
        scale = observers[0].get_global_scale()
        assert scale is not None
        assert isinstance(scale, torch.Tensor)

    def test_global_scale_consistent_across_group(self):
        observers, modules = _make_fused_group(3)
        for obs, mod in zip(observers, modules):
            obs(mod.weight)
        scales = [obs.get_global_scale() for obs in observers]
        for s in scales[1:]:
            torch.testing.assert_close(s, scales[0])


@pytest.mark.unit
class TestEagerFusedObservation:
    def test_forward_triggers_fused_observation(self):
        observers, modules = _make_fused_group(3)
        observers[0](modules[0].weight)
        for obs in observers:
            assert obs.has_statistics


@pytest.mark.unit
class TestNVFP4FusedLifecycle:
    """End-to-end test simulating the NVFP4 fused module flow:
    fuse -> observe -> get_qparams (global_scale + cooperative deletion)."""

    def test_fused_qparams_share_global_scale(self):
        observers, modules = _make_fused_group(3)
        for obs, mod in zip(observers, modules):
            obs(mod.weight)

        qparams = [obs.get_qparams() for obs in observers]
        for qp in qparams:
            assert qp["global_scale"] is not None
        for qp in qparams[1:]:
            torch.testing.assert_close(qp["global_scale"], qparams[0]["global_scale"])

    def test_stats_deleted_after_all_get_qparams(self):
        observers, modules = _make_fused_group(3)
        for obs, mod in zip(observers, modules):
            obs(mod.weight)

        # first two calls should NOT delete stats (group not fully consumed)
        observers[0].get_qparams()
        assert observers[1].has_statistics
        observers[1].get_qparams()
        assert observers[2].has_statistics

        # third call completes the group — all stats deleted
        observers[2].get_qparams()
        for obs in observers:
            assert not obs.has_statistics

    def test_unfused_stats_deleted_immediately_after_get_qparams(self):
        obs = _make_observer()
        mod = _make_linear()
        obs(mod.weight)
        obs.get_qparams()
        assert not obs.has_statistics

    def test_can_re_observe_after_deletion(self):
        observers, modules = _make_fused_group(3)
        for obs, mod in zip(observers, modules):
            obs(mod.weight)
        for obs in observers:
            obs.get_qparams()
        # stats are deleted, re-observe
        for obs, mod in zip(observers, modules):
            obs(mod.weight)
        for obs in observers:
            assert obs.has_statistics
        qparams = [obs.get_qparams() for obs in observers]
        for qp in qparams:
            assert qp["global_scale"] is not None
