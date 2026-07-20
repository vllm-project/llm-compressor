from unittest.mock import MagicMock, patch

import pytest
import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy
from torch import nn

from llmcompressor.modifiers.quantization.calibration import observe, update_qparams
from llmcompressor.modifiers.quantization.quantization import QuantizationModifier
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


def _make_module_with_observer(strategy=QuantizationStrategy.TENSOR, group_size=None):
    """Create a Linear module with a weight observer registered as a submodule."""
    mod = _make_linear()
    kwargs = {"num_bits": 8, "strategy": strategy}
    if group_size is not None:
        kwargs["group_size"] = group_size
    args = QuantizationArgs(**kwargs)
    obs = Observer.load_from_registry(
        "memoryless_minmax", base_name="weight", args=args
    )
    mod.register_module("weight_observer", obs)
    return mod


def _make_fused_module_group(n=3):
    """Create n Linear modules with fused TENSOR_GROUP weight observers."""
    modules = []
    observers = []
    for _ in range(n):
        mod = _make_linear()
        obs = _make_observer()
        mod.register_module("weight_observer", obs)
        modules.append(mod)
        observers.append(obs)
    FusionHandler.fuse(zip(observers, modules))
    return modules


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


def _mock_broadcast(*args, **kwargs):
    """Stand-in for dist.broadcast that returns a mock async work handle."""
    return MagicMock()


def _simulate_ddp_epoch(modifier, modules, rank_to_modules, module_to_rank):
    """Simulate the distributed branch of on_sequential_epoch_end.

    1. observe + update_qparams on this rank's subset (rank 0)
    2. call _broadcast_qparam_onloads on the full module list
    """
    observe(rank_to_modules[0], "weight")
    update_qparams(rank_to_modules[0], "weight")

    with (
        patch("llmcompressor.modifiers.quantization.quantization.base.dist.broadcast",
              side_effect=_mock_broadcast),
        patch("llmcompressor.modifiers.quantization.quantization.base.wait_for_comms"),
    ):
        modifier._broadcast_qparam_onloads(modules, module_to_rank)


@pytest.mark.unit
class TestDDPObserverCleanup:
    """Regression tests for DDP observer stats leak.

    In DDP, greedy_bin_packing assigns modules to ranks and only the owning
    rank calls update_qparams (which triggers get_qparams -> delete_statistics).
    Modules assigned to other ranks kept stats forever, leaking memory per layer.
    The fix adds a cleanup loop inside _broadcast_qparam_onloads that calls
    delete_statistics(check_fused=True) on all modules.

    These tests patch dist.broadcast and call _broadcast_qparam_onloads directly
    to simulate the DDP flow without requiring multiple processes.
    """

    def test_broadcast_cleans_up_unprocessed_modules(self):
        """_broadcast_qparam_onloads deletes stats on modules not owned by this rank."""
        modules = [_make_module_with_observer() for _ in range(4)]
        observe(modules, "weight")

        # simulate rank assignment: rank 0 owns first 2, rank 1 owns last 2
        rank_to_modules = {0: modules[:2], 1: modules[2:]}
        module_to_rank = {m: 0 for m in modules[:2]}
        module_to_rank.update({m: 1 for m in modules[2:]})

        modifier = QuantizationModifier(targets=[])
        _simulate_ddp_epoch(modifier, modules, rank_to_modules, module_to_rank)

        for mod in modules:
            assert not mod.weight_observer.has_statistics

    def test_fused_cleanup_with_split_ranks(self):
        """Fused observers (NVFP4 Q/K/V) split across ranks.

        rank 0 gets q_proj, rank 1 gets k_proj and v_proj.
        _broadcast_qparam_onloads cooperative deletion cleans up everything.
        """
        modules = _make_fused_module_group(n=3)
        observe(modules, "weight")

        # q on rank 0, k/v on rank 1
        rank_to_modules = {0: modules[:1], 1: modules[1:]}
        module_to_rank = {modules[0]: 0, modules[1]: 1, modules[2]: 1}

        modifier = QuantizationModifier(targets=[])
        _simulate_ddp_epoch(modifier, modules, rank_to_modules, module_to_rank)

        for mod in modules:
            assert not mod.weight_observer.has_statistics

    def test_repeated_layers_no_accumulation(self):
        """Stats from previous layers never pile up across sequential layers."""
        modifier = QuantizationModifier(targets=[])

        all_layer_modules = []
        for _ in range(5):
            layer = [_make_module_with_observer() for _ in range(4)]
            all_layer_modules.append(layer)

        for layer_idx, modules in enumerate(all_layer_modules):
            observe(modules, "weight")

            rank_to_modules = {0: modules[:2], 1: modules[2:]}
            module_to_rank = {m: 0 for m in modules[:2]}
            module_to_rank.update({m: 1 for m in modules[2:]})

            _simulate_ddp_epoch(modifier, modules, rank_to_modules, module_to_rank)

            for mod in modules:
                assert not mod.weight_observer.has_statistics, (
                    f"Layer {layer_idx}: observer stats not cleaned up"
                )
            for prev_idx in range(layer_idx):
                for mod in all_layer_modules[prev_idx]:
                    assert not mod.weight_observer.has_statistics, (
                        f"Layer {prev_idx} stats leaked into layer {layer_idx}"
                    )
