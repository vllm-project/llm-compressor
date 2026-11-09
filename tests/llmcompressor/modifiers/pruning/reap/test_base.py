"""
Unit tests for the REAP pruning modifier.

The shared pruning lifecycle is tested in
``tests/llmcompressor/modifiers/pruning/test_base.py``; the tests here cover
REAP's own behavior: the router-weighted expert activation saliency
``S_j = mean(g_j * ||f_j||_2)``, which must agree between the sparse and
calibrate-all execution paths; full-model pruning with real routing; and
distributed calibration with unevenly distributed data.
"""

import pytest
import torch
from torch import distributed as dist

from llmcompressor.core import Event, EventType
from llmcompressor.modeling.moe import context as moe_context
from llmcompressor.modeling.moe.helpers import MoEConfig
from llmcompressor.modeling.moe.linear_experts import ExpertMLP
from llmcompressor.modifiers.pruning.reap import REAPPruningModifier
from tests.llmcompressor.modifiers.pruning.fake_moe import (
    FakeMoEConfig,
    FakeMoEModel,
    make_state,
)
from tests.testing_utils import requires_gpu, torchrun


def _stats_modifier(num_experts: int = 4) -> REAPPruningModifier:
    """A REAP modifier with initialized statistics for a single layer."""
    modifier = REAPPruningModifier(sparsity=0.5)
    modifier._moe_config = MoEConfig.from_config(FakeMoEConfig(num_experts=num_experts))
    modifier._n_experts_to_drop = num_experts // 2
    modifier.init_moe_stats("layer", num_experts, 2, torch.device("cpu"))
    return modifier


def _run_batch(
    modifier: REAPPruningModifier,
    expert_outputs: dict[int, torch.Tensor],
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
):
    """Feed one calibration batch through the modifier's hooks.

    ``expert_outputs`` maps each expert index to the output its forward hook
    saw for the batch: the expert's routed tokens only under sparse
    execution, and every token under calibrate-all execution (where unrouted
    experts still run, on an empty batch of tokens or all tokens
    respectively)."""
    for expert_idx, output in expert_outputs.items():
        modifier.expert_hook("layer", expert_idx, None, (), output)
    hidden = torch.zeros(topk_indices.shape[0], 1)
    modifier.experts_block_hook(
        "layer", None, (hidden, topk_indices, topk_weights), None
    )


# ---------------------------------------------------------------------------
# Tests: saliency accumulation
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("calibrate_all", [False, True])
def test_saliency_is_mean_weighted_norm(calibrate_all, monkeypatch):
    """``S_j = mean(g_j * ||f_j||_2)`` over the tokens routed to expert
    ``j``, accumulating across batches. The sparse and calibrate-all
    execution paths must produce the same statistic from the same routing."""
    monkeypatch.setattr(moe_context, "_CALIBRATE_ALL_EXPERTS", calibrate_all)
    modifier = _stats_modifier(num_experts=4)

    # 2 tokens, top-2 routing; expert 3 is never routed
    topk_indices = torch.tensor([[0, 1], [0, 2]])
    topk_weights = torch.tensor([[0.6, 0.4], [0.5, 0.5]])

    if calibrate_all:
        # dense execution: every expert runs on every token
        outputs = {
            0: torch.tensor([[3.0], [1.0]]),
            1: torch.tensor([[2.0], [5.0]]),
            2: torch.tensor([[6.0], [4.0]]),
            3: torch.tensor([[7.0], [8.0]]),
        }
    else:
        # sparse execution: experts run on their routed tokens only, so an
        # unrouted expert still fires its hook on an empty batch of tokens
        outputs = {
            0: torch.tensor([[3.0], [1.0]]),
            1: torch.tensor([[2.0]]),
            2: torch.tensor([[4.0]]),
            3: torch.empty(0, 1),
        }
    _run_batch(modifier, outputs, topk_indices, topk_weights)

    # expert 0: (0.6*3.0 + 0.5*1.0) / 2 tokens; expert 1: 0.4*2.0 / 1;
    # expert 2: 0.5*4.0 / 1; expert 3: never routed
    assert modifier.saliency("layer").tolist() == pytest.approx([1.15, 0.8, 2.0, 0.0])

    # a second batch accumulates on top of the first
    second_outputs = (
        {i: torch.tensor([[5.0]]) for i in range(4)}
        if calibrate_all
        else {i: torch.empty(0, 1) for i in range(4)}
    )
    second_outputs[0] = torch.tensor([[1.0]])
    _run_batch(modifier, second_outputs, torch.tensor([[0]]), torch.tensor([[0.4]]))

    # expert 0: (0.6*3.0 + 0.5*1.0 + 0.4*1.0) / 3 tokens
    assert modifier.saliency("layer")[0].item() == pytest.approx(2.7 / 3)


@pytest.mark.unit
def test_saliency_agrees_between_sparse_and_calibrate_all_execution(monkeypatch):
    """Real forward passes produce the same saliency whether experts run on
    their routed tokens only or on every token (calibrate-all)."""
    torch.manual_seed(42)
    config = FakeMoEConfig(
        num_experts=8, hidden_size=16, intermediate_size=32, num_hidden_layers=2
    )
    reference_model = FakeMoEModel(config)
    reference_model.eval()
    reference_state = reference_model.state_dict()

    saliency_by_mode = {}
    for calibrate_all in (False, True):
        monkeypatch.setattr(moe_context, "_CALIBRATE_ALL_EXPERTS", calibrate_all)
        model = FakeMoEModel(config)
        model.load_state_dict(reference_state)
        model.eval()

        modifier = REAPPruningModifier(sparsity=0.5)
        state = make_state(model)
        modifier.initialize(state)
        modifier.update_event(state, Event(type_=EventType.CALIBRATION_START))

        torch.manual_seed(100)  # same calibration data in both modes
        with torch.no_grad():
            for _ in range(5):
                model(torch.randn(2, 8, config.hidden_size))

        saliency_by_mode[calibrate_all] = {
            layer_name: modifier.saliency(layer_name).clone()
            for layer_name in modifier._moe_stats
        }

    # the two paths sum the same contributions in a different order, so
    # allow for float32 reduction-order noise
    for layer_name, sparse in saliency_by_mode[False].items():
        torch.testing.assert_close(
            sparse,
            saliency_by_mode[True][layer_name],
            rtol=1e-5,
            atol=1e-6,
        )


# ---------------------------------------------------------------------------
# Tests: full-model lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_reap_full_lifecycle():
    """Calibration accumulates saliency through real routing,
    ``SEQUENTIAL_SUBGRAPH_END`` prunes every layer to its retained experts, and
    the pruning is recorded in the expert-sparsity config while the model
    config is left untouched."""
    torch.manual_seed(42)
    config = FakeMoEConfig(
        num_experts=8, hidden_size=16, intermediate_size=32, num_hidden_layers=2
    )
    model = FakeMoEModel(config)
    model.eval()

    modifier = REAPPruningModifier(sparsity=0.5)
    state = make_state(model)

    modifier.initialize(state)
    assert modifier._n_experts_to_drop == 4

    modifier.update_event(state, Event(type_=EventType.CALIBRATION_START))
    assert len(modifier._moe_stats) == 2

    with torch.no_grad():
        for _ in range(5):
            model(torch.randn(2, 8, config.hidden_size))

    modifier.update_event(state, Event(type_=EventType.SEQUENTIAL_SUBGRAPH_END))
    # every layer's statistics were consumed
    assert len(modifier._moe_stats) == 0

    modifier.update_event(state, Event(type_=EventType.CALIBRATION_END))
    modifier.finalize(state)

    assert model.config.num_experts == 8
    assert model._moe_sparsity_config.method.value == "prune"
    assert model._moe_sparsity_config.num_experts == [4, 4]

    for layer in model.layers:
        expert_count = sum(
            1 for m in layer.experts.children() if isinstance(m, ExpertMLP)
        )
        assert expert_count == 4
        assert layer.experts.num_experts == 4
        assert layer.gate.weight.shape[0] == 4
        assert layer.gate.out_features == 4

    # the pruned model still runs
    with torch.no_grad():
        out = model(torch.randn(2, 4, config.hidden_size))
    assert out.shape == (2, 4, config.hidden_size)
    assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# Tests: distributed calibration
# ---------------------------------------------------------------------------


@pytest.mark.unit
@torchrun(world_size=2, init_dist=False)
def test_reap_distributed_uneven_data():
    """Layers must be reduced and pruned identically on all ranks when
    calibration data is unevenly distributed across ranks.

    Every rank reduces every tracked layer (contributing zeros where it saw
    no tokens) and the decision of which layers collected samples is made
    from the total count summed across ranks. This covers both the case
    where only the non-source rank saw data (deciding from the source rank's
    local count alone would skip the layer everywhere) and the case where
    only the source rank saw data (a rank without data must still join every
    reduction rather than deadlocking it).
    """
    dist.init_process_group(backend="gloo", init_method="env://")
    try:
        rank = dist.get_rank()

        for data_rank in (1, 0):
            torch.manual_seed(42)
            config = FakeMoEConfig(
                num_experts=8,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
            )
            model = FakeMoEModel(config)
            model.eval()

            modifier = REAPPruningModifier(sparsity=0.5)
            state = make_state(model)

            modifier.initialize(state)
            modifier.update_event(state, Event(type_=EventType.CALIBRATION_START))

            # only data_rank processes calibration data
            if rank == data_rank:
                with torch.no_grad():
                    for _ in range(5):
                        model(torch.randn(2, 8, config.hidden_size))

            modifier.update_event(state, Event(type_=EventType.SEQUENTIAL_SUBGRAPH_END))
            modifier.update_event(state, Event(type_=EventType.CALIBRATION_END))
            modifier.finalize(state)

            # every layer was pruned from the globally summed statistics ...
            for layer in model.layers:
                assert layer.experts.num_experts == 4
                assert layer.gate.weight.shape[0] == 4

            # ... identically on every rank (retained experts are broadcast
            # from the source rank)
            gate = torch.stack([layer.gate.weight.data for layer in model.layers]).to(
                torch.float64
            )
            gathered = [torch.empty_like(gate) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, gate.contiguous())
            for other in gathered:
                torch.testing.assert_close(other, gate)

        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.unit
@requires_gpu
@torchrun(world_size=1, init_dist=True)
def test_reap_distributed_nccl_prunes_on_gpu():
    """The reduction and pruning run on the gpu (float32) over the default
    nccl-backed process group, rather than on host copies."""
    # nccl is a cuda-only backend; other accelerators (e.g. xpu) use ccl
    if not torch.accelerator.is_available() or (
        torch.accelerator.current_accelerator().type != "cuda"
    ):
        pytest.skip("nccl-backed process group requires a CUDA gpu")

    device = torch.device("cuda", dist.get_rank())
    assert dist.get_backend() == "nccl"

    torch.manual_seed(42)
    config = FakeMoEConfig(
        num_experts=8, hidden_size=16, intermediate_size=32, num_hidden_layers=2
    )
    model = FakeMoEModel(config).to(device)
    model.eval()

    modifier = REAPPruningModifier(sparsity=0.5)
    state = make_state(model)

    modifier.initialize(state)
    modifier.update_event(state, Event(type_=EventType.CALIBRATION_START))

    with torch.no_grad():
        for _ in range(5):
            model(torch.randn(2, 8, config.hidden_size, device=device))

    modifier.update_event(state, Event(type_=EventType.SEQUENTIAL_SUBGRAPH_END))
    modifier.update_event(state, Event(type_=EventType.CALIBRATION_END))
    modifier.finalize(state)

    # every layer pruned, with the router shrunk on the gpu
    for layer in model.layers:
        assert layer.experts.num_experts == 4
        assert layer.gate.weight.shape[0] == 4
        assert layer.gate.weight.device == device
