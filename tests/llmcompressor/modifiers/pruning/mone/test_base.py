"""
Unit tests for the MoNe pruning modifier.

The shared pruning lifecycle is tested in
``tests/llmcompressor/modifiers/pruning/test_base.py``; the tests here cover
MoNe's own behavior: the fusion saliency metric
``S_j = ||std_tokens(f_j)||_2 * E_tokens[g_j * 1(j in top-k)]``, the novice
vectors collapsed experts are replaced with, the structural collapse
(retained experts left-shifted, router kept at full width), novice serving
and checkpoint serialization, and distributed calibration.
"""

import glob
import json
import os

import pytest
import torch
import torch.nn as nn
from safetensors import safe_open
from torch import distributed as dist

from llmcompressor.core import Event, EventType
from llmcompressor.modeling.moe import context as moe_context
from llmcompressor.modeling.moe.helpers import MoEConfig
from llmcompressor.modeling.moe.linear_experts import ExpertMLP
from llmcompressor.modifiers.pruning.expert_sparsity.utils import get_experts_router
from llmcompressor.modifiers.pruning.mone import MoNePruningModifier
from llmcompressor.modifiers.pruning.mone.utils import mone_collapse_moe_layer
from tests.llmcompressor.modifiers.pruning.fake_moe import (
    FakeLinearExperts2D,
    FakeMoEConfig,
    FakeMoEModel,
    find_moe_blocks,
    make_state,
)
from tests.testing_utils import requires_gpu, torchrun


def _stats_modifier(num_experts: int = 4, hidden_dim: int = 2) -> MoNePruningModifier:
    """A MoNe modifier with initialized statistics for a single layer."""
    modifier = MoNePruningModifier(sparsity=0.5)
    modifier._moe_config = MoEConfig.from_config(FakeMoEConfig(num_experts=num_experts))
    modifier._n_experts_to_drop = num_experts // 2
    modifier.init_moe_stats("layer", num_experts, hidden_dim, torch.device("cpu"))
    return modifier


def _run_batch(
    modifier: MoNePruningModifier,
    expert_outputs: dict[int, torch.Tensor],
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
):
    """Feed one calibration batch through the modifier's hooks.

    ``expert_outputs`` maps each expert index to the output its forward hook
    saw for the batch: the expert's routed tokens only under sparse
    execution, and every token under calibrate-all execution."""
    for expert_idx, output in expert_outputs.items():
        modifier.expert_hook("layer", expert_idx, None, (), output)
    hidden = torch.zeros(topk_indices.shape[0], 1)
    modifier.experts_block_hook(
        "layer", None, (hidden, topk_indices, topk_weights), None
    )


# ---------------------------------------------------------------------------
# Tests: fusion saliency metric and novices
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("calibrate_all", [False, True])
def test_fusion_saliency_metric(calibrate_all, monkeypatch):
    """``S_j = ||std_tokens(f_j)||_2 * E_tokens[g_j * 1(j in top-k)]``: the
    product of the expert's output fluctuation and its frequency-weighted
    mean router gate weight. The sparse and calibrate-all execution paths
    must produce the same statistic from the same routing."""
    monkeypatch.setattr(moe_context, "_CALIBRATE_ALL_EXPERTS", calibrate_all)
    modifier = _stats_modifier(num_experts=4, hidden_dim=2)

    # token 0 routes to experts 0 and 1, token 1 to experts 1 and 2; expert 3
    # is never routed
    topk_indices = torch.tensor([[0, 1], [1, 2]])
    topk_weights = torch.tensor([[0.6, 0.4], [0.5, 0.5]])

    if calibrate_all:
        # dense execution: every expert runs on every token, and outputs of
        # tokens which were not routed to the expert are masked out
        outputs = {
            0: torch.tensor([[1.0, 2.0], [3.0, 6.0]]),
            1: torch.tensor([[10.0, 10.0], [12.0, 14.0]]),
            2: torch.tensor([[9.0, 9.0], [5.0, 5.0]]),
            3: torch.tensor([[7.0, 7.0], [8.0, 8.0]]),
        }
    else:
        # sparse execution: experts run on their routed tokens only
        outputs = {
            0: torch.tensor([[1.0, 2.0]]),
            1: torch.tensor([[10.0, 10.0], [12.0, 14.0]]),
            2: torch.tensor([[5.0, 5.0]]),
            3: torch.empty(0, 2),
        }
    _run_batch(modifier, outputs, topk_indices, topk_weights)

    # routing score: mean gate weight over the 2 tokens, zero when never
    # routed (experts 0, 1, 2 get 1, 2, and 1 tokens respectively)
    torch.testing.assert_close(
        modifier._routing_score("layer"),
        torch.tensor([0.3, 0.45, 0.25, 0.0]),
        rtol=1e-6,
        atol=1e-6,
    )

    # output fluctuation: L2 norm over dims of the per-dim output std.
    # expert 0: 1 token -> 0; expert 1: std=[1, 2] -> sqrt(5);
    # expert 2: 1 token -> 0
    fluctuation = modifier._output_fluctuation("layer")
    assert fluctuation[0].item() == pytest.approx(0.0)
    assert fluctuation[1].item() == pytest.approx(5**0.5)
    assert fluctuation[2:].tolist() == [0.0, 0.0]

    saliency = modifier.saliency("layer")
    assert saliency[0].item() == pytest.approx(0.0)
    assert saliency[1].item() == pytest.approx(5**0.5 * 0.45)
    assert saliency[2:].tolist() == [0.0, 0.0]


@pytest.mark.unit
def test_novices_are_mean_output_of_dropped_experts():
    """Each collapsed expert is replaced by its mean output over the tokens
    routed to it, with novice rows ordered to match the dropped experts."""
    modifier = _stats_modifier(num_experts=4, hidden_dim=2)

    # expert 1 is routed two tokens, expert 3 one, expert 0 one
    _run_batch(
        modifier,
        {1: torch.tensor([[1.0, 2.0]]), 3: torch.tensor([[7.0, 8.0]])},
        torch.tensor([[1, 3]]),
        torch.tensor([[0.5, 0.5]]),
    )
    _run_batch(
        modifier,
        {1: torch.tensor([[3.0, 4.0]]), 0: torch.tensor([[0.0, 0.0]])},
        torch.tensor([[1, 0]]),
        torch.tensor([[0.5, 0.5]]),
    )

    # expert 1 mean: ([1, 2] + [3, 4]) / 2; expert 3 mean: [7, 8] / 1
    novices = modifier._compute_novices("layer", dropped=[1, 3])
    torch.testing.assert_close(novices, torch.tensor([[2.0, 3.0], [7.0, 8.0]]))

    # novice rows follow the order of ``dropped``
    torch.testing.assert_close(
        modifier._compute_novices("layer", dropped=[3, 1]),
        torch.tensor([[7.0, 8.0], [2.0, 3.0]]),
    )


# ---------------------------------------------------------------------------
# Tests: novice serving
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_linear_experts_serves_novices():
    """Tokens routed past the retained experts are served the corresponding
    novice vector, weighted by their routing weight."""
    torch.manual_seed(3)
    config = FakeMoEConfig(num_experts=4, hidden_size=8, intermediate_size=16)
    experts = FakeLinearExperts2D(config)

    # keep 2 experts, collapse the other 2 into novices
    experts._modules = {"0": experts[0], "1": experts[1], "act_fn": experts.act_fn}
    experts.num_experts = 2
    novices = torch.randn(2, 8)
    experts.register_parameter("novices", nn.Parameter(novices, requires_grad=False))

    hidden = torch.randn(5, 8)
    top_k_index = torch.tensor([[0, 2], [1, 3], [2, 3], [0, 1], [3, 2]])
    top_k_weights = torch.rand(5, 2)

    out = experts(hidden, top_k_index, top_k_weights)

    ref = torch.zeros_like(hidden)
    for t in range(5):
        for slot in range(2):
            expert = top_k_index[t, slot].item()
            weight = top_k_weights[t, slot].item()
            if expert < 2:
                ref[t] += weight * experts[expert](hidden[t : t + 1]).squeeze(0)
            else:
                ref[t] += weight * novices[expert - 2]
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Tests: mone_collapse_moe_layer
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_collapse_keeps_router_width_and_sets_novices():
    config = FakeMoEConfig(num_experts=8, num_hidden_layers=1, hidden_size=16)
    model = FakeMoEModel(config)
    layer = model.layers[0]
    original_experts = [
        expert for expert in layer.experts.children() if isinstance(expert, ExpertMLP)
    ]
    original_router = layer.gate.weight.detach().clone()

    retained = [1, 2, 4, 6, 7]
    dropped = [0, 3, 5]
    novices = torch.arange(len(dropped) * 16, dtype=torch.float32).reshape(
        len(dropped), 16
    )

    mone_collapse_moe_layer("layers.0", layer, retained, dropped, novices)

    # retained experts left-shifted (same module objects)
    pruned = [
        expert for expert in layer.experts.children() if isinstance(expert, ExpertMLP)
    ]
    assert len(pruned) == 5
    for new_idx, original_idx in enumerate(retained):
        assert pruned[new_idx] is original_experts[original_idx]
    assert layer.experts.num_experts == 5

    # router keeps full R + N width, reordered to [retained..., dropped...]
    assert layer.gate.weight.shape[0] == 8
    assert layer.gate.out_features == 8
    torch.testing.assert_close(layer.gate.weight, original_router[retained + dropped])

    # novices installed, matching the dropped order
    assert layer.experts.novices is not None
    assert layer.experts.novices.shape == (3, 16)
    torch.testing.assert_close(layer.experts.novices, novices)


@pytest.mark.unit
def test_collapse_forward_preserves_retained_experts():
    """A collapsed model routed only to retained experts matches the
    original."""
    torch.manual_seed(7)
    config = FakeMoEConfig(num_experts=6, num_hidden_layers=1, hidden_size=8)
    model = FakeMoEModel(config)
    layer = model.layers[0]

    hidden = torch.randn(4, 8)
    retained = [0, 1, 2, 3]
    # force routing to retained experts only
    top_k_index = torch.tensor([[0, 1], [2, 3], [0, 2], [1, 3]])
    top_k_weights = torch.rand(4, 2)
    baseline = layer.experts(hidden, top_k_index, top_k_weights)

    novices = torch.randn(2, 8)
    mone_collapse_moe_layer("layers.0", layer, retained, [4, 5], novices)

    # after collapse, retained experts occupy rows 0..3 unchanged
    out = layer.experts(hidden, top_k_index, top_k_weights)
    torch.testing.assert_close(out, baseline, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Tests: lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_mone_full_lifecycle():
    """Calibration accumulates routing and output statistics, and
    ``SEQUENTIAL_SUBGRAPH_END`` collapses every layer's lowest-saliency experts
    into novices: retained experts are left-shifted, the router keeps its
    full width, and the collapse is recorded in the expert-sparsity
    config."""
    torch.manual_seed(42)
    config = FakeMoEConfig(
        num_experts=8, hidden_size=16, intermediate_size=32, num_hidden_layers=2
    )
    model = FakeMoEModel(config)
    model.eval()

    modifier = MoNePruningModifier(sparsity=0.5)
    state = make_state(model)

    modifier.initialize(state)
    assert modifier._n_experts_to_drop == 4

    modifier.update_event(state, Event(type_=EventType.CALIBRATION_START))
    assert len(modifier._moe_stats) == 2

    with torch.no_grad():
        for _ in range(5):
            model(torch.randn(2, 8, config.hidden_size))

    modifier.update_event(state, Event(type_=EventType.SEQUENTIAL_SUBGRAPH_END))
    assert len(modifier._moe_stats) == 0

    modifier.update_event(state, Event(type_=EventType.CALIBRATION_END))
    modifier.finalize(state)

    assert model.config.num_experts == 8
    for layer in model.layers:
        expert_count = sum(
            1 for m in layer.experts.children() if isinstance(m, ExpertMLP)
        )
        assert expert_count == 4
        assert layer.experts.num_experts == 4
        # unlike plain pruning, the router keeps its full width so the
        # novices keep receiving their routing weights
        assert layer.gate.weight.shape[0] == 8
        assert layer.experts.novices is not None
        assert layer.experts.novices.shape == (4, config.hidden_size)

    assert model._moe_sparsity_config.method.value == "novice"
    assert model._moe_sparsity_config.num_experts == [4, 4]

    with torch.no_grad():
        out = model(torch.randn(2, 4, config.hidden_size))
    assert out.shape == (2, 4, config.hidden_size)
    assert not torch.isnan(out).any()


@pytest.mark.unit
def test_mone_group_limited_router_not_implemented():
    """Group-limited routers are explicitly unsupported."""
    config = FakeMoEConfig(num_experts=8, n_group=2, topk_group=1)
    model = FakeMoEModel(config)

    modifier = MoNePruningModifier(sparsity=0.5)
    with pytest.raises(NotImplementedError, match="group-limited"):
        modifier.initialize(make_state(model))


# ---------------------------------------------------------------------------
# Tests: checkpoint serialization (linearize / repack / save)
# ---------------------------------------------------------------------------


def _tiny_qwen3_vl_moe():
    from transformers import Qwen3VLMoeConfig, Qwen3VLMoeForConditionalGeneration

    from llmcompressor.utils.dev import skip_weights_initialize

    config = Qwen3VLMoeConfig(
        text_config={
            "hidden_size": 64,
            "intermediate_size": 128,
            "moe_intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "vocab_size": 256,
            "tie_word_embeddings": False,
        },
        vision_config={
            "depth": 1,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_heads": 4,
            "out_hidden_size": 64,
        },
    )
    with skip_weights_initialize():
        model = Qwen3VLMoeForConditionalGeneration(config)
    return model


def _linearized_and_collapsed(retained=(0, 2)):
    """A tiny linearized Qwen3VLMoe model whose single MoE layer keeps
    ``retained`` experts, with the rest collapsed into random novices.

    Returns the model and its ``(moe_name, moe_block)`` pair."""
    from llmcompressor.modeling.moe.linearize import linearize_moe

    model = _tiny_qwen3_vl_moe()
    linearize_moe(model)
    ((moe_name, block),) = find_moe_blocks(model).items()

    dropped = [i for i in range(4) if i not in retained]
    novices = torch.randn(len(dropped), 64)
    mone_collapse_moe_layer(moe_name, block, list(retained), dropped, novices)
    return model, moe_name, block


@pytest.mark.unit
def test_collapse_survives_linearize_and_repack():
    """The collapsed layout survives the linearize/repack round trip: the
    retained experts are left-shifted and packed, and the novices stay a
    tensor on the experts module with the router at full width."""
    from llmcompressor.modeling.moe.linearize import repack_moe

    model, moe_name, block = _linearized_and_collapsed()
    experts, router = get_experts_router(moe_name, block)

    # linearized layout: retained experts left-shifted to 0..R-1, novices
    # installed as a parameter on the experts module
    keys = set(experts.state_dict())
    assert "novices" in keys
    assert experts.state_dict()["novices"].shape == (2, 64)
    assert any(key.startswith("0.") for key in keys)
    assert not any(key.startswith("2.") for key in keys)

    # the router keeps its full R + N width
    assert router.weight.shape[0] == 4

    # repack fuses the retained experts and keeps the novices
    experts_attr = next(
        name for name, child in block.named_children() if child is experts
    )
    repack_moe(model)
    fused = getattr(block, experts_attr)
    assert fused.gate_up_proj.shape[0] == 2
    assert fused.down_proj.shape[0] == 2
    assert fused.novices.shape == (2, 64)


@pytest.mark.unit
def test_save_pretrained_writes_moe_sparsity_config_and_novices(tmp_path):
    from compressed_tensors.config import ExpertSparsityConfig, MoESparsityMethod

    from llmcompressor.modeling.moe.linearize import repack_moe
    from llmcompressor.transformers.compression.compressed_tensors_utils import (
        modify_save_pretrained,
    )
    from llmcompressor.utils.helpers import values_by_decoder_layer

    model, moe_name, _ = _linearized_and_collapsed()

    # the modifier's finalize step records the per-layer retained counts
    num_layers = model.config.get_text_config().num_hidden_layers
    model._moe_sparsity_config = ExpertSparsityConfig(
        method=MoESparsityMethod.NOVICE,
        num_experts=values_by_decoder_layer({moe_name: 2}, num_layers),
    )
    repack_moe(model)

    modify_save_pretrained(model)
    out_dir = tmp_path / "mone_model"
    model.save_pretrained(out_dir, safe_serialization=True)

    # config.json: quantization_config.moe_sparsity_config
    with open(out_dir / "config.json") as f:
        config_data = json.load(f)
    msc = config_data["quantization_config"]["moe_sparsity_config"]
    assert msc["method"] == "novice"
    assert msc["num_experts"] == [2]

    # checkpoint: experts.novices tensor present, retained experts packed
    expert_keys = []
    for path in glob.glob(os.path.join(out_dir, "*.safetensors")):
        with safe_open(path, framework="pt") as handle:
            expert_keys.extend(k for k in handle.keys() if "mlp.experts" in k)
    assert any(k.endswith("experts.novices") for k in expert_keys)
    assert any(k.endswith("experts.gate_up_proj") for k in expert_keys)
    assert not any(".experts.0." in k for k in expert_keys)


# ---------------------------------------------------------------------------
# Tests: distributed (gloo) calibration with unevenly distributed data
# ---------------------------------------------------------------------------


@pytest.mark.unit
@torchrun(world_size=2, init_dist=False)
def test_mone_distributed_uneven_data():
    """Layers must be reduced and collapsed identically on all ranks when
    calibration data is unevenly distributed across ranks.

    Every rank reduces every tracked layer (contributing zeros where it saw
    no tokens) and the decision of which layers collected samples is made
    from the token count summed across ranks. This covers both the case
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

            modifier = MoNePruningModifier(sparsity=0.5)
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

            # every layer was collapsed from the globally summed statistics
            for layer in model.layers:
                assert layer.experts.num_experts == 4
                assert layer.experts.novices is not None
                assert layer.experts.novices.shape == (4, config.hidden_size)

            # ... identically on every rank (retained experts and novices are
            # broadcast from the source rank)
            novices = torch.stack(
                [layer.experts.novices.data for layer in model.layers]
            ).to(torch.float64)
            gathered = [torch.empty_like(novices) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, novices.contiguous())
            for other in gathered:
                torch.testing.assert_close(other, novices)

        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.unit
@requires_gpu
@torchrun(world_size=1, init_dist=True)
def test_mone_distributed_nccl_collapses_on_gpu():
    """The reduction and collapse run on the gpu (float32) over the default
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

    modifier = MoNePruningModifier(sparsity=0.5)
    state = make_state(model)

    modifier.initialize(state)
    modifier.update_event(state, Event(type_=EventType.CALIBRATION_START))

    with torch.no_grad():
        for _ in range(5):
            model(torch.randn(2, 8, config.hidden_size, device=device))

    modifier.update_event(state, Event(type_=EventType.SEQUENTIAL_SUBGRAPH_END))
    modifier.update_event(state, Event(type_=EventType.CALIBRATION_END))
    modifier.finalize(state)

    # every layer collapsed, with novices installed on the gpu
    for layer in model.layers:
        assert layer.experts.num_experts == 4
        assert layer.experts.novices is not None
        assert layer.experts.novices.shape == (4, config.hidden_size)
        assert layer.experts.novices.device == device
