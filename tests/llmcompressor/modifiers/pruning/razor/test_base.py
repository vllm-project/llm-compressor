"""
Unit tests for the RAZOR pruning modifier.

The shared pruning lifecycle is tested in
``tests/llmcompressor/modifiers/pruning/test_base.py``; the tests here cover
RAZOR's own behavior: the closed-form token damage against the direct
deletion counterfactual it is derived from, the accumulated statistics
against an independent reference implementation (through real routing, in
both sparse and calibrate-all execution, with and without a router score
correction bias), the reduction of the refill criterion to the
leave-one-out one when no unselected expert is available, score-function
resolution, full-model pruning, and distributed calibration with unevenly
distributed data.
"""

import pytest
import torch
import torch.nn.functional as F
from pydantic import ValidationError
from torch import distributed as dist

from llmcompressor.core import Event, EventType
from llmcompressor.modeling.moe import context as moe_context
from llmcompressor.modeling.moe.linear_experts import ExpertMLP
from llmcompressor.modifiers.pruning.razor import RazorPruningModifier
from llmcompressor.modifiers.pruning.razor.base import (
    _apply_score_function,
    _token_damage,
)
from tests.llmcompressor.modifiers.pruning.fake_moe import (
    FakeMoEConfig,
    FakeMoEModel,
    make_state,
)
from tests.testing_utils import torchrun

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reference_saliency(
    layer: torch.nn.Module,
    layer_inputs: list[torch.Tensor],
    criterion: str,
    num_experts: int,
    top_k: int,
    score_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the RAZOR saliency of a layer from the inputs it saw.

    An independent reimplementation of the metric: the router scores and
    expert outputs are recomputed directly from the layer's modules, and the
    per-pair damage is accumulated per expert.
    """
    sum_sq_damage = torch.zeros(num_experts)
    count = torch.zeros(num_experts)

    for hidden in layer_inputs:  # [num_tokens, hidden_dim]
        scores = F.linear(hidden, layer.gate.weight).softmax(dim=-1)
        topk_weights, topk_idx = scores.topk(top_k, dim=-1)
        weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        with torch.no_grad():
            # [num_experts, num_tokens, hidden_dim]
            outputs = torch.stack(
                [layer.experts[e](hidden) for e in range(num_experts)]
            )
        tokens = torch.arange(hidden.shape[0])
        selected = outputs[topk_idx, tokens.unsqueeze(1)]  # [T, k, d]
        consensus = (weights.unsqueeze(-1) * selected).sum(dim=1)  # [T, d]

        if criterion == "refill" and num_experts > top_k:
            selection = scores
            if score_bias is not None:
                selection = selection + score_bias
            chosen = torch.zeros_like(scores, dtype=torch.bool)
            chosen.scatter_(1, topk_idx, True)
            promoted = selection.masked_fill(chosen, -float("inf")).argmax(dim=-1)
            refill_weights = scores.gather(1, promoted.unsqueeze(1)).squeeze(
                1
            ) / topk_weights.sum(dim=-1)
            refill_outputs = outputs[promoted, tokens]  # [T, d]

        for expert_idx in range(num_experts):
            mask = topk_idx == expert_idx  # [T, k]
            if not mask.any():
                continue
            tok, slot = mask.nonzero(as_tuple=True)
            pair_weights = weights[tok, slot]
            residuals = outputs[expert_idx, tok] - consensus[tok]
            if criterion == "refill" and num_experts > top_k:
                pair_refill_weights = refill_weights[tok]
                pair_refill_residuals = refill_outputs[tok] - consensus[tok]
            else:
                pair_refill_weights = None
                pair_refill_residuals = None
            damage = _token_damage(
                criterion,
                pair_weights,
                residuals,
                pair_refill_weights,
                pair_refill_residuals,
            )
            sum_sq_damage[expert_idx] += damage.pow(2).sum()
            count[expert_idx] += damage.shape[0]

    return (sum_sq_damage / count.clamp(min=1.0)).sqrt()


def _calibrate(
    model: torch.nn.Module,
    modifier: RazorPruningModifier,
    num_batches: int,
    batch_shape: tuple,
    seed: int = 100,
) -> dict[int, list[torch.Tensor]]:
    """Run calibration batches through the model, returning the (flattened)
    input each MoE block saw per batch, by layer index.

    The inputs are recorded with test-local hooks so the reference
    implementation can recompute the metric from the very activations the
    modifier's hooks saw. Statistics are left accumulated (no
    ``SEQUENTIAL_EPOCH_END``), so ``modifier.saliency`` remains readable.
    """
    state = make_state(model)
    modifier.initialize(state)
    modifier.update_event(state, Event(type_=EventType.CALIBRATION_START))

    layer_inputs: dict[int, list[torch.Tensor]] = {
        idx: [] for idx in range(len(model.layers))
    }

    def _capture(idx):
        def hook(block, args):
            layer_inputs[idx].append(args[0].reshape(-1, args[0].shape[-1]))

        return hook

    handles = [
        block.register_forward_pre_hook(_capture(idx))
        for idx, block in enumerate(model.layers)
    ]
    try:
        torch.manual_seed(seed)
        with torch.no_grad():
            for _ in range(num_batches):
                model(torch.randn(*batch_shape))
    finally:
        for handle in handles:
            handle.remove()

    return layer_inputs


def _expected_saliency(
    model: torch.nn.Module,
    layer_inputs: dict[int, list[torch.Tensor]],
    criterion: str,
    config: FakeMoEConfig,
    score_bias: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """The reference saliency of every layer, by layer name, computed with
    the modifier's hooks disabled so the reference's own expert forwards do
    not disturb the accumulated statistics."""
    with RazorPruningModifier.disable_hooks():
        return {
            f"layers.{idx}": _reference_saliency(
                model.layers[idx],
                layer_inputs[idx],
                criterion,
                num_experts=config.num_experts,
                top_k=config.num_experts_per_tok,
                score_bias=score_bias,
            )
            for idx in range(len(model.layers))
        }


# ---------------------------------------------------------------------------
# Tests: the token-level damage formula
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("criterion", ["rcs", "rcs_loo", "refill"])
def test_token_damage_matches_direct_counterfactual(criterion):
    """The closed-form damage equals the L2 distance between the consensus
    and the deletion counterfactual mixture it is derived from (propositions
    2 and 3 of the paper)."""
    torch.manual_seed(0)
    top_k, hidden_dim = 3, 5
    weights = torch.softmax(torch.randn(top_k), dim=-1)  # selected weights
    outputs = torch.randn(top_k + 1, hidden_dim)  # k selected + 1 promoted

    consensus = (weights.unsqueeze(-1) * outputs[:top_k]).sum(dim=0)
    residuals = outputs[:top_k] - consensus

    promoted_weight = torch.tensor(0.37)
    promoted_residual = outputs[top_k] - consensus

    if criterion == "refill":
        damage = _token_damage(
            criterion,
            weights,
            residuals,
            torch.full((top_k,), promoted_weight),
            promoted_residual.expand(top_k, -1),
        )
    else:
        damage = _token_damage(criterion, weights, residuals)

    for slot in range(top_k):
        weight, expert_out = weights[slot], outputs[slot]
        if criterion == "rcs":
            expected = weight * (expert_out - consensus).norm()
        elif criterion == "rcs_loo":
            # deleting the expert renormalizes the survivors over S \ {i}
            survivor = (consensus - weight * expert_out) / (1 - weight)
            expected = (consensus - survivor).norm()
        else:
            # ... and promotes the replacement expert (Assumption 1)
            refilled = (
                consensus - weight * expert_out + promoted_weight * outputs[top_k]
            ) / (1 - weight + promoted_weight)
            expected = (consensus - refilled).norm()
        assert damage[slot].item() == pytest.approx(expected.item(), rel=1e-5)


@pytest.mark.unit
def test_token_damage_refill_reduces_to_loo_without_promotion():
    """With no unselected expert available to promote, the refill criterion
    evaluates to the leave-one-out damage."""
    torch.manual_seed(1)
    weights = torch.softmax(torch.randn(4), dim=-1)
    residuals = torch.randn(4, 6)

    refill = _token_damage("refill", weights, residuals)
    loo = _token_damage("rcs_loo", weights, residuals)
    torch.testing.assert_close(refill, loo)


@pytest.mark.unit
def test_apply_score_function():
    """Each score function transforms the router logits as specified."""
    logits = torch.tensor([[2.0, -1.0], [0.5, 0.5]])
    torch.testing.assert_close(
        _apply_score_function(logits, "softmax"), logits.softmax(dim=-1)
    )
    torch.testing.assert_close(
        _apply_score_function(logits, "sigmoid"), logits.sigmoid()
    )
    torch.testing.assert_close(
        _apply_score_function(logits, "sqrt_softplus"), F.softplus(logits).sqrt()
    )
    with pytest.raises(ValueError, match="score function"):
        _apply_score_function(logits, "relu")


# ---------------------------------------------------------------------------
# Tests: score-function resolution
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "config_value, expected",
    [("sigmoid", "sigmoid"), ("softmax", "softmax"), ("tanh", "softmax")],
)
def test_score_function_inferred_from_config(config_value, expected):
    """The score function is inferred from the model config's scoring field,
    falling back to softmax for unknown values."""
    config = FakeMoEConfig(num_experts=8)
    config.scoring_func = config_value
    model = FakeMoEModel(config)

    modifier = RazorPruningModifier(sparsity=0.5)
    modifier.initialize(make_state(model))
    assert modifier._resolved_score_function == expected


@pytest.mark.unit
def test_score_function_override_wins():
    """An explicit score_function is used as-is."""
    config = FakeMoEConfig(num_experts=8)
    config.scoring_func = "sigmoid"
    model = FakeMoEModel(config)

    modifier = RazorPruningModifier(sparsity=0.5, score_function="sqrt_softplus")
    modifier.initialize(make_state(model))
    assert modifier._resolved_score_function == "sqrt_softplus"


@pytest.mark.unit
def test_invalid_score_function_rejected():
    with pytest.raises(ValidationError):
        RazorPruningModifier(sparsity=0.5, score_function="relu")


# ---------------------------------------------------------------------------
# Tests: saliency accumulation
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("criterion", ["rcs", "rcs_loo", "refill"])
@pytest.mark.parametrize("calibrate_all", [False, True])
def test_saliency_matches_reference_implementation(
    criterion, calibrate_all, monkeypatch
):
    """Accumulated statistics agree with an independent computation of the
    metric from the layer's own routing and expert modules, in both the
    sparse and calibrate-all execution paths."""
    monkeypatch.setattr(moe_context, "_CALIBRATE_ALL_EXPERTS", calibrate_all)

    torch.manual_seed(42)
    config = FakeMoEConfig(
        num_experts=8, hidden_size=16, intermediate_size=32, num_hidden_layers=2
    )
    model = FakeMoEModel(config)
    model.eval()

    modifier = RazorPruningModifier(sparsity=0.5, criterion=criterion)
    layer_inputs = _calibrate(model, modifier, num_batches=4, batch_shape=(2, 8, 16))

    expected = _expected_saliency(model, layer_inputs, criterion, config)
    for layer_idx, (layer_name, expected_saliency) in enumerate(expected.items()):
        torch.testing.assert_close(
            modifier.saliency(layer_name),
            expected_saliency,
            rtol=1e-5,
            atol=1e-6,
            msg=lambda m, i=layer_idx: f"layer {i}: saliency differs\n{m}",
        )


@pytest.mark.unit
def test_saliency_matches_reference_with_correction_bias(monkeypatch):
    """An ``e_score_correction_bias`` steers which unselected expert the
    refill criterion promotes (the router's selection rule) without entering
    the mixture weights."""
    monkeypatch.setattr(moe_context, "_CALIBRATE_ALL_EXPERTS", False)

    torch.manual_seed(42)
    config = FakeMoEConfig(
        num_experts=8, hidden_size=16, intermediate_size=32, num_hidden_layers=2
    )
    model = FakeMoEModel(config)
    model.eval()

    # a large bias dominates the selection ranking of the promoted expert
    score_bias = torch.randn(config.num_experts) * 5.0
    for layer in model.layers:
        layer.gate.e_score_correction_bias = torch.nn.Parameter(score_bias.clone())

    modifier = RazorPruningModifier(sparsity=0.5, criterion="refill")
    layer_inputs = _calibrate(model, modifier, num_batches=4, batch_shape=(2, 8, 16))

    expected = _expected_saliency(
        model, layer_inputs, "refill", config, score_bias=score_bias
    )
    for layer_idx, (layer_name, expected_saliency) in enumerate(expected.items()):
        torch.testing.assert_close(
            modifier.saliency(layer_name),
            expected_saliency,
            rtol=1e-5,
            atol=1e-6,
            msg=lambda m, i=layer_idx: f"layer {i}: saliency differs\n{m}",
        )


@pytest.mark.unit
@pytest.mark.parametrize("criterion", ["rcs", "rcs_loo", "refill"])
def test_saliency_agrees_between_sparse_and_calibrate_all_execution(
    criterion, monkeypatch
):
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

        modifier = RazorPruningModifier(sparsity=0.5, criterion=criterion)
        _calibrate(model, modifier, num_batches=5, batch_shape=(2, 8, 16))

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


@pytest.mark.unit
def test_refill_reduces_to_loo_when_all_experts_are_selected(tmp_path):
    """When the router selects every expert, no unselected expert is
    available to promote, so the refill criterion reduces to the
    leave-one-out one."""
    torch.manual_seed(42)
    config = FakeMoEConfig(
        num_experts=4, hidden_size=16, intermediate_size=32, num_hidden_layers=2
    )
    config.num_experts_per_tok = 4  # top-k selects every expert
    reference_state = FakeMoEModel(config).state_dict()

    saliency_by_criterion = {}
    for criterion in ("refill", "rcs_loo"):
        model = FakeMoEModel(config)
        model.load_state_dict(reference_state)
        model.eval()

        # report-only mode (sparsity is 0, nothing is pruned): the
        # accumulated saliency alone is compared
        modifier = RazorPruningModifier(
            sparsity=0.0,
            criterion=criterion,
            report_path=str(tmp_path / f"{criterion}.json"),
        )
        _calibrate(model, modifier, num_batches=4, batch_shape=(2, 8, 16))
        saliency_by_criterion[criterion] = {
            layer_name: modifier.saliency(layer_name).clone()
            for layer_name in modifier._moe_stats
        }

    for layer_name, refill in saliency_by_criterion["refill"].items():
        torch.testing.assert_close(refill, saliency_by_criterion["rcs_loo"][layer_name])


# ---------------------------------------------------------------------------
# Tests: full-model lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_razor_full_lifecycle():
    """Calibration accumulates saliency through real routing,
    ``SEQUENTIAL_EPOCH_END`` prunes every layer to its retained experts, and
    the pruning is recorded in the expert-sparsity config while the model
    config is left untouched."""
    torch.manual_seed(42)
    config = FakeMoEConfig(
        num_experts=8, hidden_size=16, intermediate_size=32, num_hidden_layers=2
    )
    model = FakeMoEModel(config)
    model.eval()

    modifier = RazorPruningModifier(sparsity=0.5)
    state = make_state(model)

    modifier.initialize(state)
    assert modifier._n_experts_to_drop == 4

    modifier.update_event(state, Event(type_=EventType.CALIBRATION_START))
    assert len(modifier._moe_stats) == 2

    with torch.no_grad():
        for _ in range(5):
            model(torch.randn(2, 8, config.hidden_size))

    modifier.update_event(state, Event(type_=EventType.SEQUENTIAL_EPOCH_END))
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
def test_razor_distributed_uneven_data():
    """Layers must be reduced and pruned identically on all ranks when
    calibration data is unevenly distributed across ranks.

    Every rank reduces every tracked layer (contributing zeros where it saw
    no tokens) and the decision of which layers collected samples is made
    from the total count summed across ranks, so the conditional RMS pools
    the squared damage and counts across ranks rather than averaging
    per-rank scores.
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

            modifier = RazorPruningModifier(sparsity=0.5)
            state = make_state(model)

            modifier.initialize(state)
            modifier.update_event(state, Event(type_=EventType.CALIBRATION_START))

            # only data_rank processes calibration data
            if rank == data_rank:
                with torch.no_grad():
                    for _ in range(5):
                        model(torch.randn(2, 8, config.hidden_size))

            modifier.update_event(state, Event(type_=EventType.SEQUENTIAL_EPOCH_END))
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
