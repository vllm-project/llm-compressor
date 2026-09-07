from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch.fx import symbolic_trace
from transformers import LlamaConfig, LlamaForCausalLM

from llmcompressor.args.dataset_arguments import DatasetArguments
from llmcompressor.core import Event, EventType, State, create_session
from llmcompressor.modifiers import ModifierFactory
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.layerwise_qad import LayerwiseQADModifier
from llmcompressor.modifiers.qad import QADModifier
from llmcompressor.modifiers.qad.base import _masked_mse, _output_loss
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.pipelines.cache import IntermediatesCache
from llmcompressor.pipelines.sequential.helpers import Subgraph
from llmcompressor.pipelines.sequential.pipeline import SequentialPipeline
from llmcompressor.utils.helpers import DisableQuantization


class BranchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.left = torch.nn.Linear(32, 32, bias=False)
        self.right = torch.nn.Linear(32, 32, bias=False)
        self.down = torch.nn.Linear(32, 32, bias=False)

    def forward(self, x, residual):
        # Shared module, nonlinear functions, residual, multiple inputs/outputs.
        left = self.left(x).tanh()
        right = self.right(x).sigmoid()
        return {
            "hidden": self.down(left * right) + residual,
            "aux": (self.left(residual), None),
            "metadata": 3,
        }


def _subgraph(model):
    graph = symbolic_trace(model).graph
    return Subgraph(
        graph, {n.target for n in graph.nodes if n.op == "placeholder"}, set()
    )


def _quantizer(kind, scheme="NVFP4A16"):
    kwargs = dict(targets="Linear", scheme=scheme, ignore=["lm_head"])
    return (
        GPTQModifier(**kwargs, actorder="static")
        if kind == "gptq"
        else QuantizationModifier(**kwargs)
    )


def _prepare(kind="rtn", dtype=torch.float32, **qad_kwargs):
    torch.manual_seed(17)
    model = BranchModel().to(dtype)
    reference = deepcopy(model)
    subgraph = _subgraph(model)
    batches = [
        {
            "x": torch.randn(1, 4, 32, dtype=dtype),
            "residual": torch.randn(1, 4, 32, dtype=dtype),
        }
        for _ in range(6)
    ]
    cache = IntermediatesCache.from_dataloader(batches)
    state = State(model=model)
    quant = _quantizer(kind)
    qad = QADModifier(**qad_kwargs)
    quant.on_initialize(state)
    qad.on_initialize(state)
    start = Event(type_=EventType.CALIBRATION_START)
    quant.on_calibration_start(state, start)
    qad.on_calibration_start(
        state,
        start,
        subgraphs=[subgraph],
        dataset_args=SimpleNamespace(propagate_error=True),
    )
    return model, reference, subgraph, batches, cache, state, quant, qad


@pytest.mark.parametrize("kind", ["rtn", "gptq"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_quantizer_then_joint_qad_preserves_teacher_and_qparams(kind, dtype):
    model, reference, sg, batches, cache, state, quant, qad = _prepare(
        kind,
        dtype,
        num_epochs=3,
        learning_rate=0.003,
        gradient_accumulation_steps=2,
        max_grad_norm=0.1,
    )
    modules = sg.submodules(model)
    with DisableQuantization(model):
        qad.on_sequential_epoch_start(state, None, modules, sg, cache, 0)
        assert all(
            not m.quantization_enabled
            for m in modules
            if hasattr(m, "quantization_enabled")
        )
        assert not quant._hessians if kind == "gptq" else True
        # Teacher is from original weights, including the complete residual/branch.
        for batch, cached in zip(batches, qad._batches):
            expected = reference(**batch)
            torch.testing.assert_close(cached.target["hidden"], expected["hidden"])
            torch.testing.assert_close(cached.target["aux"][0], expected["aux"][0])
            assert cached.target["metadata"] == 3
        for batch in batches:
            model(**batch)
        quant.on_sequential_epoch_end(state, None, modules)
        qparams = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if "scale" in name or "zero_point" in name
        }
        with patch(
            "torch.nn.utils.clip_grad_norm_", wraps=torch.nn.utils.clip_grad_norm_
        ) as clip:
            qad.on_sequential_epoch_end(state, None, modules)
        assert clip.call_count == qad.optimizer_steps["subgraph_0"]
        assert qad.optimizer_steps["subgraph_0"] == 9  # 5 train batches, groups 2/2/1
        assert qad._graph is None and not qad._batches
        for name, expected in qparams.items():
            torch.testing.assert_close(
                model.get_parameter(name), expected, rtol=0, atol=0
            )
        assert all(torch.isfinite(p.float()).all() for p in model.parameters())
        assert all(p.grad is None for p in model.parameters())
        assert (
            qad.best_validation_losses["subgraph_0"]
            <= qad.validation_histories["subgraph_0"][0]
        )
        if kind == "gptq":
            assert not quant._hessians and not quant._num_samples


@pytest.mark.parametrize("kind", ["rtn", "gptq"])
@pytest.mark.parametrize(
    "targets,per_subgraph",
    [
        (["LlamaDecoderLayer"], 1),
        (["LlamaDecoderLayer"], 2),
        (["re:.*self_attn.q_proj", "re:.*mlp.down_proj"], 2),
    ],
)
def test_real_llama_sequential_pipeline(kind, targets, per_subgraph):
    torch.manual_seed(21)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=32,
        )
    )
    # The tracer traces attention in eager mode. Keep runtime mask generation
    # consistent when targets split inside attention instead of at decoder blocks.
    model.config._attn_implementation = "eager"
    data = [
        {
            "input_ids": torch.randint(0, 64, (1, 8)),
            "attention_mask": torch.ones(1, 8, dtype=torch.long),
            "loss_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]]),
        }
        for _ in range(4)
    ]
    quant = _quantizer(kind)
    qad = QADModifier(num_epochs=1, learning_rate=0.001)
    args = DatasetArguments(
        pipeline="sequential",
        sequential_targets=targets,
        sequential_targets_per_subgraph=per_subgraph,
        use_loss_mask=True,
    )
    with create_session() as session:
        session.initialize(model=model, recipe=[quant, qad], start=-1)
        SequentialPipeline()(model, data, args)
        actual = next(
            m for m in session.lifecycle.recipe.modifiers if isinstance(m, QADModifier)
        )
        assert actual.optimizer_steps
        assert all(steps == 3 for steps in actual.optimizer_steps.values())
        assert actual._graph is None and not actual._batches
        session.finalize()
    with torch.no_grad():
        assert torch.isfinite(model(input_ids=data[0]["input_ids"]).logits).all()


def test_teacher_cache_is_independent_and_masked():
    model, reference, sg, batches, cache, state, quant, qad = _prepare()
    state.loss_masks = [torch.tensor([[1, 1, 0, 0]]) for _ in batches]
    with DisableQuantization(model):
        qad.on_sequential_epoch_start(state, None, sg.submodules(model), sg, cache, 0)
    saved_input = qad._batches[0].inputs["x"].clone()
    saved_target = qad._batches[0].target["hidden"].clone()
    batches[0]["x"].zero_()
    with torch.no_grad():
        model.left.weight.zero_()
    torch.testing.assert_close(qad._batches[0].inputs["x"], saved_input)
    torch.testing.assert_close(qad._batches[0].target["hidden"], saved_target)
    prediction = saved_target.clone()
    prediction[:, 2:] += 100
    assert _masked_mse(prediction, saved_target, qad._batches[0].loss_mask) == 0


def test_partial_accumulation_matches_large_batch():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    qad = QADModifier(gradient_accumulation_steps=2, max_grad_norm=None)
    qad._batches = [1.0, 3.0, 5.0]
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    with patch.object(
        qad,
        "_batch_loss",
        side_effect=lambda target: (parameter - target).square().sum(),
    ):
        steps = qad._train_epoch(optimizer, [parameter], [parameter], [0, 1, 2])
    assert steps == 2
    # First mean gradient = -2, w=1.2; final single-batch gradient=-7.6, w=1.96.
    torch.testing.assert_close(parameter, torch.tensor([1.96]))


@pytest.mark.parametrize(
    "losses,patience,expected_epoch,best_epoch",
    [
        ([0.5, 0.4, 0.6], 1, 2, 1),
        ([1.0, 0.9995, 0.9989, 1.0, 1.0, 1.0], 3, 5, 2),
        ([0.1, 0.2], 1, 1, 0),
    ],
)
def test_early_stopping_restores_best_including_initial(
    losses, patience, expected_epoch, best_epoch
):
    parameter = torch.nn.Parameter(torch.zeros(1))
    qad = QADModifier(num_epochs=10, early_stopping_patience=patience)
    qad._name = "test"
    epoch = 0

    def train(*args):
        nonlocal epoch
        epoch += 1
        with torch.no_grad():
            parameter.fill_(epoch)
        return 1

    with (
        patch.object(qad, "_evaluate", side_effect=losses),
        patch.object(qad, "_train_epoch", side_effect=train),
    ):
        steps, epochs, best = qad._train_with_validation(
            None, [parameter], [parameter], [0], [1]
        )
    assert epochs == steps == expected_epoch
    assert best == min(losses)
    assert parameter.item() == best_epoch
    assert qad.validation_histories["test"] == losses


def test_output_loss_uses_all_float_leaves():
    pred = {"a": torch.ones(1, 2, 4), "b": (torch.full((1, 2, 4), 2.0), None)}
    target = {"a": torch.zeros(1, 2, 4), "b": (torch.zeros(1, 2, 4), None)}
    assert _output_loss(pred, target, None) == 2.5
    with pytest.raises(ValueError, match="valid tokens"):
        _output_loss(pred, target, torch.zeros(1, 2))


def test_factory_and_old_name_compatibility():
    ModifierFactory.refresh()
    for name in ["QADModifier", "LayerwiseQADModifier"]:
        modifier = ModifierFactory.create(
            name, allow_registered=True, allow_experimental=True
        )
        assert isinstance(modifier, QADModifier)
    assert LayerwiseQADModifier is QADModifier


def test_requires_preceding_quantizer_and_sequential_pipeline():
    qad = QADModifier()
    state = State(model=BranchModel())
    with pytest.raises(ValueError, match="after a weight quantization"):
        qad.on_initialize(state)
    with pytest.raises(ValueError, match="sequential"):
        qad.on_calibration_start(state, None)
    with pytest.raises(ValueError, match="propagate_error"):
        qad.on_calibration_start(
            state,
            None,
            subgraphs=[],
            dataset_args=SimpleNamespace(propagate_error=False),
        )


def test_empty_subgraph_is_skipped_and_cross_stage_sharing_rejected():
    model, reference, sg, batches, cache, state, quant, qad = _prepare()
    qad.on_sequential_epoch_start(state, None, [], sg, cache, 0)
    qad.on_sequential_epoch_end(state, None, [])
    assert not qad.optimizer_steps
    with pytest.raises(ValueError, match="shared across subgraphs"):
        qad.on_calibration_start(
            state,
            None,
            subgraphs=[sg, sg],
            dataset_args=SimpleNamespace(propagate_error=True),
        )


def test_validation_split():
    qad = QADModifier(seed=17)
    train, valid = qad._split_batch_indices(512)
    assert len(train) == 460 and len(valid) == 52
    assert set(train).isdisjoint(valid)
    assert (train, valid) == qad._split_batch_indices(512)
    with pytest.raises(ValueError, match="at least two"):
        qad._split_batch_indices(1)


@pytest.mark.parametrize("kind", ["rtn", "gptq"])
def test_sequential_pipeline_without_qad(kind):
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
    )
    data = [{"input_ids": torch.randint(0, 64, (1, 8))} for _ in range(2)]
    args = DatasetArguments(sequential_targets=["LlamaDecoderLayer"])
    with create_session() as session:
        session.initialize(model=model, recipe=[_quantizer(kind)], start=-1)
        SequentialPipeline()(model, data, args)
        session.finalize()
    with torch.no_grad():
        assert torch.isfinite(model(**data[0]).logits).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_joint_training_reduces_reconstruction_error(dtype):
    model, reference, sg, batches, cache, state, quant, qad = _prepare(
        dtype=dtype,
        num_epochs=15,
        learning_rate=0.003,
        early_stopping_patience=15,
    )
    modules = sg.submodules(model)
    with DisableQuantization(model):
        qad.on_sequential_epoch_start(state, None, modules, sg, cache, 0)
        for batch in batches:
            model(**batch)
        quant.on_sequential_epoch_end(state, None, modules)
        with torch.no_grad():
            # Deliberately introduce reconstruction error without changing qparams.
            # QAD should correct all three jointly using the original teacher.
            for module in (model.left, model.right, model.down):
                module.weight.add_(0.02)
        before = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
            if name.endswith(".weight")
        }
        qad.on_sequential_epoch_end(state, None, modules)
    history = qad.validation_histories["subgraph_0"]
    assert min(history[1:]) < history[0] * 0.8
    for name, value in before.items():
        assert not torch.equal(model.get_parameter(name), value)
