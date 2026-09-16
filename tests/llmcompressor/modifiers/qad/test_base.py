from copy import deepcopy
from unittest.mock import patch

import pytest
import torch
from pydantic import ValidationError
from transformers import LlamaConfig, LlamaForCausalLM

from llmcompressor.args.dataset_arguments import DatasetArguments
from llmcompressor.core import Event, EventType, State, create_session
from llmcompressor.modifiers import ModifierFactory
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.layerwise_qad import LayerwiseQADModifier
from llmcompressor.modifiers.qad import QADModifier
from llmcompressor.modifiers.qad.base import _masked_mse, _output_loss
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq import AWQModifier
from llmcompressor.pipelines.sequential.pipeline import SequentialPipeline
from llmcompressor.utils.dev import get_main_device
from llmcompressor.utils.helpers import DisableQuantization


class BranchBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.left = torch.nn.Linear(32, 32, bias=False)
        self.right = torch.nn.Linear(32, 32, bias=False)
        self.down = torch.nn.Linear(32, 32, bias=False)

    def forward(self, x, residual):
        left = self.left(x).tanh()
        right = self.right(x).sigmoid()
        return {
            "hidden": self.down(left * right) + residual,
            "aux": (self.left(residual), None),
            "metadata": 3,
        }


class BranchModel(torch.nn.Module):
    _no_split_modules = ["BranchBlock"]

    def __init__(self):
        super().__init__()
        self.block = BranchBlock()

    def forward(self, x, residual):
        # Exercise both positional and keyword inputs to the captured module.
        return self.block(x, residual=residual)


def _quantizer(kind, scheme="NVFP4A16"):
    kwargs = (
        dict(
            config_groups={"group_0": dict(scheme, targets=["Linear"])},
            ignore=["lm_head"],
        )
        if isinstance(scheme, dict)
        else dict(targets="Linear", scheme=scheme, ignore=["lm_head"])
    )
    return (
        GPTQModifier(**kwargs, actorder="static")
        if kind == "gptq"
        else QuantizationModifier(**kwargs)
    )


def _prepare(kind="rtn", dtype=torch.float32, **qad_kwargs):
    torch.manual_seed(17)
    model = BranchModel().to(dtype)
    reference = deepcopy(model)
    batches = [
        {
            "x": torch.randn(1, 4, 32, dtype=dtype),
            "residual": torch.randn(1, 4, 32, dtype=dtype),
        }
        for _ in range(6)
    ]
    state = State(model=model)
    quant = _quantizer(kind)
    qad = QADModifier(**qad_kwargs)
    quant.on_initialize(state)
    qad.on_initialize(state)
    start = Event(type_=EventType.CALIBRATION_START)
    quant.on_calibration_start(state, start)
    qad.on_calibration_start(state, start)
    return model, reference, batches, state, quant, qad


def _calibrate(model, batches, state):
    with torch.no_grad():
        for index, batch in enumerate(batches):
            state.current_batch_idx = index
            model(**batch)


def _tiny_llama(layers=2):
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=layers,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=32,
        )
    )
    model.config._attn_implementation = "eager"
    return model


@pytest.mark.parametrize("kind", ["rtn", "gptq"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_quantizer_then_qad_preserves_teacher_qparams_and_hooks(kind, dtype):
    model, reference, batches, state, quant, qad = _prepare(
        kind,
        dtype,
        num_epochs=3,
        learning_rate=0.003,
        gradient_accumulation_steps=2,
        max_grad_norm=0.1,
        reobserve_weights=False,
    )
    modules = list(model.block.modules())
    with DisableQuantization(model):
        _calibrate(model, batches, state)
        for batch, cached in zip(batches, qad._captured[model.block]):
            torch.testing.assert_close(cached.target, reference(**batch))
            torch.testing.assert_close(cached.args[0], batch["x"])
            torch.testing.assert_close(cached.kwargs["residual"], batch["residual"])
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
        assert clip.call_count == qad.optimizer_steps["block"] == 9
        assert qad._block is None and not qad._batches and not qad._captured
        for name, expected in qparams.items():
            torch.testing.assert_close(
                model.get_parameter(name), expected, rtol=0, atol=0
            )
        assert all(torch.isfinite(p.float()).all() for p in model.parameters())
        assert all(p.grad is None for p in model.parameters())
        assert (
            qad.best_validation_losses["block"] <= qad.validation_histories["block"][0]
        )
        if kind == "gptq":
            assert not quant._hessians and not quant._num_samples
    qad.on_calibration_end(state, None)
    assert not qad._hooks


@pytest.mark.parametrize("kind", ["rtn", "gptq", "awq"])
@pytest.mark.parametrize("scheme", ["NVFP4A16", "W4A16"])
def test_real_llama_sequential_pipeline(kind, scheme):
    if kind == "awq" and not torch.accelerator.is_available():
        pytest.skip("AWQ calibration requires an accelerator for pinned memory")
    torch.manual_seed(21)
    model = _tiny_llama().to(get_main_device())
    data = [
        {
            "input_ids": torch.randint(0, 64, (1, 8)),
            "attention_mask": torch.ones(1, 8, dtype=torch.long),
            "loss_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]]),
        }
        for _ in range(4)
    ]
    # W4A16's default group size is 128; use a tiny-model-compatible group.
    if scheme == "W4A16":
        scheme = {
            "weights": {
                "num_bits": 4,
                "type": "int",
                "symmetric": True,
                "strategy": "group",
                "group_size": 16,
            }
        }
    qad = QADModifier(num_epochs=1, learning_rate=0.001)
    args = DatasetArguments(
        sequential_targets=["LlamaDecoderLayer"],
        sequential_targets_per_subgraph=1,
        use_loss_mask=True,
    )
    with create_session() as session:
        recipe = [AWQModifier(n_grid=4)] if kind == "awq" else []
        recipe.extend([_quantizer(kind, scheme), qad])
        session.initialize(model=model, recipe=recipe, start=-1)
        SequentialPipeline()(model, data, args)
        assert qad.optimizer_steps == {"model.layers.0": 3, "model.layers.1": 3}
        assert not qad._captured and not qad._hooks
        session.finalize()
    with torch.no_grad():
        ids = data[0]["input_ids"].to(model.device)
        assert torch.isfinite(model(input_ids=ids).logits).all()


def test_teacher_cache_is_independent_and_masked():
    model, reference, batches, state, quant, qad = _prepare()
    state.loss_masks = [torch.tensor([[1, 1, 0, 0]]) for _ in batches]
    with DisableQuantization(model):
        _calibrate(model, batches, state)
    stored = qad._captured[model.block][0]
    saved_input, saved_target = stored.args[0].clone(), stored.target["hidden"].clone()
    batches[0]["x"].zero_()
    state.loss_masks[0].zero_()
    with torch.no_grad():
        model.block.left.weight.zero_()
    torch.testing.assert_close(stored.args[0], saved_input)
    torch.testing.assert_close(stored.target["hidden"], saved_target)
    prediction = saved_target.clone()
    prediction[:, 2:] += 100
    assert _masked_mse(prediction, saved_target, stored.loss_mask) == 0
    qad.on_finalize(state)
    assert not qad._captured and not qad._hooks


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
        steps = qad._train_epoch(
            optimizer,
            [parameter],
            [parameter],
            [0, 1, 2],
            torch.amp.GradScaler("cpu", enabled=False),
        )
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
    qad = QADModifier(
        num_epochs=10, early_stopping_patience=patience, reobserve_weights=False
    )
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


def test_validation_split():
    qad = QADModifier(seed=17)
    train, valid = qad._split_batch_indices(512)
    assert len(train) == 460 and len(valid) == 52
    assert set(train).isdisjoint(valid)
    assert (train, valid) == qad._split_batch_indices(512)
    with pytest.raises(ValueError, match="at least two"):
        qad._split_batch_indices(1)


def test_requires_preceding_quantizer_and_rejects_removed_teacher_option():
    with pytest.raises(ValueError, match="after a weight quantization"):
        QADModifier().on_initialize(State(model=BranchModel()))
    with pytest.raises(ValidationError):
        QADModifier(teacher_mode="full")


def test_cross_block_shared_weights_rejected():
    model = _tiny_llama()
    model.model.layers[1].self_attn.q_proj.weight = model.model.layers[
        0
    ].self_attn.q_proj.weight
    state = State(model=model)
    _quantizer("rtn").on_initialize(state)
    with pytest.raises(ValueError, match="shared across blocks"):
        QADModifier().on_initialize(state)


def test_multiple_blocks_per_stage_rejected_and_hooks_cleaned():
    model = _tiny_llama()
    data = [{"input_ids": torch.randint(0, 64, (1, 8))} for _ in range(2)]
    qad = QADModifier()
    with create_session() as session:
        session.initialize(model=model, recipe=[_quantizer("rtn"), qad], start=-1)
        with pytest.raises(ValueError, match="one target module per subgraph"):
            SequentialPipeline()(
                model, data, DatasetArguments(sequential_targets_per_subgraph=2)
            )
        assert not qad._hooks and not qad._captured


def test_training_failure_releases_cache_and_restores_grad_flags():
    model, _, batches, state, quant, qad = _prepare()
    flags = {name: p.requires_grad for name, p in model.named_parameters()}
    with DisableQuantization(model):
        _calibrate(model, batches, state)
        modules = list(model.block.modules())
        quant.on_sequential_epoch_end(state, None, modules)
        with patch.object(
            qad, "_train_with_validation", side_effect=RuntimeError("failed")
        ):
            with pytest.raises(RuntimeError, match="failed"):
                qad.on_sequential_epoch_end(state, None, modules)
    assert not qad._hooks and not qad._captured and not qad._pending
    assert qad._block is None and not qad._batches
    for name, flag in flags.items():
        assert model.get_parameter(name).requires_grad == flag


def test_repeated_block_call_rejected():
    model, _, batches, state, quant, qad = _prepare()
    state.current_batch_idx = 0
    with DisableQuantization(model), torch.no_grad():
        model(**batches[0])
        with pytest.raises(ValueError, match="once per calibration batch"):
            model(**batches[0])
    assert not qad._hooks and not qad._captured


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_joint_training_reduces_reconstruction_error_with_fixed_scales(dtype):
    model, reference, batches, state, quant, qad = _prepare(
        dtype=dtype,
        num_epochs=15,
        learning_rate=0.003,
        early_stopping_patience=15,
        reobserve_weights=False,
    )
    modules = list(model.block.modules())
    with DisableQuantization(model):
        _calibrate(model, batches, state)
        quant.on_sequential_epoch_end(state, None, modules)
        with torch.no_grad():
            for module in (model.block.left, model.block.right, model.block.down):
                module.weight.add_(0.02)
        before = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
            if name.endswith(".weight")
        }
        qad.on_sequential_epoch_end(state, None, modules)
    history = qad.validation_histories["block"]
    assert min(history[1:]) < history[0] * 0.8
    for name, value in before.items():
        assert not torch.equal(model.get_parameter(name), value)
    qad.on_finalize(state)
