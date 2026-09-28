from collections import namedtuple
from copy import deepcopy
from unittest.mock import patch

import pytest
import torch
from compressed_tensors.offload import offload_module
from pydantic import ValidationError
from transformers import LlamaConfig, LlamaForCausalLM

from llmcompressor.args.dataset_arguments import DatasetArguments
from llmcompressor.core import Event, EventType, State, create_session
from llmcompressor.entrypoints.oneshot import Oneshot
from llmcompressor.modifiers import ModifierFactory
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.qad import QADModifier
from llmcompressor.modifiers.qad.base import _output_loss
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq import AWQModifier
from llmcompressor.pipelines.cache import IntermediatesCache
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


def _target_cache(targets):
    cache = IntermediatesCache()
    for target in targets:
        cache.append({"target": target})
    return cache


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
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
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
            torch.testing.assert_close(cached["target"], reference(**batch))
            torch.testing.assert_close(cached["args"][0], batch["x"])
            torch.testing.assert_close(cached["kwargs"]["residual"], batch["residual"])
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
@pytest.mark.parametrize("sequential_prefetch", [False, True])
def test_real_llama_sequential_pipeline(kind, scheme, sequential_prefetch):
    if kind == "awq" and not torch.accelerator.is_available():
        pytest.skip("AWQ calibration requires an accelerator for pinned memory")
    torch.manual_seed(21)
    model = _tiny_llama().to(get_main_device())
    data = [
        {
            "input_ids": torch.randint(0, 64, (1, 8)),
            "attention_mask": torch.ones(1, 8, dtype=torch.long),
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
        sequential_prefetch=sequential_prefetch,
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


def test_teacher_cache_is_independent():
    model, reference, batches, state, quant, qad = _prepare()
    with DisableQuantization(model):
        _calibrate(model, batches, state)
    cache = qad._captured[model.block]
    assert isinstance(cache, IntermediatesCache)
    stored = cache.fetch(0)
    saved_input = stored["args"][0].clone()
    saved_target = stored["target"]["hidden"].clone()
    batches[0]["x"].zero_()
    with torch.no_grad():
        model.block.left.weight.zero_()
    stored = cache.fetch(0)
    torch.testing.assert_close(stored["args"][0], saved_input)
    torch.testing.assert_close(stored["target"]["hidden"], saved_target)
    qad.on_finalize(state)
    assert not qad._captured and not qad._hooks


@pytest.mark.parametrize("sequential_prefetch", [False, True])
def test_cache_replay_preserves_selected_batch_order(sequential_prefetch):
    model, reference, batches, state, quant, qad = _prepare()
    with DisableQuantization(model):
        _calibrate(model, batches, state)
    qad._batches = qad._captured[model.block]
    qad._sequential_prefetch = sequential_prefetch
    indices = [4, 1, 5, 1]
    replayed = list(qad._iter_batches(indices))
    assert len(replayed) == len(indices)
    for index, batch in zip(indices, replayed):
        torch.testing.assert_close(batch["args"][0], batches[index]["x"])
        torch.testing.assert_close(batch["target"], reference(**batches[index]))
    qad.on_finalize(state)
    assert not qad._batches and not qad._captured


@pytest.mark.parametrize("sequential_prefetch", [False, True])
def test_partial_accumulation_matches_large_batch(sequential_prefetch):
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    qad = QADModifier(gradient_accumulation_steps=2, max_grad_norm=None)
    qad._batches = _target_cache([1.0, 3.0, 5.0])
    qad._sequential_prefetch = sequential_prefetch
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    with patch.object(
        qad,
        "_batch_loss",
        side_effect=lambda batch: (parameter - batch["target"]).square().sum(),
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_output_loss_uses_all_float_leaves(dtype):
    output = namedtuple("Output", ["hidden", "metadata"])
    first = torch.ones(1, 2, 4, dtype=dtype, requires_grad=True)
    second = torch.full((1, 3, 2), 2.0, dtype=dtype, requires_grad=True)
    pred = {
        "a": first,
        "b": (second, None),
        "c": output(torch.empty(0), torch.tensor([1])),
        "metadata": "student",
    }
    target = {
        "a": torch.zeros_like(first),
        "b": (torch.zeros_like(second), None),
        "c": output(torch.empty(0), torch.tensor([2])),
        "metadata": "teacher",
    }
    loss = _output_loss(pred, target)
    assert loss.dtype == torch.float32
    assert loss == 2.5
    grads = torch.autograd.grad(loss, [first, second])
    torch.testing.assert_close(grads[0], torch.full_like(first, 1 / first.numel()))
    torch.testing.assert_close(grads[1], torch.full_like(second, 2 / second.numel()))


@pytest.mark.parametrize(
    "prediction,target",
    [
        ([0], (0,)),
        ({"a": 0}, {"b": 0}),
        ((0,), (0, 0)),
        ({"a": [0]}, {"a": 0}),
        (None, torch.ones(1)),
    ],
)
def test_output_loss_rejects_mismatched_structures(prediction, target):
    with pytest.raises(ValueError, match="output structures differ"):
        _output_loss(prediction, target)


def test_output_loss_rejects_mismatched_tensor_shapes():
    with pytest.raises(ValueError, match="output shapes differ"):
        _output_loss({"hidden": torch.ones(2)}, {"hidden": torch.zeros(3)})


def test_factory():
    ModifierFactory.refresh()
    modifier = ModifierFactory.create(
        "QADModifier", allow_registered=True, allow_experimental=True
    )
    assert isinstance(modifier, QADModifier)


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


@pytest.mark.parametrize(
    "distributed,world_size", [(False, None), (True, 1), (True, 2)]
)
def test_qad_requires_single_process_calibration(distributed, world_size):
    state = State(model=BranchModel())
    _quantizer("rtn").on_initialize(state)
    qad = QADModifier()
    with (
        patch(
            "llmcompressor.modifiers.qad.base.is_distributed", return_value=distributed
        ),
        patch("torch.distributed.get_world_size", return_value=world_size) as get_size,
    ):
        if world_size == 2:
            with pytest.raises(ValueError, match="single-process calibration"):
                qad.on_initialize(state)
        else:
            assert qad.on_initialize(state)
    if not distributed:
        get_size.assert_not_called()


@pytest.mark.parametrize("pipeline", ["independent", "basic", "datafree", "default"])
def test_oneshot_rejects_nonsequential_qad_before_calibration(pipeline, tmp_path):
    model = _tiny_llama()
    model.config.save_pretrained(tmp_path)
    model.config.name_or_path = str(tmp_path)
    qad = QADModifier()
    pipeline_kwargs = {} if pipeline == "default" else {"pipeline": pipeline}
    with patch("llmcompressor.entrypoints.oneshot.pre_process"):
        entrypoint = Oneshot(
            model=model, recipe=[_quantizer("rtn"), qad], **pipeline_kwargs
        )
    with create_session(), patch.object(model, "forward") as forward:
        with pytest.raises(ValueError, match='requires pipeline="sequential"'):
            entrypoint.apply_recipe_modifiers(calibration_dataloader=[])
    forward.assert_not_called()
    assert not qad._hooks and not qad._captured


@pytest.mark.parametrize("pipeline", ["sequential", "SEQUENTIAL", None])
def test_oneshot_accepts_sequential_qad(pipeline, tmp_path):
    model = _tiny_llama(layers=1).to(get_main_device())
    model.config.save_pretrained(tmp_path)
    model.config.name_or_path = str(tmp_path)
    qad = QADModifier(num_epochs=1)
    data = [{"input_ids": torch.randint(0, 64, (1, 8))} for _ in range(2)]
    with patch("llmcompressor.entrypoints.oneshot.pre_process"):
        entrypoint = Oneshot(
            model=model, recipe=[_quantizer("rtn"), qad], pipeline=pipeline
        )
    with create_session():
        entrypoint.apply_recipe_modifiers(calibration_dataloader=data)
    assert qad.optimizer_steps == {"model.layers.0": 1}
    assert not qad._hooks and not qad._captured


def test_oneshot_rejects_qad_without_error_propagation_before_calibration(tmp_path):
    model = _tiny_llama()
    model.config.save_pretrained(tmp_path)
    model.config.name_or_path = str(tmp_path)
    qad = QADModifier()
    with patch("llmcompressor.entrypoints.oneshot.pre_process"):
        entrypoint = Oneshot(
            model=model,
            recipe=[_quantizer("rtn"), qad],
            pipeline="sequential",
            propagate_error=False,
        )
    with create_session(), patch.object(model, "forward") as forward:
        with pytest.raises(ValueError, match="requires propagate_error=True"):
            entrypoint.apply_recipe_modifiers(calibration_dataloader=[])
    forward.assert_not_called()
    assert not qad._hooks and not qad._captured


@pytest.mark.parametrize("offloaded", [False, True])
def test_cross_block_shared_weights_rejected(offloaded):
    model = _tiny_llama()
    model.model.layers[1].self_attn.q_proj.weight = model.model.layers[
        0
    ].self_attn.q_proj.weight
    state = State(model=model)
    _quantizer("rtn").on_initialize(state)
    if offloaded:
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                offload_module(module, onload_device="meta", offload_device="cpu")
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
    assert not qad._hooks and not qad._captured
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
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
