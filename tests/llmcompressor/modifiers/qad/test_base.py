import re
from contextlib import contextmanager
from copy import deepcopy
from unittest.mock import patch

import pytest
import torch
from loguru import logger
from pydantic import ValidationError
from transformers import LlamaConfig, LlamaForCausalLM

from llmcompressor.args.dataset_arguments import DatasetArguments
from llmcompressor.core import Event, EventType, State, create_session
from llmcompressor.entrypoints.oneshot import Oneshot
from llmcompressor.modifiers import ModifierFactory
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.qad import QADModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq import AWQModifier
from llmcompressor.pipelines.cache import IntermediatesCache, IntermediateValue
from llmcompressor.pipelines.sequential.pipeline import SequentialPipeline
from llmcompressor.utils.dev import get_main_device
from llmcompressor.utils.helpers import DisableQuantization


class BranchSeqTarget(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.left = torch.nn.Linear(32, 32, bias=False)
        self.right = torch.nn.Linear(32, 32, bias=False)
        self.down = torch.nn.Linear(32, 32, bias=False)

    def forward(self, x, residual):
        left = self.left(x).tanh()
        right = self.right(x).sigmoid()
        # Only the first output is distilled, like decoder layers' hidden states
        return self.down(left * right) + residual, self.left(residual)


class BranchModel(torch.nn.Module):
    _no_split_modules = ["BranchSeqTarget"]

    def __init__(self):
        super().__init__()
        self.seq_target = BranchSeqTarget()

    def forward(self, x, residual):
        # Exercise both positional and keyword inputs to the captured module.
        return self.seq_target(x, residual=residual)


class ChainModel(torch.nn.Module):
    _no_split_modules = ["BranchSeqTarget"]

    def __init__(self, parallel=False):
        super().__init__()
        self.first = BranchSeqTarget()
        self.second = BranchSeqTarget()
        self.parallel = parallel

    def forward(self, x, residual):
        hidden, _ = self.first(x, residual=residual)
        # In parallel, the second sequential target does not take the first one's output
        return self.second(x if self.parallel else hidden, residual=residual)


@contextmanager
def _capture_logs():
    logs = []
    handler_id = logger.add(logs.append, format="{message}", level="INFO")
    try:
        yield logs
    finally:
        logger.remove(handler_id)


def _logged_updates(logs):
    """Target name -> optimizer updates, from QAD's per-target summary log"""
    matches = (re.match(r"QAD (\S+): (\d+) updates", log) for log in logs)
    return {match[1]: int(match[2]) for match in matches if match}


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


def _prepare(kind="rtn", dtype=torch.float32, model_class=BranchModel, **qad_kwargs):
    torch.manual_seed(17)
    model = model_class().to(dtype)
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


def _target_batches(targets):
    cache = IntermediatesCache()
    for target in targets:
        cache.append({"target": target})
    return cache.batch_intermediates


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
        lr=0.003,
        gradient_accumulation_steps=2,
        max_grad_norm=0.1,
        reobserve_weights=False,
    )
    modules = list(model.seq_target.modules())
    with DisableQuantization(model):
        _calibrate(model, batches, state)
        for index, batch in enumerate(batches):
            cached_input = qad._input_caches[model.seq_target].fetch(index)
            cached_output = qad._output_caches[model.seq_target].fetch(index)
            torch.testing.assert_close(cached_output["target"], reference(**batch))
            torch.testing.assert_close(cached_input["args"][0], batch["x"])
            torch.testing.assert_close(
                cached_input["kwargs"]["residual"], batch["residual"]
            )
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
        assert clip.call_count == 9
        assert not qad._input_caches and not qad._output_caches
        for name, expected in qparams.items():
            torch.testing.assert_close(
                model.get_parameter(name), expected, rtol=0, atol=0
            )
        assert all(torch.isfinite(p.float()).all() for p in model.parameters())
        assert all(p.grad is None for p in model.parameters())
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
    qad = QADModifier(num_epochs=1, lr=0.001)
    args = DatasetArguments(
        sequential_targets=["LlamaDecoderLayer"],
        sequential_targets_per_subgraph=1,
        sequential_prefetch=sequential_prefetch,
    )
    with create_session() as session:
        recipe = [AWQModifier(n_grid=4)] if kind == "awq" else []
        recipe.extend([_quantizer(kind, scheme), qad])
        session.initialize(model=model, recipe=recipe, start=-1)
        with _capture_logs() as logs:
            SequentialPipeline()(model, data, args)
        assert _logged_updates(logs) == {"model.layers.0": 3, "model.layers.1": 3}
        assert not qad._input_caches and not qad._output_caches and not qad._hooks
        session.finalize()
    with torch.no_grad():
        ids = data[0]["input_ids"].to(model.device)
        assert torch.isfinite(model(input_ids=ids).logits).all()


def test_teacher_cache_is_independent():
    model, reference, batches, state, quant, qad = _prepare()
    with DisableQuantization(model):
        _calibrate(model, batches, state)
    input_cache = qad._input_caches[model.seq_target]
    output_cache = qad._output_caches[model.seq_target]
    assert isinstance(input_cache, IntermediatesCache)
    assert isinstance(output_cache, IntermediatesCache)
    stored_input = input_cache.fetch(0)
    stored_output = output_cache.fetch(0)
    saved_input = stored_input["args"][0].clone()
    saved_target = stored_output["target"][0].clone()
    batches[0]["x"].zero_()
    with torch.no_grad():
        model.seq_target.left.weight.zero_()
    stored_input = input_cache.fetch(0)
    stored_output = output_cache.fetch(0)
    torch.testing.assert_close(stored_input["args"][0], saved_input)
    torch.testing.assert_close(stored_output["target"][0], saved_target)
    qad.on_finalize(state)
    assert not qad._input_caches and not qad._output_caches and not qad._hooks


def test_cache_replay_preserves_selected_batch_order():
    model, reference, batches, state, quant, qad = _prepare()
    with DisableQuantization(model):
        _calibrate(model, batches, state)
    input_entries = qad._input_caches[model.seq_target].batch_intermediates
    output_entries = qad._output_caches[model.seq_target].batch_intermediates
    entries = [
        {
            "args": input_entries[index]["args"],
            "kwargs": input_entries[index]["kwargs"],
            "links": IntermediateValue([], None),
            "target": output_entries[index]["target"],
        }
        for index in range(len(input_entries))
    ]
    indices = [4, 1, 5, 1]
    replayed = list(IntermediatesCache([entries[i] for i in indices]).iter_prefetch())
    assert len(replayed) == len(indices)
    for index, batch in zip(indices, replayed):
        torch.testing.assert_close(batch["args"][0], batches[index]["x"])
        torch.testing.assert_close(batch["target"], reference(**batches[index]))
    qad.on_finalize(state)
    assert not qad._input_caches and not qad._output_caches


def test_partial_accumulation_matches_large_batch():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    qad = QADModifier(gradient_accumulation_steps=2, max_grad_norm=None)
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    with patch.object(
        qad,
        "_batch_loss",
        side_effect=lambda _, batch: (parameter - batch["target"]).square().sum(),
    ):
        steps = qad._train_epoch(
            None,
            optimizer,
            [parameter],
            [parameter],
            _target_batches([1.0, 3.0, 5.0]),
            torch.amp.GradScaler("cpu", enabled=False),
        )
    assert steps == 2
    # First mean gradient = -2, w=1.2; final single-batch gradient=-7.6, w=1.96.
    torch.testing.assert_close(parameter, torch.tensor([1.96]))


@pytest.mark.parametrize(
    "losses,best_epoch",
    [
        ([0.5, 0.4, 0.6], 1),
        ([1.0, 0.9995, 0.9989, 1.0], 2),
        ([0.1, 0.2], 0),
    ],
)
def test_best_validation_weights_restored_including_initial(losses, best_epoch):
    module = torch.nn.Linear(1, 1, bias=False)
    parameter = module.weight
    parameter.data.zero_()
    qad = QADModifier(num_epochs=len(losses) - 1, reobserve_weights=False)
    qad._seq_target_names = {module: "test"}
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
        steps = qad._train_with_validation(
            [module], [module], None, [parameter], [parameter], [0], [1]
        )
    assert steps == len(losses) - 1
    assert parameter.item() == best_epoch


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_batch_loss_uses_first_tuple_output(dtype):
    hidden = torch.ones(1, 2, 4, dtype=dtype)
    target = torch.zeros_like(hidden)
    qad = QADModifier()
    batch = {"args": (), "kwargs": {}, "links": [], "target": target}
    loss = qad._batch_loss([lambda: hidden], batch)
    assert loss.dtype == torch.float32
    assert loss == 1
    # Trailing outputs, such as MoE top-k indices, are not distilled
    extra = torch.ones(1, 2, 2, dtype=torch.long)
    batch["target"] = (target, extra * 5)
    assert qad._batch_loss([lambda: (hidden, extra)], batch) == loss


def test_batch_loss_chains_hidden_states():
    qad = QADModifier()
    batch = {
        "args": (torch.ones(2),),
        "kwargs": {},
        # A later target keeps its other inputs; its first input is recomputed
        "links": [{"args": (None, torch.full((2,), 3.0)), "kwargs": {"scale": 2}}],
        "target": torch.full((2,), 10.0),
    }
    first = lambda x: (x + 1, None)  # noqa: E731
    second = lambda hidden, shift, scale: hidden * scale + shift  # noqa: E731
    # (1 + 1) * 2 + 3 = 7, so each element misses the target by 3
    assert qad._batch_loss([first, second], batch) == 9


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
    assert not qad._hooks and not qad._input_caches and not qad._output_caches


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
    with create_session(), _capture_logs() as logs:
        entrypoint.apply_recipe_modifiers(calibration_dataloader=data)
    assert _logged_updates(logs) == {"model.layers.0": 1}
    assert not qad._hooks and not qad._input_caches and not qad._output_caches


@pytest.mark.parametrize("propagate_error", [True, False])
def test_oneshot_qad_with_and_without_error_propagation(propagate_error, tmp_path):
    model = _tiny_llama().to(get_main_device())
    model.config.save_pretrained(tmp_path)
    model.config.name_or_path = str(tmp_path)
    qad = QADModifier(num_epochs=1)
    data = [{"input_ids": torch.randint(0, 64, (1, 8))} for _ in range(2)]
    with patch("llmcompressor.entrypoints.oneshot.pre_process"):
        entrypoint = Oneshot(
            model=model,
            recipe=[_quantizer("rtn"), qad],
            pipeline="sequential",
            propagate_error=propagate_error,
        )
    with create_session(), _capture_logs() as logs:
        entrypoint.apply_recipe_modifiers(calibration_dataloader=data)
    assert _logged_updates(logs) == {"model.layers.0": 1, "model.layers.1": 1}
    assert not qad._hooks and not qad._input_caches and not qad._output_caches


def test_quantized_weights_outside_targets_keep_quantizer_result(tmp_path):
    model = _tiny_llama().to(get_main_device())
    model.config.save_pretrained(tmp_path)
    model.config.name_or_path = str(tmp_path)
    lm_head = model.lm_head.weight.detach().clone()
    qad = QADModifier(num_epochs=1)
    data = [{"input_ids": torch.randint(0, 64, (1, 8))} for _ in range(2)]
    # Quantize the LM head too; it lies outside every sequential target.
    quantizer = QuantizationModifier(targets="Linear", scheme="NVFP4A16")
    with patch("llmcompressor.entrypoints.oneshot.pre_process"):
        entrypoint = Oneshot(
            model=model, recipe=[quantizer, qad], pipeline="sequential"
        )
    with create_session(), _capture_logs() as logs:
        entrypoint.apply_recipe_modifiers(calibration_dataloader=data)
    assert _logged_updates(logs) == {"model.layers.0": 1, "model.layers.1": 1}
    assert model.lm_head.quantization_scheme.weights is not None
    torch.testing.assert_close(model.lm_head.weight, lm_head)


@pytest.mark.parametrize(
    "per_subgraph,expected",
    [
        (2, {"model.layers.0..model.layers.1": 3, "model.layers.2..model.layers.3": 3}),
        (3, {"model.layers.0..model.layers.2": 3, "model.layers.3": 3}),
    ],
)
@pytest.mark.parametrize("sequential_prefetch", [False, True])
def test_subgraph_targets_train_jointly(per_subgraph, expected, sequential_prefetch):
    torch.manual_seed(23)
    model = _tiny_llama(layers=4).to(get_main_device())
    data = [{"input_ids": torch.randint(0, 64, (1, 8))} for _ in range(4)]
    qad = QADModifier(num_epochs=1, lr=0.001)
    args = DatasetArguments(
        sequential_targets_per_subgraph=per_subgraph,
        sequential_prefetch=sequential_prefetch,
    )
    with create_session() as session:
        session.initialize(model=model, recipe=[_quantizer("rtn"), qad], start=-1)
        with _capture_logs() as logs:
            SequentialPipeline()(model, data, args)
        assert _logged_updates(logs) == expected
        assert (
            not qad._input_caches
            and not qad._output_caches
            and not qad._predecessors
            and not qad._hooks
        )
        session.finalize()


def test_chain_caches_only_last_teacher_and_trains_every_target():
    model, reference, batches, state, quant, qad = _prepare(
        model_class=ChainModel, num_epochs=2, lr=0.003, reobserve_weights=False
    )
    modules = list(model.modules())
    with DisableQuantization(model):
        _calibrate(model, batches, state)
        assert qad._predecessors == {model.first: None, model.second: model.first}
        for index, batch in enumerate(batches):
            first_output = qad._output_caches[model.first].fetch(index)
            second_input = qad._input_caches[model.second].fetch(index)
            second_output = qad._output_caches[model.second].fetch(index)
            assert "target" not in first_output
            # The first sequential target's output is recomputed rather than cached
            assert second_input["args"] == (None,)
            torch.testing.assert_close(second_output["target"], reference(**batch))
        quant.on_sequential_epoch_end(state, None, modules)
        # The first sequential target trains only through the second target's output
        norms = []
        model.first.left.weight.register_hook(lambda grad: norms.append(grad.norm()))
        with _capture_logs() as logs:
            qad.on_sequential_epoch_end(state, None, modules)
    # 5 training batches per epoch; 1 is held out for validation
    assert _logged_updates(logs) == {"first..second": 10}
    assert len(norms) == 10 and all(norm > 0 for norm in norms)
    qad.on_calibration_end(state, None)


def test_unchained_subgraph_targets_rejected():
    model, _, batches, state, quant, qad = _prepare(
        model_class=lambda: ChainModel(parallel=True)
    )
    modules = list(model.modules())
    with DisableQuantization(model):
        _calibrate(model, batches, state)
        quant.on_sequential_epoch_end(state, None, modules)
        with pytest.raises(ValueError, match="previous target's output"):
            qad.on_sequential_epoch_end(state, None, modules)
    qad.on_finalize(state)


def test_training_failure_restores_grad_flags():
    model, _, batches, state, quant, qad = _prepare()
    flags = {name: p.requires_grad for name, p in model.named_parameters()}
    with DisableQuantization(model):
        _calibrate(model, batches, state)
        modules = list(model.seq_target.modules())
        quant.on_sequential_epoch_end(state, None, modules)
        with patch.object(
            qad, "_train_with_validation", side_effect=RuntimeError("failed")
        ):
            with pytest.raises(RuntimeError, match="failed"):
                qad.on_sequential_epoch_end(state, None, modules)
    for name, flag in flags.items():
        assert model.get_parameter(name).requires_grad == flag


def test_repeated_seq_target_call_rejected():
    model, _, batches, state, quant, qad = _prepare()
    state.current_batch_idx = 0
    with DisableQuantization(model), torch.no_grad():
        model(**batches[0])
        with pytest.raises(ValueError, match="once per calibration batch"):
            model(**batches[0])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_joint_training_reduces_reconstruction_error_with_fixed_scales(dtype):
    model, reference, batches, state, quant, qad = _prepare(
        dtype=dtype,
        num_epochs=15,
        lr=0.003,
        reobserve_weights=False,
    )
    modules = list(model.seq_target.modules())
    with DisableQuantization(model):
        _calibrate(model, batches, state)
        quant.on_sequential_epoch_end(state, None, modules)
        with torch.no_grad():
            for module in (
                model.seq_target.left,
                model.seq_target.right,
                model.seq_target.down,
            ):
                module.weight.add_(0.02)
        before = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
            if name.endswith(".weight")
        }
        with _capture_logs() as logs:
            qad.on_sequential_epoch_end(state, None, modules)
    pattern = r"QAD seq_target (?:initial|epoch \d+) validation MSE ([^,\s]+)"
    matches = (re.match(pattern, log) for log in logs)
    initial, *epochs = [float(match[1]) for match in matches if match]
    assert min(epochs) < initial * 0.8
    for name, value in before.items():
        assert not torch.equal(model.get_parameter(name), value)
    qad.on_finalize(state)
