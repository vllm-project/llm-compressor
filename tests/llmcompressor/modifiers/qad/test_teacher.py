from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from pydantic import ValidationError
from transformers import LlamaConfig, LlamaForCausalLM

from llmcompressor.args.dataset_arguments import DatasetArguments
from llmcompressor.core import create_session
from llmcompressor.modifiers.qad import QADModifier
from llmcompressor.modifiers.qad.base import (
    _distillation_graph,
    _map_tensors,
    _output_loss,
    _quantized_modules,
)
from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.pipelines.cache import IntermediatesCache
from llmcompressor.pipelines.sequential import SequentialPipeline
from llmcompressor.utils.helpers import DisableQuantization

from .test_base import _prepare, _quantizer


@pytest.mark.parametrize("kind", ["rtn", "gptq"])
@pytest.mark.parametrize("teacher_mode", ["local", "full"])
@pytest.mark.parametrize("separate_data", [False, True])
@pytest.mark.parametrize("structure", ["blocks", "unquantized_gap", "fine_grained"])
def test_teacher_matches_pristine_model_or_local_inputs(
    kind, teacher_mode, separate_data, structure
):
    torch.manual_seed(67)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=3,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=32,
        )
    ).eval()
    model.config._attn_implementation = "eager"
    reference = deepcopy(model)  # Test oracle only; production never copies weights.

    def make_data(count, length):
        return [
            {
                "input_ids": torch.randint(0, 64, (1, length)),
                "attention_mask": torch.ones(1, length, dtype=torch.long),
                "loss_mask": torch.tensor([[1] * (length - 1) + [0]]),
            }
            for _ in range(count)
        ]

    primary = make_data(3, 8)
    auxiliary = make_data(4, 6) if separate_data else primary
    reference_cache = IntermediatesCache.from_dataloader(deepcopy(auxiliary))
    quant = _quantizer(kind)
    if structure == "unquantized_gap":
        quant = type(quant)(
            targets="Linear",
            scheme="NVFP4A16",
            ignore=["lm_head", "re:model.layers.1.*"],
            **({"actorder": "static"} if kind == "gptq" else {}),
        )
    qad = QADModifier(teacher_mode=teacher_mode, learning_rate=0.001)
    args = DatasetArguments(
        sequential_targets=(
            ["re:.*self_attn.q_proj", "re:.*mlp.down_proj"]
            if structure == "fine_grained"
            else ["LlamaDecoderLayer"]
        ),
        use_loss_mask=True,
    )
    original_start = QADModifier.on_sequential_epoch_start
    differences = []
    seen = []

    def start(self, state, event, modules, **kwargs):
        sg = kwargs["subgraph"]
        names = {id(module): name for name, module in state.model.named_modules()}
        ref_modules = [
            reference.get_submodule(names[id(module)])
            for module in _quantized_modules(modules)
        ]
        ref_graph = (
            _distillation_graph(reference, sg, ref_modules) if ref_modules else None
        )
        expected = []
        with torch.no_grad():
            for index in range(len(reference_cache)):
                inputs = reference_cache.fetch(index, sg.input_names)
                if ref_graph is not None:
                    expected.append(ref_graph(**inputs))
                # Advance a separate, entirely untouched reference model using
                # the original pipeline subgraph, including unquantized gaps.
                output = sg.forward(reference, **inputs)
                reference_cache.update(index, output)
                reference_cache.delete(index, sg.consumed_names)
        original_start(self, state, event, modules, **kwargs)
        if self._graph is None:
            return
        seen.append(kwargs["subgraph_index"])
        with torch.no_grad(), HooksMixin.disable_hooks():
            for index, batch in enumerate(self._batches):
                student_cache = (
                    kwargs["additional_activations"]["qad"]
                    if separate_data
                    else kwargs["activations"]
                )
                torch.testing.assert_close(
                    batch.inputs, student_cache.fetch(index, batch.inputs.keys())
                )
                local_target = self._graph(**batch.inputs)
                torch.testing.assert_close(
                    batch.target,
                    expected[index] if teacher_mode == "full" else local_target,
                )
                # Measure disagreement using the same loss as training, without
                # requiring any particular output container structure.
                differences.append(_output_loss(local_target, expected[index], None))

    with patch.object(QADModifier, "on_sequential_epoch_start", start):
        with create_session() as session:
            session.initialize(model=model, recipe=[quant, qad], start=-1)
            SequentialPipeline()(
                model,
                primary,
                args,
                additional_dataloaders={"qad": auxiliary} if separate_data else None,
            )
            assert len(seen) >= 2
            if structure == "unquantized_gap":
                assert len(seen) == 2
            assert qad._teacher_activations is None
            assert qad._graph is None and not qad._batches
            session.finalize()
    assert differences[0] < 1e-10  # Before any upstream quantization, modes agree.
    assert max(differences) > 1e-8  # Later local/full targets actually differ.


def test_full_teacher_multi_output_graph_and_cache_cleanup():
    model, reference, sg, batches, cache, state, quant, qad = _prepare(
        teacher_mode="full"
    )
    with DisableQuantization(model):
        qad.on_sequential_epoch_start(state, None, sg.submodules(model), sg, cache, 0)
    for batch, stored in zip(batches, qad._batches):
        torch.testing.assert_close(stored.target, reference(**batch))
    assert qad._teacher_activations is None  # Last stage releases the stream.
    qad.on_finalize(state)
    assert not qad._batches and qad._graph is None


def test_full_teacher_cache_does_not_alias_student_and_cleans_up_on_error():
    model, reference, sg, batches, cache, state, quant, qad = _prepare(
        teacher_mode="full"
    )
    qad._prepare_teacher_stream(cache, 0)
    expected = _map_tensors(cache.fetch(0), lambda t: t.clone())
    batches[0]["x"].zero_()
    torch.testing.assert_close(qad._teacher_activations.fetch(0), expected)
    with patch(
        "llmcompressor.modifiers.qad.base.GraphModule.__call__",
        side_effect=RuntimeError("teacher forward failed"),
    ):
        with pytest.raises(RuntimeError, match="teacher forward failed"):
            qad.on_sequential_epoch_start(
                state, None, sg.submodules(model), sg, cache, 0
            )
    assert qad._teacher_activations is None
    assert qad._graph is None and not qad._batches


def test_teacher_mode_validation_and_order():
    with pytest.raises(ValidationError):
        QADModifier(teacher_mode="unknown")
    qad = QADModifier(teacher_mode="full")
    with pytest.raises(ValueError, match="all subgraphs in order"):
        qad._prepare_teacher_stream(IntermediatesCache.empty(2, "cpu"), 1)


def test_full_teacher_rejects_transforms_that_change_future_weights():
    from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier

    model, reference, sg, batches, cache, state, quant, qad = _prepare(
        teacher_mode="full"
    )
    session = SimpleNamespace(
        lifecycle=SimpleNamespace(
            recipe=SimpleNamespace(modifiers=[SmoothQuantModifier(), quant, qad])
        )
    )
    with patch("llmcompressor.modifiers.qad.base.active_session", return_value=session):
        with pytest.raises(ValueError, match="smoothing or rotation"):
            qad.on_initialize(state)
