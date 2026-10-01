from copy import deepcopy
from unittest.mock import patch

import pytest
import torch

from llmcompressor.args.dataset_arguments import DatasetArguments
from llmcompressor.core import create_session
from llmcompressor.modifiers.qad import QADModifier
from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.pipelines.sequential import SequentialPipeline
from llmcompressor.utils.helpers import DisableQuantization

from .test_base import _calibrate, _prepare, _quantizer, _tiny_llama


@pytest.mark.parametrize("kind", ["rtn", "gptq"])
@pytest.mark.parametrize("per_subgraph", [1, 2])
def test_local_teacher_uses_final_upstream_outputs(kind, per_subgraph):
    torch.manual_seed(67)
    model = _tiny_llama(layers=2 * per_subgraph).eval()
    reference = deepcopy(model)
    data = [{"input_ids": torch.randint(0, 64, (1, 8))} for _ in range(3)]
    qad = QADModifier(lr=0.001)
    original_optimize = QADModifier._apply_distillation
    # Observe actual propagation without running an extra whole-model forward.
    propagated = []
    seen = []

    def first_subgraph_output(module, args, output):
        if HooksMixin._HOOKS_DISABLED and not torch.is_grad_enabled():
            propagated.append(
                output.detach().clone()
                if isinstance(output, torch.Tensor)
                else output[0].detach().clone()
            )

    def optimize(self, blocks, train, validation):
        names = [self._module_names[block] for block in blocks]
        ref_blocks = [reference.get_submodule(name) for name in names]
        with torch.no_grad():
            for batch in self._iter_batches(train + validation):
                # The teacher is the unquantized chain run from the first input
                output = ref_blocks[0](*batch["args"], **batch["kwargs"])
                for ref_block, inputs in zip(ref_blocks[1:], batch["links"]):
                    output = ref_block(output, *inputs["args"][1:], **inputs["kwargs"])
                torch.testing.assert_close(batch["target"], output)
            if kind == "rtn":
                # RTN leaves weights unchanged, so the chain with quantization
                # still disabled reproduces the teacher exactly
                assert self._evaluate(blocks, train + validation) == 0
        if names[0] == f"model.layers.{per_subgraph}":
            # The last N calls of the first subgraph's last block are the
            # pipeline's propagation pass, after QAD validation and final weight
            # materialization.
            outputs = propagated[-len(data) :]
            indices = sum(self._split_batch_indices(len(data)), [])
            for index, batch in zip(indices, self._iter_batches(train + validation)):
                hidden = (
                    batch["args"][0]
                    if batch["args"]
                    else batch["kwargs"]["hidden_states"]
                )
                torch.testing.assert_close(
                    hidden, outputs[index].to(hidden.device), rtol=0, atol=0
                )
        seen.append(self._name(blocks))
        original_optimize(self, blocks, train, validation)

    last = model.model.layers[per_subgraph - 1]
    handle = last.register_forward_hook(first_subgraph_output)
    try:
        with patch.object(QADModifier, "_apply_distillation", optimize):
            with create_session() as session:
                session.initialize(
                    model=model, recipe=[_quantizer(kind), qad], start=-1
                )
                args = DatasetArguments(sequential_targets_per_subgraph=per_subgraph)
                SequentialPipeline()(model, data, args)
                session.finalize()
    finally:
        handle.remove()
    if per_subgraph == 1:
        assert seen == ["model.layers.0", "model.layers.1"]
    else:
        assert seen == [
            "model.layers.0..model.layers.1",
            "model.layers.2..model.layers.3",
        ]


def test_teacher_capture_does_not_run_extra_forwards():
    model, _, batches, state, quant, qad = _prepare()
    with patch.object(model.block, "forward", wraps=model.block.forward) as forward:
        with DisableQuantization(model):
            _calibrate(model, batches, state)
        assert forward.call_count == len(batches)
        assert len(qad._block_cache[model.block]) == len(batches)
    qad.on_finalize(state)


def test_teacher_capture_requires_sequential_batch_context():
    model, _, batches, state, quant, qad = _prepare()
    with DisableQuantization(model):
        with pytest.raises(ValueError, match="sequential"):
            model(**batches[0])


def test_missing_weight_qparams_rejected_before_optimization():
    model, _, batches, state, quant, qad = _prepare()
    with DisableQuantization(model):
        _calibrate(model, batches, state)
        # Simulate a preceding method that has not prepared its qparams.
        model.block.left.weight_scale = None
        with pytest.raises(ValueError, match="initialized weight_scale"):
            qad.on_sequential_epoch_end(state, None, list(model.block.modules()))
