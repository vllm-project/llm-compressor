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
def test_local_teacher_uses_final_upstream_outputs(kind):
    torch.manual_seed(67)
    model = _tiny_llama().eval()
    reference = deepcopy(model)
    data = [{"input_ids": torch.randint(0, 64, (1, 8))} for _ in range(3)]
    qad = QADModifier(learning_rate=0.001)
    original_optimize = QADModifier._optimize_block
    # Observe actual propagation without running an extra whole-model forward.
    propagated = []
    seen = []

    def first_block_output(module, args, output):
        if HooksMixin._HOOKS_DISABLED and not torch.is_grad_enabled():
            propagated.append(
                output.detach().clone()
                if isinstance(output, torch.Tensor)
                else output[0].detach().clone()
            )

    def optimize(self, modules):
        block_name = self._name
        ref_block = reference.get_submodule(block_name)
        with torch.no_grad():
            for batch in self._batches:
                torch.testing.assert_close(
                    batch.target, ref_block(*batch.args, **batch.kwargs)
                )
        if block_name == "model.layers.1":
            # The last N calls of block 0 are the pipeline's propagation pass,
            # after QAD validation and final weight materialization.
            for batch, output in zip(self._batches, propagated[-len(data) :]):
                hidden = batch.args[0] if batch.args else batch.kwargs["hidden_states"]
                torch.testing.assert_close(
                    hidden, output.to(hidden.device), rtol=0, atol=0
                )
        seen.append(block_name)
        original_optimize(self, modules)

    handle = model.model.layers[0].register_forward_hook(first_block_output)
    try:
        with patch.object(QADModifier, "_optimize_block", optimize):
            with create_session() as session:
                session.initialize(
                    model=model, recipe=[_quantizer(kind), qad], start=-1
                )
                SequentialPipeline()(model, data, DatasetArguments())
                session.finalize()
    finally:
        handle.remove()
    assert seen == ["model.layers.0", "model.layers.1"]


def test_teacher_capture_does_not_run_extra_forwards():
    model, _, batches, state, quant, qad = _prepare()
    with patch.object(model.block, "forward", wraps=model.block.forward) as forward:
        with DisableQuantization(model):
            _calibrate(model, batches, state)
        assert forward.call_count == len(batches)
        assert len(qad._captured[model.block]) == len(batches)
    qad.on_finalize(state)


def test_teacher_capture_requires_sequential_batch_context():
    model, _, batches, state, quant, qad = _prepare()
    with DisableQuantization(model):
        with pytest.raises(ValueError, match="sequential"):
            model(**batches[0])
    assert not qad._hooks and not qad._captured


def test_missing_weight_qparams_rejected_before_optimization():
    model, _, batches, state, quant, qad = _prepare()
    with DisableQuantization(model):
        _calibrate(model, batches, state)
        # Simulate a preceding method that has not prepared its qparams.
        model.block.left.weight_scale = None
        with pytest.raises(ValueError, match="initialized weight_scale"):
            qad.on_sequential_epoch_end(state, None, list(model.block.modules()))
    assert not qad._hooks and not qad._captured
