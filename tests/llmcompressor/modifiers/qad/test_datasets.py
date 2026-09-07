from unittest.mock import patch

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from llmcompressor.args.dataset_arguments import DatasetArguments
from llmcompressor.core import create_session
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.qad import QADModifier
from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.pipelines.sequential import SequentialPipeline

from .test_base import _quantizer


@pytest.mark.parametrize("kind", ["rtn", "gptq"])
@pytest.mark.parametrize("prefetch", [False, True])
@pytest.mark.parametrize("teacher_mode", ["local", "full"])
def test_independent_streams_teacher_masks_and_ptq_statistics(
    kind, prefetch, teacher_mode
):
    torch.manual_seed(57)
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
    model.config._attn_implementation = "eager"

    def data(count, batch_size, length, low, high):
        return [
            {
                "input_ids": torch.randint(low, high, (batch_size, length)),
                "attention_mask": torch.ones(batch_size, length, dtype=torch.long),
                "loss_mask": torch.tensor([[1] * (length - 2) + [0, 0]])
                .expand(batch_size, -1)
                .clone(),
            }
            for _ in range(count)
        ]

    primary = data(2, 1, 8, 0, 32)
    auxiliary = data(5, 2, 6, 32, 64)
    quant = _quantizer(kind)
    qad = QADModifier(num_epochs=1, learning_rate=0.001, teacher_mode=teacher_mode)
    args = DatasetArguments(
        sequential_targets=["LlamaDecoderLayer"],
        sequential_prefetch=prefetch,
        use_loss_mask=True,
    )
    pending = {}
    seen = []
    ptq_calls = []
    original_start = QADModifier.on_sequential_epoch_start
    original_end = QADModifier.on_sequential_epoch_end
    original_calibrate = GPTQModifier.calibrate_module

    def start(self, state, event, modules, **kwargs):
        index = kwargs["subgraph_index"]
        caches = {
            "ptq": kwargs["activations"],
            **kwargs["additional_activations"],
        }
        for name, cache in caches.items():
            for i, expected in enumerate(pending.get(name, [])):
                actual = cache.fetch(i, expected.keys())
                torch.testing.assert_close(actual, expected)
        original_start(self, state, event, modules, **kwargs)
        if self._graph is not None:
            seen.append(index)
            assert len(self._batches) == len(auxiliary)
            with HooksMixin.disable_hooks(), torch.no_grad():
                for i, batch in enumerate(self._batches):
                    # Current subgraph is still original; upstream cache already
                    # contains final quantized/QAD outputs, checked above.
                    inputs = caches["qad"].fetch(i, batch.inputs.keys())
                    torch.testing.assert_close(batch.inputs, inputs)
                    if teacher_mode == "local":
                        torch.testing.assert_close(batch.target, self._graph(**inputs))
                    torch.testing.assert_close(
                        batch.loss_mask, auxiliary[i]["loss_mask"]
                    )
        pending["caches"] = caches

    def end(self, state, event, modules, **kwargs):
        original_end(self, state, event, modules, **kwargs)
        sg = kwargs["subgraph"]
        # Independently compute the next inputs immediately after QAD. Both
        # streams must propagate these final weights, with hooks suppressed.
        with HooksMixin.disable_hooks(), torch.no_grad():
            for name, cache in pending.pop("caches").items():
                pending[name] = [
                    {
                        key: value.detach().clone()
                        if isinstance(value, torch.Tensor)
                        else value
                        for key, value in sg.forward(
                            state.model, **cache.fetch(i, sg.input_names)
                        ).items()
                    }
                    for i in range(len(cache))
                ]

    def calibrate(self, module, args, output):
        ptq_calls.append(tuple(args[0].shape[:2]))
        return original_calibrate(self, module, args, output)

    with (
        patch.object(QADModifier, "on_sequential_epoch_start", start),
        patch.object(QADModifier, "on_sequential_epoch_end", end),
        patch.object(GPTQModifier, "calibrate_module", calibrate),
        create_session() as session,
    ):
        session.initialize(model=model, recipe=[quant, qad], start=-1)
        SequentialPipeline()(
            model, primary, args, additional_dataloaders={"qad": auxiliary}
        )
        assert len(seen) == 2
        assert all(steps == 4 for steps in qad.optimizer_steps.values())
        if kind == "gptq":
            # Two calibration batches for each of seven linears in two layers.
            assert ptq_calls == [(1, 8)] * 28
        session.finalize()
