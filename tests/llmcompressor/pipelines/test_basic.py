from copy import deepcopy

import pytest
import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from torch.utils.data import DataLoader
from transformers import LlamaConfig, LlamaForCausalLM

from llmcompressor.args.dataset_arguments import DatasetArguments
from llmcompressor.core import LifecycleCallbacks, create_session
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.pipelines.basic import BasicPipeline, run_calibration
from llmcompressor.pytorch.utils.helpers import tensors_to_device
from llmcompressor.utils.dev import get_main_device
from llmcompressor.utils.helpers import DisableQuantization, calibration_forward_context


class BranchedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2, bias=False)
        torch.nn.init.eye_(self.linear.weight)

    def forward(self, x, loss_mask=None, **kwargs):
        if x.sum() < 0:
            x = -x
        if loss_mask is not None:
            x = x * loss_mask
        return self.linear(x) * kwargs.get("scale", 1)


def test_basic_pipeline_preserves_untraced_calibration():
    model = BranchedModel()
    mask = torch.zeros(1, 2)
    samples = [
        {"x": -torch.ones(1, 2), "loss_mask": torch.ones(1, 2), "scale": 1},
        {"x": 2 * torch.ones(1, 2), "loss_mask": mask, "scale": 1},
        {"x": 3 * torch.ones(1, 2), "loss_mask": torch.ones(1, 2), "scale": 0},
    ]
    order = [2, 0, 1]
    dataloader = DataLoader(samples, batch_size=None, sampler=order)
    args = DatasetArguments(
        sequential_targets=["Linear"],
        sequential_targets_per_subgraph=4,
        log_sequential_error=True,
        use_loss_mask=True,
    )
    original_args = deepcopy(args)
    pre_calls = []
    outputs = []

    with (
        model.register_forward_pre_hook(
            lambda _module, _args, kwargs: pre_calls.append(kwargs["x"].cpu().clone()),
            with_kwargs=True,
        ),
        model.register_forward_hook(
            lambda _module, _args, output: outputs.append(output)
        ),
        create_session() as session,
    ):
        session.initialize(model=model, start=-1)
        BasicPipeline()(model, dataloader, args)
        assert len(pre_calls) == len(outputs) == len(samples)
        for received, index in zip(pre_calls, order):
            torch.testing.assert_close(received, samples[index]["x"])
        assert torch.count_nonzero(outputs[0]) == 0
        torch.testing.assert_close(outputs[1], torch.ones_like(outputs[1]))
        assert torch.count_nonzero(outputs[2]) == 0
        for received, index in zip(session.state.loss_masks, order):
            torch.testing.assert_close(received.cpu(), samples[index]["loss_mask"])
        session.finalize()

    assert args == original_args


@pytest.mark.parametrize("num_batches", [1, 3])
def test_run_calibration_without_dataset_arguments(num_batches):
    model = BranchedModel()
    samples = [{"x": -torch.ones(1, 2)} for _ in range(num_batches)]
    dataloader = DataLoader(samples, batch_size=None)
    outputs = []

    with (
        model.register_forward_hook(
            lambda _module, _args, output: outputs.append(output)
        ),
        create_session() as session,
    ):
        session.initialize(model=model, start=-1)
        run_calibration(model, dataloader)
        assert len(outputs) == num_batches
        session.finalize()


def test_basic_pipeline_quantization_matches_direct_calibration():
    """The root subgraph must match the original basic calibration loop."""
    device = get_main_device()
    model = (
        LlamaForCausalLM(
            LlamaConfig(
                vocab_size=64,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=4,
                max_position_embeddings=32,
            )
        )
        .eval()
        .to(device)
    )
    reference = deepcopy(model)
    samples = [
        {"input_ids": torch.tensor([[1, 2, 3, 4]]), "attention_mask": torch.ones(1, 4)},
        {
            "input_ids": torch.tensor([[5, 6, 7, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0]]),
        },
    ]

    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(num_bits=8, strategy="channel"),
        input_activations=QuantizationArgs(
            num_bits=8, strategy="tensor", dynamic=False
        ),
    )
    for candidate in (reference, model):
        with create_session() as session:
            session.initialize(
                model=candidate,
                recipe=QuantizationModifier(config_groups={"group": deepcopy(scheme)}),
                start=-1,
            )
            dataloader = DataLoader(samples, batch_size=None)
            if candidate is reference:
                LifecycleCallbacks.calibration_start()
                with calibration_forward_context(candidate), DisableQuantization(
                    candidate
                ):
                    for batch_index, batch in enumerate(dataloader):
                        session.state.current_batch_idx = batch_index
                        candidate(**tensors_to_device(batch, device))
                LifecycleCallbacks.sequential_epoch_end(list(candidate.modules()))
                LifecycleCallbacks.calibration_end()
            else:
                BasicPipeline()(candidate, dataloader, DatasetArguments())
            session.finalize()

    expected = reference.state_dict()
    actual = model.state_dict()
    assert expected.keys() == actual.keys()
    assert any(name.endswith("input_scale") for name in expected)
    for name, value in expected.items():
        torch.testing.assert_close(actual[name].cpu(), value.cpu(), rtol=0, atol=0)
    inputs = tensors_to_device(samples[0], device)
    with torch.no_grad():
        torch.testing.assert_close(model(**inputs).logits, reference(**inputs).logits)
