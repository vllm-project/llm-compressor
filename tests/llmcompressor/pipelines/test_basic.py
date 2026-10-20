import torch
from torch.utils.data import DataLoader

from llmcompressor.args.dataset_arguments import DatasetArguments
from llmcompressor.core import create_session
from llmcompressor.pipelines.basic import BasicPipeline, run_calibration


class BranchedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2, bias=False)

    def forward(self, x, loss_mask=None):
        if x.sum() < 0:
            x = -x
        if loss_mask is not None:
            x = x * loss_mask
        return self.linear(x)


def test_basic_pipeline_preserves_untraced_calibration():
    model = BranchedModel()
    mask = torch.zeros(1, 2)
    samples = [{"x": -torch.ones(1, 2)}, {"x": torch.ones(1, 2), "loss_mask": mask}]
    args = DatasetArguments(sequential_targets=["Linear"], use_loss_mask=True)
    outputs = []

    with (
        model.register_forward_hook(
            lambda _module, _args, output: outputs.append(output)
        ),
        create_session() as session,
    ):
        session.initialize(model=model, start=-1)
        BasicPipeline()(model, DataLoader(samples, batch_size=None), args)
        assert len(outputs) == len(samples)
        assert torch.count_nonzero(outputs[1]) == 0
        assert session.state.loss_masks[0] is None
        assert torch.equal(session.state.loss_masks[1].cpu(), mask)
        session.finalize()

    assert args.sequential_targets == ["Linear"]
    assert args.propagate_error is True


def test_run_calibration_without_dataset_arguments():
    model = BranchedModel()
    dataloader = DataLoader([{"x": -torch.ones(1, 2)}], batch_size=None)
    outputs = []

    with (
        model.register_forward_hook(
            lambda _module, _args, output: outputs.append(output)
        ),
        create_session() as session,
    ):
        session.initialize(model=model, start=-1)
        run_calibration(model, dataloader)
        assert len(outputs) == 1
        session.finalize()
