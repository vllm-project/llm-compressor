import torch

from llmcompressor.pipelines.sequential.helpers import get_sequential_ancestors
from llmcompressor.args import DatasetArguments


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = torch.nn.Sequential(torch.nn.Linear(10, 20), torch.nn.ReLU())
        self.fc = torch.nn.Linear(20, 5)

    def forward(self, x):
        x = self.seq(x)
        return self.fc(x)


def test_get_sequential_ancestors():
    model = DummyModel()

    assert get_sequential_ancestors(model, set()) == set()
    assert get_sequential_ancestors(model, {model}) == set()
    assert get_sequential_ancestors(model, {model.fc}) == {model}
    assert get_sequential_ancestors(model, {model.seq[0]}) == {model, model.seq}
    assert get_sequential_ancestors(model, {model.seq[1]}) == {model, model.seq}


def test_disable_quantization_during_calibration_config():
    """Test that the disable_quantization_during_calibration configuration works correctly."""
    # Test default value (should be False)
    dataset_args = DatasetArguments()
    assert dataset_args.disable_quantization_during_calibration is False
    
    # Test setting to True
    dataset_args = DatasetArguments(disable_quantization_during_calibration=True)
    assert dataset_args.disable_quantization_during_calibration is True
    
    # Test setting to False explicitly
    dataset_args = DatasetArguments(disable_quantization_during_calibration=False)
    assert dataset_args.disable_quantization_during_calibration is False
    
    # Test with other arguments
    dataset_args = DatasetArguments(
        dataset="test_dataset",
        disable_quantization_during_calibration=True
    )
    assert dataset_args.disable_quantization_during_calibration is True
    assert dataset_args.dataset == "test_dataset"


def test_disable_quantization_metadata():
    """Test that the configuration has proper metadata."""
    import inspect
    from llmcompressor.args.dataset_arguments import DatasetArguments
    
    # Get the field info
    fields = DatasetArguments.__dataclass_fields__
    field_info = fields.get('disable_quantization_during_calibration')
    
    assert field_info is not None
    assert field_info.default is False
    assert 'help' in field_info.metadata
    assert 'disable' in field_info.metadata['help'].lower()
    assert 'quantization' in field_info.metadata['help'].lower()
