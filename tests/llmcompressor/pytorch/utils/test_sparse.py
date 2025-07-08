import pytest
import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from torch.nn import Linear, Module, ReLU

from llmcompressor.pytorch.utils import ModuleSparsificationInfo


class FakeQuantizedModel(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(8, 16, bias=True)  # Quantized
        self.fc2 = Linear(16, 4, bias=True)  # Unquantized
        self.relu = ReLU()

        self.fc1.quantization_scheme = QuantizationScheme(
            targets=["model.fc1"],
            weights=QuantizationArgs(
                precision=8,
                granularity="per_tensor",
                algorithm="gptq",
                blocksize=128,
            ),
        )


def test_module_quantization_info():
    model = FakeQuantizedModel()
    state = model.state_dict()

    # Simulate quantized weights: replace float32 weights with int8
    state["fc1.weight"] = torch.randint(
        -128, 127, state["fc1.weight"].shape, dtype=torch.int8
    )

    # Keep fc1.bias, fc2.weight, fc2.bias all as float32
    info = ModuleSparsificationInfo(model, state_dict=state)

    # fc1 (quantized): 8 * 16 weights + 16 biases = 144 parameters.
    # fc2 (not quantized): 16 * 4 weights + 4 biases = 68 parameters.
    # Total parameters: 144 + 68 = 212.
    # Quantized percentage: (144 / 212) * 100 ≈ 67.92%.
    percent = info.params_quantized_percent

    assert percent == pytest.approx(67.92, abs=1e-2)


class FakeSparsedModel(Module):
    def __init__(self):
        super().__init__()
        self.linear_dense = Linear(10, 10, bias=True)  # no sparsity
        self.linear_sparse = Linear(10, 10, bias=True)  # sparse layer

        # Inject sparsity into linear_sparse.weight (50% zeros)
        with torch.no_grad():
            weight = self.linear_sparse.weight
            weight.view(-1)[:50] = 0.0


def test_module_sparsity_info():
    model = FakeSparsedModel()
    state = model.state_dict()

    info = ModuleSparsificationInfo(model, state_dict=state)

    # linear_dense: 10 * 10 weights + 10 biases = 110 parameters.
    # linear_sparse: 10 * 10 weights + 10 biases = 110 parameters.
    # Total parameters: 110 + 110 = 220
    # Number of sparse (zero) parameters: 50 (from linear_sparse.weight).
    # Sparsity percentage: (50 / 220) * 100 ≈ 22.73%.
    percent = info.params_sparse_percent

    assert percent == pytest.approx(22.73, abs=1e-2)
