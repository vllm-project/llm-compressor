import os
from typing import Dict, Iterable

import pytest
import torch
from torch import Tensor
from torch.nn import Linear, Module, ReLU, Sequential

from llmcompressor.pytorch.utils import (
    tensor_sparsity,
    tensors_module_forward,
    tensors_to_device,
    tensors_to_precision,
)
from tests.testing_utils import requires_gpu


@pytest.mark.unit
@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "tensors",
    [
        (),
        [],
        {},
        torch.randn(1, 8, 16, 32),
        torch.randn(8, 8, 16, 32),
        (torch.randn(1, 8), torch.randn(8, 8)),
        [torch.randn(1, 8), torch.randn(8, 8)],
        {"key": torch.randn(1, 8), "key2": torch.randn(8, 8)},
        [[torch.randn(1, 8)], torch.randn(8, 8)],
    ],
)
def test_tensors_to_device_cpu(tensors):
    out = tensors_to_device(tensors, "cpu")

    if isinstance(out, Tensor):
        assert not out.is_cuda
    elif isinstance(out, Iterable):
        for tens in out:
            if isinstance(tens, Tensor):
                assert not tens.is_cuda
    elif isinstance(out, Dict):
        for key, tens in out.items():
            if isinstance(tens, Tensor):
                assert not tens.is_cuda


@pytest.mark.unit
@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@requires_gpu
@pytest.mark.parametrize(
    "tensors",
    [
        (),
        [],
        {},
        torch.randn(1, 8, 16, 32),
        torch.randn(8, 8, 16, 32),
        (torch.randn(1, 8), torch.randn(8, 8)),
        [torch.randn(1, 8), torch.randn(8, 8)],
        {"key": torch.randn(1, 8), "key2": torch.randn(8, 8)},
        [[torch.randn(1, 8)], torch.randn(8, 8)],
    ],
)
def test_tensors_to_device_cuda(tensors):
    out = tensors_to_device(tensors, "cuda")

    if isinstance(out, Tensor):
        assert out.is_cuda
    elif isinstance(out, Iterable):
        for tens in out:
            if isinstance(tens, Tensor):
                assert tens.is_cuda
    elif isinstance(out, Dict):
        for key, tens in out.items():
            if isinstance(tens, Tensor):
                assert tens.is_cuda


@pytest.mark.unit
@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "tensors",
    [
        (),
        [],
        {},
        torch.randn(1, 8, 16, 32),
        torch.randn(8, 8, 16, 32),
        (torch.randn(1, 8), torch.randn(8, 8)),
        [torch.randn(1, 8), torch.randn(8, 8)],
        {"key": torch.randn(1, 8), "key2": torch.randn(8, 8)},
        [[torch.randn(1, 8)], torch.randn(8, 8)],
    ],
)
def test_tensors_to_precision_full_cpu(tensors):
    out = tensors_to_precision(tensors, True)

    if isinstance(out, Tensor):
        assert out.dtype == torch.float32
    elif isinstance(out, Iterable):
        for tens in out:
            if isinstance(tens, Tensor):
                assert tens.dtype == torch.float32
    elif isinstance(out, Dict):
        for key, tens in out.items():
            if isinstance(tens, Tensor):
                assert tens.dtype == torch.float32


@pytest.mark.unit
@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "tensors",
    [
        (),
        [],
        {},
        torch.randn(1, 8, 16, 32),
        torch.randn(8, 8, 16, 32),
        (torch.randn(1, 8), torch.randn(8, 8)),
        [torch.randn(1, 8), torch.randn(8, 8)],
        {"key": torch.randn(1, 8), "key2": torch.randn(8, 8)},
        [[torch.randn(1, 8)], torch.randn(8, 8)],
    ],
)
def test_tensors_to_precision_half_cpu(tensors):
    out = tensors_to_precision(tensors, False)

    if isinstance(out, Tensor):
        assert out.dtype == torch.float16
    elif isinstance(out, Iterable):
        for tens in out:
            if isinstance(tens, Tensor):
                assert tens.dtype == torch.float16
    elif isinstance(out, Dict):
        for key, tens in out.items():
            if isinstance(tens, Tensor):
                assert tens.dtype == torch.float16


@pytest.mark.unit
@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "tensors",
    [
        (),
        [],
        {},
        torch.randn(1, 8, 16, 32),
        torch.randn(8, 8, 16, 32),
        (torch.randn(1, 8), torch.randn(8, 8)),
        [torch.randn(1, 8), torch.randn(8, 8)],
        {"key": torch.randn(1, 8), "key2": torch.randn(8, 8)},
        [[torch.randn(1, 8)], torch.randn(8, 8)],
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda availability")
def test_tensors_to_precision_full_cuda(tensors):
    tensors = tensors_to_device(tensors, "cuda")
    out = tensors_to_precision(tensors, True)

    if isinstance(out, Tensor):
        assert out.dtype == torch.float32
    elif isinstance(out, Iterable):
        for tens in out:
            if isinstance(tens, Tensor):
                assert tens.dtype == torch.float32
    elif isinstance(out, Dict):
        for key, tens in out.items():
            if isinstance(tens, Tensor):
                assert tens.dtype == torch.float32


@pytest.mark.unit
@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "tensors",
    [
        (),
        [],
        {},
        torch.randn(1, 8, 16, 32),
        torch.randn(8, 8, 16, 32),
        (torch.randn(1, 8), torch.randn(8, 8)),
        [torch.randn(1, 8), torch.randn(8, 8)],
        {"key": torch.randn(1, 8), "key2": torch.randn(8, 8)},
        [[torch.randn(1, 8)], torch.randn(8, 8)],
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda availability")
def test_tensors_to_precision_half_cuda(tensors):
    tensors = tensors_to_device(tensors, "cuda")
    out = tensors_to_precision(tensors, False)

    if isinstance(out, Tensor):
        assert out.dtype == torch.float16
    elif isinstance(out, Iterable):
        for tens in out:
            if isinstance(tens, Tensor):
                assert tens.dtype == torch.float16
    elif isinstance(out, Dict):
        for key, tens in out.items():
            if isinstance(tens, Tensor):
                assert tens.dtype == torch.float16


@pytest.mark.unit
@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
class SimpleModule(Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.fc1 = Linear(input_size, 16, bias=True)
        self.relu1 = ReLU()
        self.fc2 = Linear(16, 32, bias=True)
        self.relu2 = ReLU()

    def forward(self, inp):
        out = self.fc1(inp)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)

        return out

    @staticmethod
    def example_input(batch_size: int, input_size: int):
        return torch.randn(batch_size, input_size)

    @staticmethod
    def example_output(batch_size: int):
        return torch.randn(batch_size, 32)


class ComplexModule(Module):
    def __init__(self, input_size_one: int, input_size_two: int):
        super().__init__()
        self.branch1 = Sequential(
            Linear(input_size_one, 16, bias=True), ReLU(), Linear(16, 32), ReLU()
        )
        self.branch2 = Sequential(
            Linear(input_size_two, 16, bias=True),
            ReLU(),
            Linear(16, 32, bias=True),
            ReLU(),
        )
        self.tower = Sequential(Linear(64, 32, bias=True), ReLU())

    def forward(self, inp_one, inp_two):
        out_one = self.branch1(inp_one)
        out_two = self.branch2(inp_two)
        out = torch.cat([out_one, out_two], dim=1)
        out = self.tower(out)

        return out

    @staticmethod
    def example_list_input(batch_size: int, input_size_one: int, input_size_two: int):
        return [
            torch.randn(batch_size, input_size_one),
            torch.randn(batch_size, input_size_two),
        ]

    @staticmethod
    def example_dict_input(batch_size: int, input_size_one: int, input_size_two: int):
        return {
            "inp_one": torch.randn(batch_size, input_size_one),
            "inp_two": torch.randn(batch_size, input_size_two),
        }

    @staticmethod
    def example_output(batch_size: int):
        return torch.randn(batch_size, 32)


@pytest.mark.unit
@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "module,tensors,check_feat_lab_inp",
    [
        (SimpleModule(8), SimpleModule.example_input(1, 8), False),
        (SimpleModule(8), SimpleModule.example_input(16, 8), False),
        (ComplexModule(8, 4), ComplexModule.example_list_input(1, 8, 4), False),
        (ComplexModule(8, 4), ComplexModule.example_list_input(16, 8, 4), False),
        (ComplexModule(8, 4), ComplexModule.example_dict_input(1, 8, 4), False),
        (ComplexModule(8, 4), ComplexModule.example_dict_input(16, 8, 4), False),
        (
            SimpleModule(8),
            (SimpleModule.example_input(1, 8), SimpleModule.example_output(1)),
            True,
        ),
        (
            SimpleModule(8),
            [SimpleModule.example_input(16, 8), SimpleModule.example_output(16)],
            True,
        ),
        (
            ComplexModule(8, 4),
            [
                ComplexModule.example_list_input(1, 8, 4),
                ComplexModule.example_output(1),
            ],
            True,
        ),
        (
            ComplexModule(8, 4),
            (
                ComplexModule.example_list_input(16, 8, 4),
                ComplexModule.example_output(16),
            ),
            True,
        ),
        (
            ComplexModule(8, 4),
            (
                ComplexModule.example_dict_input(1, 8, 4),
                ComplexModule.example_output(1),
            ),
            True,
        ),
        (
            ComplexModule(8, 4),
            [
                ComplexModule.example_dict_input(16, 8, 4),
                ComplexModule.example_output(16),
            ],
            True,
        ),
    ],
)
def test_tensors_module_forward(module, tensors, check_feat_lab_inp):
    out = tensors_module_forward(tensors, module, check_feat_lab_inp)
    assert len(out)


@pytest.mark.unit
@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@requires_gpu
@pytest.mark.parametrize(
    "module,tensors,check_feat_lab_inp",
    [
        (SimpleModule(8), SimpleModule.example_input(1, 8), False),
        (SimpleModule(8), SimpleModule.example_input(16, 8), False),
        (ComplexModule(8, 4), ComplexModule.example_list_input(1, 8, 4), False),
        (ComplexModule(8, 4), ComplexModule.example_list_input(16, 8, 4), False),
        (ComplexModule(8, 4), ComplexModule.example_dict_input(1, 8, 4), False),
        (ComplexModule(8, 4), ComplexModule.example_dict_input(16, 8, 4), False),
        (
            SimpleModule(8),
            (SimpleModule.example_input(1, 8), SimpleModule.example_output(1)),
            True,
        ),
        (
            SimpleModule(8),
            [SimpleModule.example_input(16, 8), SimpleModule.example_output(16)],
            True,
        ),
        (
            ComplexModule(8, 4),
            [
                ComplexModule.example_list_input(1, 8, 4),
                ComplexModule.example_output(1),
            ],
            True,
        ),
        (
            ComplexModule(8, 4),
            (
                ComplexModule.example_list_input(16, 8, 4),
                ComplexModule.example_output(16),
            ),
            True,
        ),
        (
            ComplexModule(8, 4),
            (
                ComplexModule.example_dict_input(1, 8, 4),
                ComplexModule.example_output(1),
            ),
            True,
        ),
        (
            ComplexModule(8, 4),
            [
                ComplexModule.example_dict_input(16, 8, 4),
                ComplexModule.example_output(16),
            ],
            True,
        ),
    ],
)
def test_tensors_module_forward_cuda(module, tensors, check_feat_lab_inp):
    module = module.to("cuda")
    tensors = tensors_to_device(tensors, "cuda")
    out = tensors_module_forward(tensors, module, check_feat_lab_inp)
    assert out is not None


@pytest.mark.unit
@pytest.mark.flaky(reruns=2, min_passes=1)
@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "tensor,dim,expected_sparsity",
    [
        (torch.zeros(8, 16), None, torch.tensor(1.0)),
        (torch.zeros(8, 16), 0, torch.ones(8)),
        (torch.zeros(8, 16), 1, torch.ones(16)),
        (torch.zeros(8, 16), [0, 1], torch.ones(8, 16)),
        (torch.zeros(8, 16), [1, 0], torch.ones(16, 8)),
        (torch.zeros(8, 16, 32, 8), [3, 1, 2], torch.ones(8, 16, 32)),
        (torch.ones(8, 16), None, torch.tensor(0.0)),
        (torch.ones(8, 16), 0, torch.zeros(8)),
        (torch.ones(8, 16), 1, torch.zeros(16)),
        (torch.ones(8, 16), [0, 1], torch.zeros(8, 16)),
        (torch.ones(8, 16), [1, 0], torch.zeros(16, 8)),
        (torch.ones(8, 16, 32, 8), [3, 1, 2], torch.zeros(8, 16, 32)),
        (torch.randn(8, 16), None, torch.tensor(0.0)),
        (torch.randn(8, 16), 0, torch.zeros(8)),
        (torch.randn(8, 16), 1, torch.zeros(16)),
        (torch.randn(8, 16), [0, 1], torch.zeros(8, 16)),
        (torch.randn(8, 16), [1, 0], torch.zeros(16, 8)),
        (torch.randn(8, 16, 32, 8), [3, 1, 2], torch.zeros(8, 16, 32)),
        (
            torch.tensor([10.0, 0.0, 1.0, 3.0, 2.0, 0.0, 8.0, 0.0, 5.0, 0.0]),
            None,
            torch.tensor(0.4),
        ),
    ],
)
def test_tensor_sparsity(tensor, dim, expected_sparsity):
    sparsity = tensor_sparsity(tensor, dim)
    assert expected_sparsity.shape == sparsity.shape
    assert torch.sum((sparsity - expected_sparsity).abs()) < 0.001


@pytest.mark.unit
@pytest.mark.flaky(reruns=2, min_passes=1)
@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@requires_gpu
@pytest.mark.parametrize(
    "tensor,dim,expected_sparsity",
    [
        (torch.zeros(8, 16), None, torch.tensor(1.0)),
        (torch.zeros(8, 16), 0, torch.ones(8)),
        (torch.zeros(8, 16, 32, 8), [3, 1, 2], torch.ones(8, 16, 32)),
        (torch.ones(8, 16), None, torch.tensor(0.0)),
        (torch.ones(8, 16), 0, torch.zeros(8)),
        (torch.ones(8, 16, 32, 8), [3, 1, 2], torch.zeros(8, 16, 32)),
        (torch.randn(8, 16), None, torch.tensor(0.0)),
        (torch.randn(8, 16), 0, torch.zeros(8)),
        (torch.randn(8, 16, 32, 8), [3, 1, 2], torch.zeros(8, 16, 32)),
        (
            torch.tensor([10.0, 0.0, 1.0, 3.0, 2.0, 0.0, 8.0, 0.0, 5.0, 0.0]),
            None,
            torch.tensor(0.4),
        ),
    ],
)
def test_tensor_sparsity_cuda(tensor, dim, expected_sparsity):
    tensor = tensor.to("cuda")
    sparsity = tensor_sparsity(tensor, dim)
    assert expected_sparsity.shape == sparsity.shape
    assert torch.sum((sparsity.detach().cpu() - expected_sparsity).abs()) < 0.001
