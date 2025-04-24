from collections import OrderedDict, namedtuple
from typing import List
from unittest.mock import Mock

import pytest
import torch
from torch import Tensor
from torch.nn import (
    AdaptiveAvgPool2d,
    Conv2d,
    Linear,
    Module,
    ReLU,
    Sequential,
    Sigmoid,
)
from torch.optim import SGD, Adam
from torch.utils.data import Dataset

__all__ = [
    "SAMPLE_STAGED_RECIPE",
    "test_epoch",
    "test_steps_per_epoch",
    "test_loss",
    "LinearNet",
    "MLPNet",
    "FlatMLPNet",
    "ConvNet",
    "QATMatMulTestNet",
    "MLPDataset",
    "ConvDataset",
    "create_optim_sgd",
    "create_optim_adam",
]


@pytest.fixture
def test_epoch() -> float:
    return 0.0


@pytest.fixture
def test_steps_per_epoch() -> int:
    return 100


@pytest.fixture
def test_loss() -> Tensor:
    return torch.tensor(0.0)


LayerDesc = namedtuple("LayerDesc", ["name", "input_size", "output_size", "bias"])


class LinearNet(Module):
    _LAYER_DESCS = None

    @staticmethod
    def layer_descs() -> List[LayerDesc]:
        if LinearNet._LAYER_DESCS is None:
            LinearNet._LAYER_DESCS = []
            model = LinearNet()

            for name, layer in model.named_modules():
                if not isinstance(layer, Linear):
                    continue

                LinearNet._LAYER_DESCS.append(
                    LayerDesc(
                        name,
                        [layer.in_features],
                        [layer.out_features],
                        layer.bias is not None,
                    )
                )

        return LinearNet._LAYER_DESCS

    def __init__(self):
        super().__init__()
        self.seq = Sequential(
            OrderedDict(
                [
                    ("fc1", Linear(8, 16, bias=True)),
                    ("fc2", Linear(16, 32, bias=True)),
                    (
                        "block1",
                        Sequential(
                            OrderedDict(
                                [
                                    ("fc1", Linear(32, 16, bias=True)),
                                    ("fc2", Linear(16, 8, bias=True)),
                                ]
                            )
                        ),
                    ),
                ]
            )
        )
        self.config = Mock(use_cache=False)

    def forward(self, inp: Tensor):
        return self.seq(inp)

    def _get_no_split_modules(self, device_map):
        return []


class MLPNet(Module):
    _LAYER_DESCS = None

    @staticmethod
    def layer_descs() -> List[LayerDesc]:
        if MLPNet._LAYER_DESCS is None:
            MLPNet._LAYER_DESCS = []
            model = MLPNet()

            for name, layer in model.named_modules():
                if isinstance(layer, Linear):
                    MLPNet._LAYER_DESCS.append(
                        LayerDesc(
                            name,
                            [layer.in_features],
                            [layer.out_features],
                            layer.bias is not None,
                        )
                    )
                elif isinstance(layer, ReLU):
                    MLPNet._LAYER_DESCS.append(
                        LayerDesc(
                            name,
                            [],
                            [],
                            False,
                        )
                    )

        return MLPNet._LAYER_DESCS

    def __init__(self):
        super().__init__()
        self.seq = Sequential(
            OrderedDict(
                [
                    ("fc1", Linear(8, 16, bias=True)),
                    ("act1", ReLU()),
                    ("fc2", Linear(16, 32, bias=True)),
                    ("act2", ReLU()),
                    ("fc3", Linear(32, 64, bias=True)),
                    ("sig", Sigmoid()),
                ]
            )
        )

    def forward(self, inp: Tensor):
        return self.seq(inp)


class FlatMLPNet(Module):
    _LAYER_DESCS = None

    @staticmethod
    def layer_descs() -> List[LayerDesc]:
        if FlatMLPNet._LAYER_DESCS is None:
            FlatMLPNet._LAYER_DESCS = []
            model = FlatMLPNet()

            for name, layer in model.named_modules():
                if isinstance(layer, Linear):
                    FlatMLPNet._LAYER_DESCS.append(
                        LayerDesc(
                            name,
                            [layer.in_features],
                            [layer.out_features],
                            layer.bias is not None,
                        )
                    )
                elif isinstance(layer, ReLU):
                    FlatMLPNet._LAYER_DESCS.append(
                        LayerDesc(
                            name,
                            [],
                            [],
                            False,
                        )
                    )

        return FlatMLPNet._LAYER_DESCS

    def __init__(self):
        super().__init__()
        self.seq = Sequential(
            OrderedDict(
                [
                    ("fc1", Linear(16, 16, bias=True)),
                    ("act1", ReLU()),
                    ("fc2", Linear(16, 16, bias=True)),
                    ("act2", ReLU()),
                    ("fc3", Linear(16, 16, bias=True)),
                    ("sig", Sigmoid()),
                ]
            )
        )

    def forward(self, inp: Tensor):
        return self.seq(inp)


class ConvNet(Module):
    _LAYER_DESCS = None

    @staticmethod
    def layer_descs() -> List[LayerDesc]:
        if ConvNet._LAYER_DESCS is None:
            ConvNet._LAYER_DESCS = [
                LayerDesc("seq.conv1", (3, 28, 28), (16, 14, 14), True),
                LayerDesc("seq.act1", [16, 14, 14], (16, 14, 14), False),
                LayerDesc("seq.conv2", (16, 14, 14), (32, 7, 7), True),
                LayerDesc("seq.act2", (32, 7, 7), (32, 7, 7), False),
                LayerDesc("mlp.fc", (32,), (10,), True),
            ]

        return ConvNet._LAYER_DESCS

    def __init__(self):
        super().__init__()
        self.seq = Sequential(
            OrderedDict(
                [
                    (
                        "conv1",
                        Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=True),
                    ),
                    ("act1", ReLU()),
                    (
                        "conv2",
                        Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=True),
                    ),
                    ("act2", ReLU()),
                ]
            )
        )
        self.pool = AdaptiveAvgPool2d(1)
        self.mlp = Sequential(
            OrderedDict([("fc", Linear(32, 10, bias=True)), ("sig", Sigmoid())])
        )

    def forward(self, inp: Tensor):
        out = self.seq(inp)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        classes = self.mlp(out)

        return classes


class _QATMatMul(Module):
    def __init__(self):
        super().__init__()

        # behaves like normal torch.matmul unless a LLM Compressor QuantizationModifier
        # is initialized
        self.wrap_qat = True
        self.qat_wrapper_kwargs = {
            "num_inputs": 2,
            "input_qconfigs": ["asymmetric", "symmetric"],
        }

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        return torch.matmul(a, b)


class QATMatMulTestNet(Module):
    _LAYER_DESCS = None

    @staticmethod
    def layer_descs() -> List[LayerDesc]:
        if QATMatMulTestNet._LAYER_DESCS is None:
            QATMatMulTestNet._LAYER_DESCS = [
                LayerDesc("fc", (16, 16), (16, 16), True),
                LayerDesc("matmul", (16, 16), (16, 16), False),
            ]

        return QATMatMulTestNet._LAYER_DESCS

    def __init__(self):
        super().__init__()
        self.fc = Linear(16, 16, bias=True)
        self.matmul = _QATMatMul()

    def forward(self, inp: Tensor):
        inp = self.fc(inp)
        inp = self.matmul(inp, inp)
        return inp


class MLPDataset(Dataset):
    def __init__(self, features: int = 8, classes: int = 64, length: int = 1000):
        self._length = length
        self._x_feats = [torch.randn(features) for _ in range(length)]
        self._y_labs = [torch.randn(classes) for _ in range(length)]

    def __getitem__(self, index: int):
        return self._x_feats[index], self._y_labs[index]

    def __len__(self) -> int:
        return self._length


class ConvDataset(Dataset):
    def __init__(
        self, features: int = 3, size: int = 28, classes: int = 10, length: int = 1000
    ):
        self._length = length
        self._x_feats = [torch.randn(features, size, size) for _ in range(length)]
        self._y_labs = [torch.randn(classes) for _ in range(length)]

    def __getitem__(self, index: int):
        return self._x_feats[index], self._y_labs[index]

    def __len__(self) -> int:
        return self._length


def create_optim_sgd(model: Module, lr: float = 0.0001) -> SGD:
    return SGD(model.parameters(), lr=lr)


def create_optim_adam(model: Module, lr: float = 0.0001) -> Adam:
    return Adam(model.parameters(), lr=lr)


SAMPLE_STAGED_RECIPE = """
stage_1:
    training_modifiers:
      - !EpochRangeModifier
        start_epoch: 0.0
        end_epoch: 50.0

      - !SetLearningRateModifier
        start_epoch: 0.0
        learning_rate: 0.1

    pruning_modifiers:
      - !GMPruningModifier
        start_epoch: 0
        end_epoch: 40
        init_sparsity: 0.05
        final_sparsity: 0.85
        params: ["params.1", "params.2"]
        update_frequency: 0.5

      - !GMPruningModifier
        start_epoch: 0
        end_epoch: 40
        init_sparsity: 0.05
        final_sparsity: 0.95
        params: ["params.3"]
        update_frequency: 0.5

stage_2:
    training_modifiers:
      - !EpochRangeModifier
        start_epoch: 50.0
        end_epoch: 52.0

      - !SetLearningRateModifier
        start_epoch: 0.0
        learning_rate: 0.0001

    quantization_modifiers:
      - !QuantizationModifier
          start_epoch: 50.0
"""
