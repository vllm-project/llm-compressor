from typing import Any

import torch
from scipy.linalg import hadamard
from torch.nn import Module

from llmcompressor.core import State
from llmcompressor.modifiers import Modifier, ModifierFactory


def get_hadK(n, transpose=False):
    hadK, K = None, None
    if n % 172 == 0:  # llama-2-7b up
        assert is_pow2(n // 172)

        K = 172
        hadK = get_had172().T if transpose else get_had172()
    elif n % 156 == 0:  # llama-1-30b 3x hidden
        assert is_pow2(n // 156)

        K = 156
        hadK = get_had156().T if transpose else get_had156()
    elif n % 140 == 0:  # llama-1-30b intermediate
        assert is_pow2(n // 140)

        K = 140
        hadK = get_had140().T if transpose else get_had140()
    elif n % 108 == 0:  # llama-1-13b intermediate
        assert is_pow2(n // 108)

        K = 108
        hadK = get_had108().T if transpose else get_had108()
    elif n % 60 == 0:  # llama-1-13b 3x hidden
        assert is_pow2(n // 60)

        K = 60
        hadK = get_had60().T if transpose else get_had60()
    elif n % 52 == 0:  # llama-1-13b 1x hidden
        assert is_pow2(n // 52)

        K = 52
        hadK = get_had52().T if transpose else get_had52()
    elif n % 36 == 0:
        assert is_pow2(n // 36)

        K = 36
        hadK = get_had36().T if transpose else get_had36()
    elif n % 28 == 0:
        assert is_pow2(n // 28)

        K = 28
        hadK = get_had28().T if transpose else get_had28()
    elif n % 44 == 0:
        assert is_pow2(n // 44)

        K = 44
        hadK = get_had44().T if transpose else get_had44()
    elif n % 40 == 0:
        assert is_pow2(n // 40)

        K = 40
        hadK = get_had40().T if transpose else get_had40()
    elif n % 20 == 0:
        assert is_pow2(n // 20)

        K = 20
        hadK = get_had20().T if transpose else get_had20()
    elif n % 12 == 0:
        assert is_pow2(n // 12)

        K = 12
        hadK = get_had12().T if transpose else get_had12()
    else:
        assert is_pow2(n)

        K = 1

    return hadK, K


def is_pow2(n):
    return (n & (n - 1) == 0) and (n > 0)


def random_hadamard_matrix(size, device):
    # See https://cornell-relaxml.github.io/quip-sharp/ , Section "Randomized Hadamard Transformation"
    Q = torch.randint(low=0, high=2, size=(size,)).to(torch.float64)
    Q = Q * 2 - 1
    Q = torch.diag(Q)
    return matmul_hadU(Q).to(device)


def matmul_hadU(X, transpose=False):
    n = X.shape[-1]
    hadK, K = get_hadK(n, transpose)
    input = X.clone().view(-1, n, 1)
    output = input.clone()
    while input.shape[1] > K:
        input = input.view(input.shape[0], input.shape[1] // 2, 2, input.shape[2])
        output = output.view(input.shape)
        output[:, :, 0, :] = input[:, :, 0, :] + input[:, :, 1, :]
        output[:, :, 1, :] = input[:, :, 0, :] - input[:, :, 1, :]
        output = output.view(input.shape[0], input.shape[1], -1)
        (input, output) = (output, input)
    del output

    if K > 1:
        # Do not explicitly repeat - OOM
        # input = torch.bmm(
        #     hadK.repeat(len(input), 1, 1).to(input.device).to(input.dtype), input)
        # Use bcast instead
        input = hadK.view(1, K, K).to(input) @ input

    return input.view(X.shape) / torch.tensor(n).sqrt()


__all__ = ["SpinQuantModifier"]


class SpinQuantModifier(Modifier):
    dummy_args: str = None
    global_transform: Any = None
    transforms: Any = None
    scheme: Any = None
    targets: Any = None
    ignore: Any = None

    def on_initialize(self, state: State, **kwargs) -> bool:
        # should be updated based on model size
        # self.global_transform = random_hadamard_matrix(2048, "cuda")  # specific to SpinQuant
        self.global_transform = torch.eye(2048, 2048).to("cuda")
        self.transforms = self.update_weight_transform()
        # state.model.apply(self.register_transforms)

        # Apply QuantModifier
        self._quantization_modifier = None
        self._build_quant_modifier()
        self._quantization_modifier.initialize(state, **kwargs)

        return True

    def update_weight_transform(self):
        # SpinQuant specific transforms
        transforms = {
            "group_0": {
                "weights": {
                    "targets": ["re:.*o_proj*", "re:.*down_proj*"],
                    "transpose": True,
                    "type": self.global_transform,
                }
            },
            "group1": {
                "weights": {
                    "targets": [
                        "re:.*gate_proj.*",
                        "re:.*up_proj.*",
                        "re:.*q_proj.*",
                        "re:.*k_proj.*",
                        "re:.*v_proj.*",
                    ],
                    "transpose": False,
                    "type": self.global_transform,
                }
            },
        }
        return transforms

    # TODO: add additional argument for transform?
    # TODO: merge in?
    # Note: if we do this with hooks, we dont impact the wrapped QDQ
    def apply_r1_transform(self, module: Module, _args: Any, output: torch.Tensor):
        if isinstance(module, torch.nn.modules.sparse.Embedding):
            transform = self.global_transform.to(output.dtype)
            return output @ transform

        transform = self.global_transform.to(output[0].dtype)
        return output[0] @ transform.T, output[1]

    # Add after we've added the transforms to the wrapped forward pass
    def register_transforms(self, module: Module):
        # Assume for now we're not quantizing the embedding layer or output fro the ModuleList
        # add case to support this after
        if isinstance(module, torch.nn.modules.sparse.Embedding):
            self.register_hook(module, self.apply_r1_transform, "forward")
        elif isinstance(module, torch.nn.modules.container.ModuleList):
            self.register_hook(module[-1], self.apply_r1_transform, "forward")

    def _build_quant_modifier(self):
        """
        Build a quantization modifier based on the specified config_groups,
        ignore list, and num_calibration_steps.

        :postcondition: self._quantization_modifier is set to the built
            quantization modifier
        """

        quantization_args_names = [
            "config_groups",
            "targets",
            "scheme",
            "num_calibration_steps",
            "ignore",
            "disable_quantization_observer_epoch",
            "transforms",
        ]

        quant_args = {
            key: getattr(self, key)
            for key in quantization_args_names
            if getattr(self, key, False)
        }

        vllm_quant_config = {"QuantizationModifier": quant_args}
        self._build_quant_modifier_from_dict(vllm_quant_config)

    def _build_quant_modifier_from_dict(self, quant_config):
        modifier_type = list(quant_config.keys())[0]
        modifier_args = quant_config[modifier_type]
        self._quantization_modifier = ModifierFactory.create(
            modifier_type,
            allow_registered=True,
            allow_experimental=True,
            **modifier_args,
        )
