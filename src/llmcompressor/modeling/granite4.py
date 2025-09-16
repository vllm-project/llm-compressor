import torch
from transformers.models.granitemoehybrid.modeling_granitemoehybrid import (
    GraniteMoeHybridParallelExperts,
)


class GraniteMoeHybridParallelExpertsLinear(torch.nn.Linear):
    def __init__(self, num_experts: int, input_size: int, output_size: int) -> None:
        """Use a real Linear so that llmcompressor and vllm can handle it easier.
        1. Change .weight from 3D [num_experts, output_size, input_size] to 2D
            [num_experts * output_size, input_size] before calling llm-compressor
        2. Change it back to 3D before saving ckpt
        """
        super().__init__(
            input_size, output_size * num_experts, bias=False, device="meta"
        )
        self.num_experts = num_experts
        self.input_size = input_size
        self.output_size = output_size
        self.is_2d: bool = True

    @classmethod
    def from_3d_expert(cls, original: GraniteMoeHybridParallelExperts):
        """Reshape weights of GraniteMoeHybridParallelExperts module into 2D and store
        them as weights of this "Linear" module.
        """
        newMoeLin = cls(original.num_experts, original.input_size, original.output_size)
        newMoeLin.weight = torch.nn.Parameter(
            original.weight.view(-1, original.input_size).clone(),
            requires_grad=False,
        )
        original.to("cpu")
        newMoeLin.is_2d = True
        return newMoeLin

    def to_3d_expert(self) -> None:
        """Convert weights and quantization parameters from 2D to 3D shape."""
        dim0_mul = self.num_experts * self.output_size
        assert (
            self.weight.shape == torch.Size((dim0_mul, self.input_size))
            and hasattr(self, "weight_scale")
            and self.weight_scale.shape == torch.Size((dim0_mul, 1))
        ), "Shape mismatch, please check."

        self.weight = torch.nn.Parameter(
            self.weight.view(
                self.num_experts, self.output_size, self.input_size
            ).clone(),
            requires_grad=False,
        )
        self.weight_scale = torch.nn.Parameter(
            self.weight_scale.view(self.num_experts, self.output_size, 1).clone(),
            requires_grad=False,
        )
        if hasattr(self, "weight_zero_point"):
            assert self.weight_zero_point.shape == torch.Size((dim0_mul, 1))
            self.weight_zero_point = torch.nn.Parameter(
                self.weight_zero_point.view(
                    self.num_experts, self.output_size, 1
                ).clone(),
                requires_grad=False,
            )
        self.is_2d = False

    def forward(self, inputs, expert_size):
        """Modified from original forward()"""

        input_list = inputs.split(expert_size, dim=0)

        weight_3d = self.weight.view(
            self.num_experts, self.output_size, self.input_size
        )
        output_list = []
        for i in range(self.num_experts):
            output_list.append(torch.nn.functional.linear(input_list[i], weight_3d[i]))

        results = torch.cat(output_list, dim=0)
        return results

    def __repr__(self):
        if self.is_2d:
            sizes_str = f"(out={self.weight.shape[0]},in={self.weight.shape[1]})"
        else:
            sizes_str = (
                f"(exp={self.weight.shape[0]},out={self.weight.shape[1]},"
                f"in={self.weight.shape[2]})"
            )
        return f"{self.__class__.__name__}{sizes_str}"
