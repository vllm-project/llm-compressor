
from compressed_tensors.utils import register_offload_parameter
from compressed_tensors.quantization import QuantizationStatus

import torch
from transformers.models.granitemoehybrid.configuration_granitemoehybrid import (
    GraniteMoeHybridConfig,
)

from transformers.models.granitemoehybrid.modeling_granitemoehybrid import (
    GraniteMoeHybridParallelExperts,
)

from llmcompressor.utils.dev import skip_weights_initialize


class GraniteMoeHybridParallelExpertsModList(torch.nn.Module):
    def __init__(self, config: GraniteMoeHybridConfig, original) -> None:
        """Change .weight from a 3d tensor [num_experts, output_size, input_size] to a ModuleList of
        nn.Linears. Note Linears's weight are not pointers to the original .weight
        """
        super().__init__()
        self.num_experts = original.num_experts
        self.input_size = original.input_size
        self.output_size = original.output_size
        org_dev = original.weight.device

        # use real nn.Linear so that llm-compressor can handle it automatically
        self.experts = torch.nn.ModuleList()
        for i in range(self.num_experts):
            with skip_weights_initialize():
                self.experts.append(
                    torch.nn.Linear(self.input_size, self.output_size, bias=False, device=org_dev)
                )
            self.experts[i].weight.data = original.weight[i].clone().contiguous()
        # TODO memory management for large models? pointer assignment instead copy?
        original.to("cpu")

    def forward(self, inputs, expert_size):
        """Modified from original forward()"""
        input_list = inputs.split(expert_size, dim=0)
        output_list = []
        for i in range(self.num_experts):
            # original forward call was:
            #   output_list.append(F.linear(input_list[i], self.weight[i]))
            # NOTE F.linear takes transposed W, i.e. [out, in]
            output_list.append(
                self.experts[i](input_list[i].to(self.experts[i].weight.device))
            )

        results = torch.cat(output_list, dim=0)
        return results

    def dequant_experts_3d_weight(self):
        assert (hasattr(self, "experts") and
                hasattr(self.experts[0], "weight_scale")
                ), "this module cannot be converted back to dequant 3D moe."

        weight_3d = torch.empty(
            self.num_experts, self.output_size, self.input_size, dtype=torch.bfloat16,
            requires_grad=False,
        )
        for i in range(self.num_experts):
            weight_3d[i].copy_(
                self.experts[i].weight.to(torch.bfloat16) * self.experts[i].weight_scale
            )
        return weight_3d

    def __repr__(self):
        return f"{self.__class__.__name__}  {self.experts}"


class GraniteMoeHybridParallelExpertsLinear(torch.nn.Linear):
    def __init__(self, num_experts: int, input_size: int, output_size: int) -> None:
        """Use a real Linear so that llmcompressor and vllm can handle it easier.
        1. Change .weight from 3D [num_experts, output_size, input_size] to 2D
            [num_experts * output_size, input_size] before calling llm-compressor
        2. Change it back to 3D before saving ckpt
        """
        super().__init__(input_size, output_size*num_experts, bias=False, device="meta")
        self.num_experts = num_experts
        self.input_size = input_size
        self.output_size = output_size
        self.is_2d: bool = True

    @classmethod
    def from_3d_expert(cls, original: GraniteMoeHybridParallelExperts):
        """Extract weights of a GraniteMoeHybridParallelExperts module (transformers), convert it
        into 2D shape and store them into this "Linear" module.
        """
        newMoeLin = cls(original.num_experts, original.input_size, original.output_size)
        newMoeLin.weight = torch.nn.Parameter(
            original.weight.view(-1, original.input_size).clone(), requires_grad=False,
        )
        original.to("cpu")
        newMoeLin.is_2d = True
        return newMoeLin

    def to_3d_expert(self) -> None:
        """Convert weights and quantization parameters from 2D to 3D shape."""
        dim0_mul = self.num_experts * self.output_size
        assert (
            self.weight.shape == torch.Size((dim0_mul, self.input_size)) and
            hasattr(self, "weight_scale") and
            self.weight_scale.shape == torch.Size((dim0_mul, 1))
        ), "Shape mismatch, please check."

        self.weight = torch.nn.Parameter(
            self.weight.view(self.num_experts, self.output_size, self.input_size).clone(),
            requires_grad=False,
        )
        self.weight_scale = torch.nn.Parameter(
            self.weight_scale.view(self.num_experts, self.output_size, 1).clone(),
            requires_grad=False,
        )
        if hasattr(self, "weight_zero_point"):
            assert self.weight_zero_point.shape == torch.Size((dim0_mul, 1))
            self.weight_zero_point = torch.nn.Parameter(
                self.weight_zero_point.view(self.num_experts, self.output_size, 1).clone(),
                requires_grad=False,
            )
        self.is_2d = False

    def forward(self, inputs, expert_size):
        """Modified from original forward()"""

        input_list = inputs.split(expert_size, dim=0)
        # [CL] consider the case of CompressedLinear
        if getattr(self, "quantization_status", None) == QuantizationStatus.COMPRESSED:

            weight_data = self.compressor.decompress_module(self)
            param = torch.nn.Parameter(weight_data, dtype=torch.bfloat16, requires_grad=False)
            register_offload_parameter(self, "weight", param)

            self.quantization_status = QuantizationStatus.FROZEN

        weight_3d = self.weight.view(self.num_experts, self.output_size, self.input_size)
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
        return (
            f"{self.__class__.__name__}{sizes_str}"
        )


def replace(config: GraniteMoeHybridConfig, module: GraniteMoeHybridParallelExperts):
    return GraniteMoeHybridParallelExpertsModList(config=config.get_text_config(), original=module)
