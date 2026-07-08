import torch
from compressed_tensors.offload import get_cache_init_kwargs, offload_module
from transformers.models.granitemoe.configuration_granitemoe import GraniteMoeConfig
from transformers.models.granitemoe.modeling_granitemoe import GraniteMoeExperts # temporary fix

from llmcompressor.modeling.moe.context import get_calibrate_all_experts_flag
from llmcompressor.modeling.moe.linear_experts import LinearExperts2D
from llmcompressor.utils.dev import skip_weights_initialize


class GraniteMoeLinearExperts(LinearExperts2D):
    is_concatenated = False
    is_transposed = False
    has_bias = False
    has_gate = False

    @classmethod
    @torch.no_grad()
    def from_experts_module(
        cls, experts: "GraniteMoeParallelExperts", config: GraniteMoeConfig
    ):
        assert experts.num_experts == config.num_local_experts

        with skip_weights_initialize():
            self = cls(
                experts.num_experts, experts.input_size, experts.output_size, config
            )
            self.num_experts = experts.num_experts

        # TODO: experiment with copying views, not values
        for i in range(experts.num_experts):
            self[i].weight.copy_(experts.weight[i])

        # copy offloading from original
        offload_kwargs = get_cache_init_kwargs(experts)
        for module in self.modules():
            offload_module(module, **offload_kwargs)

        return self

    def __init__(
        self,
        num_experts: int,
        input_size: int,
        output_size: int,
        config: GraniteMoeConfig,
    ) -> None:
        self.num_experts = num_experts
        self.input_size = input_size
        self.output_size = output_size

        torch.nn.ModuleList.__init__(
            self,
            [
                torch.nn.Linear(input_size, output_size, bias=False, dtype=config.dtype)
                for _ in range(num_experts)
            ],
        )

    def forward(self, inputs: torch.Tensor, expert_size: list[int]):
        """
        Forward pass of the GraniteMoeParallelExperts module.
        Args:
            inputs (Tensor):
                Input tensor.
            expert_size:
                Expert size information.
        Returns:
            Tensor: Output tensor.
        """
        output_list = []

        for i in range(self.num_experts):
            if get_calibrate_all_experts_flag():
                expert_out = self[i](inputs).split(expert_size, dim=0)[i]
            else:
                expert_out = self[i](inputs.split(expert_size, dim=0)[i])
            output_list.append(expert_out)

        return torch.cat(output_list, dim=0)


# register in registry
LinearExperts2D._registry[GraniteMoeExperts] = GraniteMoeLinearExperts
