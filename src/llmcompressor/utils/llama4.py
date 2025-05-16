from typing import Callable, Dict, Optional, Union

import torch
import tqdm
from accelerate.hooks import attach_align_device_hook
from compressed_tensors.utils import (
    align_module_device,
    delete_offload_parameter,
    has_offloaded_params,
    register_offload_parameter,
)
from torch.nn import Module
from transformers import Llama4ForConditionalGeneration
from transformers.models.llama4.configuration_llama4 import Llama4Config
from transformers.models.llama4.modeling_llama4 import Llama4TextExperts

from llmcompressor.utils.dev import skip_weights_initialize


def module_bfs(
    module: Module,
    func: Callable[[Module], Module],
    pre: bool = True,
    progress: Union[bool, tqdm.tqdm] = False,
) -> Module:
    if progress is True:
        total = len(list(module.modules()))
        progress = tqdm.tqdm(total=total)

    if pre:
        module = func(module)

    for name, child in list(module.named_children()):
        module.add_module(name, module_bfs(child, func, pre, progress))

    if not pre:
        module = func(module)

    if isinstance(progress, tqdm.tqdm):
        progress.update(1)

    return module


class Llama4TextMLP(torch.nn.Module):
    def __init__(self, hidden_size, intermediate_size, act_fn):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.act_fn = act_fn

        self.gate_up_proj = torch.nn.Linear(
            self.hidden_size, 2 * self.intermediate_size, bias=False
        )
        self.down_proj = torch.nn.Linear(
            self.intermediate_size, self.hidden_size, bias=False
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(hidden_states)
        gate, up = (
            gate_up[:, : self.intermediate_size],
            gate_up[:, self.intermediate_size :],
        )
        return self.down_proj(up * self.act_fn(gate))


class Llama4TextExpertsLinear(torch.nn.Module):
    def __init__(
        self,
        config: Llama4Config,
        act_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.intermediate_size = config.intermediate_size
        self.hidden_size = config.hidden_size

        self.experts = torch.nn.ModuleList(
            [
                Llama4TextMLP(self.hidden_size, self.intermediate_size, act_fn)
                for _ in range(self.num_experts)
            ]
        )

        # self.register_state_dict_post_hook(self.Hook())

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = torch.empty_like(hidden_states)
        output = output.view(self.num_experts, -1, self.hidden_size)
        hidden_states = hidden_states.view(self.num_experts, -1, self.hidden_size)
        for expert_index, expert in enumerate(self.experts):
            output[expert_index] = expert(hidden_states[expert_index])

        return output.view(-1, self.hidden_size)

    @classmethod
    def from_module(cls, module: Llama4TextExperts) -> "Llama4TextExpertsLinear":
        config = Llama4Config(
            num_local_experts=module.num_experts,
            intermediate_size=module.intermediate_size,
            hidden_size=module.hidden_size,
        )
        with skip_weights_initialize():
            instance = cls(config, module.act_fn)

        if has_offloaded_params(module):
            weights_map = module._hf_hook.weights_map
            for name, expert in instance.experts.named_children():
                attach_align_device_hook(
                    module=expert,
                    execution_device=module._hf_hook.execution_device,
                    offload=module._hf_hook.offload,
                    weights_map=weights_map,
                    module_name=f"experts.{name}",
                    skip_keys=module._hf_hook.skip_keys,
                )

        with align_module_device(module):
            gate_up_proj = module.gate_up_proj.data.transpose(-2, -1)
            down_proj = module.down_proj.data.transpose(-2, -1)

            for expert_index in range(module.num_experts):
                register_offload_parameter(
                    instance.experts[expert_index].gate_up_proj,
                    "weight",
                    torch.nn.Parameter(gate_up_proj[expert_index]),
                )
                register_offload_parameter(
                    instance.experts[expert_index].down_proj,
                    "weight",
                    torch.nn.Parameter(down_proj[expert_index]),
                )

        for name, _ in list(module.named_parameters()):
            delete_offload_parameter(module, name)

        return instance

    class Hook:
        def __call__(
            self,
            module: "Llama4TextExpertsLinear",
            state_dict: Dict[str, torch.Tensor],
            prefix: str,
            local_metadata,
        ):
            gate_up_proj = torch.empty(
                module.num_experts, module.hidden_size, module.intermediate_size * 2
            )
            down_proj = torch.empty(
                module.num_experts, module.intermediate_size, module.hidden_size
            )
            for expert_index in range(module.num_experts):
                gate_up_key = f"{prefix}experts.{expert_index}.gate_up_proj.weight"
                down_key = f"{prefix}experts.{expert_index}.down_proj.weight"

                gate_up_proj[expert_index] = state_dict[gate_up_key].T
                down_proj[expert_index] = state_dict[down_key].T

                del state_dict[gate_up_key]
                del state_dict[down_key]

            state_dict[f"{prefix}gate_up_proj"] = gate_up_proj
            state_dict[f"{prefix}down_proj"] = down_proj


def linearize_moe(
    model: Llama4ForConditionalGeneration,
) -> Llama4ForConditionalGeneration:
    def replace_fn(module: Module) -> Module:
        if module.__class__.__name__ == "Llama4TextExperts":
            return Llama4TextExpertsLinear.from_module(module)
        else:
            return module

    return module_bfs(model, replace_fn, progress=True)
