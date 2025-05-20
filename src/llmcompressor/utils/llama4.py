from transformers import AutoTokenizer, Llama4ForConditionalGeneration, Llama4Processor
from transformers.quantizers.quantizers_utils import get_module_from_name

from llmcompressor.utils.dev import skip_weights_initialize, skip_weights_download

import torch

def convert_model_for_quantization(model):
    import torch.nn as nn

    for name, module in model.named_modules():
        module_class_name = module.__class__.__name__
        if module_class_name == "Llama4TextMoe":
            # Access the fused weights
            gate_up_proj = module.gate_up_proj  # Shape: (num_experts, hidden_size, intermediate_size * 2)
            down_proj = module.down_proj  # Shape: (num_experts, intermediate_size, hidden_size)
            
            parent_module, module_name = get_module_from_name(model, name)
            parent_module._modules[module_name] = SequentialLlama4TextMoe(
                model.config.get_text_config(),
                gate_up_proj,
                down_proj
            )


class SequentialLlama4TextMoe(torch.nn.Module):
    def __init__(self, config):
        from transformers.models.llama4.modeling_llama4 import Llama4TextMLP

        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_local_experts
        self.experts = SequentialLlama4TextExperts(config)  # use sequential
        self.router = torch.nn.Linear(config.hidden_size, config.num_local_experts, bias=False)
        self.shared_expert = Llama4TextMLP(config)

    def forward(self, hidden_states):
        batch, seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = self.router(hidden_states)
        tokens_per_expert = batch * seq_len

        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=1)
        router_scores = (
            torch.full_like(router_logits, float("-inf")).scatter_(1, router_indices, router_top_value).transpose(0, 1)
        )
        # We do this to make sure we have -inf for non topK tokens before going through the !
        # Here we are just creating a tensor to index each and every single one of the hidden states. Let s maybe register a buffer for this!
        router_indices = (
            torch.arange(tokens_per_expert, device=hidden_states.device).view(1, -1).expand(router_scores.size(0), -1)
        )
        router_scores = torch.sigmoid(router_scores.float()).to(hidden_states.dtype)

        router_indices = router_indices.reshape(-1, 1).expand(-1, hidden_dim)
        routed_in = torch.gather(
            input=hidden_states,
            dim=0,
            index=router_indices,
        ).to(hidden_states.device)
        # we gather inputs corresponding to each expert based on the router indices
        routed_in = routed_in * router_scores.reshape(-1, 1)
        routed_out = self.experts(routed_in)
        out = self.shared_expert(hidden_states)
        # now that we finished expert computation -> we scatter add because we gathered previously
        # we have to do this because we used all experts on all tokens. This is faster than the for loop, tho you are compute bound
        # this scales a lot better if you do EP!

        # TODO: move into self.experts
        out.scatter_add_(dim=0, index=router_indices, src=routed_out.view(-1, hidden_dim))
        return out, router_scores


class SequentialLlama4TextExperts(torch.nn.ModuleList):
    """
    A module that implements a compressed version of a list of expert modules.
    This is specifically designed to work with Llama4TextExperts in MoE layers.
    """

    def __init__(self, config, gate_up_proj, down_proj):
        from transformers.models.llama4.modeling_llama4 import Llama4TextMLP
        import torch.nn as nn
        
        # Initialize empty MLPs
        with skip_weights_initialize():
            super().__init__([Llama4TextMLP(config) for _ in range(gate_up_proj.shape[0])])
        self.num_experts = gate_up_proj.shape[0]
        
        # Split and assign the weights to individual MLPs
        hidden_size = gate_up_proj.shape[1]
        intermediate_size = down_proj.shape[1]
        
        for expert_idx in range(self.num_experts):
            # Extract weights for this expert
            expert_gate_up = gate_up_proj[expert_idx]  # (hidden_size, intermediate_size * 2)
            expert_down = down_proj[expert_idx]  # (intermediate_size, hidden_size)
            
            # Split gate_up into gate and up projections
            gate_proj = expert_gate_up[:, :intermediate_size]
            up_proj = expert_gate_up[:, intermediate_size:]
            
            # Assign weights to the MLP
            self[expert_idx].gate_proj.weight.data = gate_proj.t()  # Transpose to match expected shape
            self[expert_idx].up_proj.weight.data = up_proj.t()
            self[expert_idx].down_proj.weight.data = expert_down.t()

    def forward(
        self,
        hidden_states: "torch.Tensor",
    ) -> "torch.Tensor":
        hidden_states = hidden_states.reshape(self.num_experts, -1, hidden_states.shape[-1])
        routed_out = torch.zeros_like(hidden_states)
        # TODO: use an additive accumulator
        for expert_idx in range(self.num_experts):
            routed_out[expert_idx] = self[expert_idx](hidden_states[expert_idx])
        return routed_out

if __name__ == "__main__":
    model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
    model_output = "llama4-reshaped-all-2"
    with skip_weights_download(Llama4ForConditionalGeneration):
        model = Llama4ForConditionalGeneration.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16, disable_custom_kernels=True)
    processor = Llama4Processor.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    convert_model_for_quantization(model)