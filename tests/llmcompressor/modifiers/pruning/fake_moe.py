"""
A tiny synthetic MoE model for testing the expert pruning modifiers.

``FakeMoEModel`` is a stack of MoE blocks (no attention or embeddings) whose
experts are ``LinearExperts2D`` modules behind a top-k router, matching the
module layout the expert pruning modifiers detect and prune.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.configuration_utils import PretrainedConfig

from llmcompressor.core import State
from llmcompressor.modeling.moe.linear_experts import LinearExperts2D
from llmcompressor.modifiers.pruning.expert_sparsity.utils import get_experts_router

__all__ = [
    "FakeMoEConfig",
    "FakeLinearExperts2D",
    "FakeMoEModel",
    "find_moe_blocks",
    "make_state",
]


class FakeMoEConfig(PretrainedConfig):
    """Minimal PreTrainedConfig for testing LinearExperts2D."""

    model_type = "fake_moe"

    def __init__(
        self,
        num_experts: int = 8,
        hidden_size: int = 32,
        intermediate_size: int = 64,
        num_hidden_layers: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Required for MoEConfig.from_config
        self.num_experts = num_experts
        self.num_experts_per_tok = 2
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_act = "silu"
        self.use_bias = False
        self.dtype = torch.float32
        self.num_hidden_layers = num_hidden_layers
        # Router config
        self.norm_topk_prob = True


class FakeLinearExperts2D(LinearExperts2D):
    """
    LinearExperts2D subclass with class variables set for testing.
    These match the standard MoE expert format (gate + up projections).
    """

    is_concatenated = False
    is_transposed = False
    has_bias = False
    has_gate = True

    @staticmethod
    def _apply_gate(gate_up_out: torch.Tensor) -> torch.Tensor:
        """SwiGLU gating: SiLU(gate) * up."""
        gate, up = gate_up_out.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * up


class FakeRouter(nn.Module):
    """
    Top-K router for a MoE layer, based on standard MoE router
    implementations (e.g. Qwen3MoeTopKRouter).
    """

    def __init__(self, config: FakeMoEConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_dim = config.hidden_size
        self.out_features = config.num_experts

        # Router weight: [num_experts, hidden_dim], an nn.Parameter to match
        # the structure expected by prune_moe_layer
        self.weight = nn.Parameter(torch.randn(config.num_experts, config.hidden_size))

    def forward(self, hidden_states: torch.Tensor):
        """
        Route tokens to their top-k experts.

        :param hidden_states: [num_tokens, hidden_dim]
        :return: router logits, top-k weights, and selected expert indices,
            each [num_tokens, top_k] (except the logits)
        """
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)

        router_logits = F.linear(hidden_states, self.weight)
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        router_weights, selected_experts = torch.topk(router_probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            router_weights = router_weights / router_weights.sum(dim=-1, keepdim=True)

        return router_logits, router_weights.to(router_logits.dtype), selected_experts


class FakeMoEBlock(nn.Module):
    """Single MoE block with a router and experts."""

    def __init__(self, config: FakeMoEConfig):
        super().__init__()
        self.gate = FakeRouter(config)
        self.experts = FakeLinearExperts2D(config)
        self.num_experts = config.num_experts

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        _, router_weights, selected_experts = self.gate(hidden_states_flat)
        expert_output = self.experts(
            hidden_states_flat, selected_experts, router_weights
        )

        return expert_output.view(batch_size, seq_len, hidden_dim)


class FakeMoEModel(nn.Module):
    """A stack of MoE blocks (no attention or embeddings)."""

    def __init__(self, config: FakeMoEConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [FakeMoEBlock(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


def make_state(model: nn.Module) -> State:
    return State(
        model=model,
        teacher_model=None,
        optimizer=None,
        optim_wrapped=False,
        loss=None,
        batch_data=None,
    )


def find_moe_blocks(model: nn.Module) -> dict[str, nn.Module]:
    """Map each MoE block's name to its module, using the same detection as
    the expert pruning modifiers: a module with a ``LinearExperts2D`` child
    behind a router."""
    blocks = {}
    for name, module in model.named_modules():
        if get_experts_router(name, module)[0] is not None:
            blocks[name] = module
    return blocks
