"""
A tiny synthetic MoE model for testing the expert pruning modifiers.

``FakeMoEModel`` is a stack of MoE blocks (no attention or embeddings) whose
experts are ``LinearExperts2D`` modules behind a top-k router, matching the
module layout the expert pruning modifiers detect and prune.

``FakeLatentMoEModel`` mirrors nemotron_h's latent MoE instead: the experts
consume a down-projection of the block input (``moe_latent_size``) while the
router keeps scoring the block input itself, with sigmoid scores and an
``e_score_correction_bias`` steering expert selection.
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
    "FakeLatentMoEConfig",
    "FakeLinearExperts2D",
    "FakeGatelessLinearExperts2D",
    "FakeMoEModel",
    "FakeLatentMoEModel",
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


class FakeGatelessLinearExperts2D(LinearExperts2D):
    """
    LinearExperts2D subclass matching nemotron_h experts: no gate projection,
    the config's activation is applied directly after up_proj.
    """

    is_concatenated = False
    is_transposed = False
    has_bias = False
    has_gate = False


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


class FakeLatentMoEConfig(FakeMoEConfig):
    """
    ``FakeMoEConfig`` for a nemotron_h-style latent MoE: the experts consume a
    down-projection of the block input (``moe_latent_size``) while the router
    keeps scoring the block input, with sigmoid scores and a single (inert)
    routing group.
    """

    model_type = "fake_latent_moe"

    def __init__(self, moe_latent_size: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.moe_latent_size = moe_latent_size
        self.hidden_act = "relu2"  # nemotron_h expert activation
        self.scoring_func = "sigmoid"  # nemotron_h router score function
        self.n_group = 1  # single-group routing: no group limitation
        self.topk_group = 1


class FakeSigmoidRouter(nn.Module):
    """
    NemotronH-style top-k router: sigmoid scores, with an
    ``e_score_correction_bias`` steering expert selection and the selected
    scores renormalized into mixture weights.
    """

    def __init__(self, config: FakeLatentMoEConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_dim = config.hidden_size
        self.out_features = config.num_experts

        self.weight = nn.Parameter(torch.randn(config.num_experts, config.hidden_size))
        self.e_score_correction_bias = nn.Parameter(torch.randn(config.num_experts))

    def forward(self, hidden_states: torch.Tensor):
        """
        Route tokens to their top-k experts.

        :param hidden_states: [num_tokens, hidden_dim]
        :return: router logits, top-k weights, and selected expert indices,
            each [num_tokens, top_k] (except the logits)
        """
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)

        router_logits = F.linear(hidden_states, self.weight)
        scores = router_logits.sigmoid()
        # the correction bias steers selection only; it is not part of the
        # mixture weights
        selection_scores = scores + self.e_score_correction_bias
        _, selected_experts = torch.topk(
            selection_scores, self.top_k, dim=-1, sorted=False
        )
        router_weights = scores.gather(1, selected_experts)
        if self.norm_topk_prob:
            router_weights = router_weights / (
                router_weights.sum(dim=-1, keepdim=True) + 1e-20
            )

        return router_logits, router_weights.to(router_logits.dtype), selected_experts


class FakeLatentMoEBlock(nn.Module):
    """
    NemotronH-style MoE block: the router scores the block input, which is
    then projected down to ``moe_latent_size`` for the experts and projected
    back up afterwards.
    """

    def __init__(self, config: FakeLatentMoEConfig):
        super().__init__()
        self.gate = FakeSigmoidRouter(config)
        self.experts = FakeGatelessLinearExperts2D(config)
        self.fc1_latent_proj = nn.Linear(
            config.hidden_size, config.moe_latent_size, bias=False
        )
        self.fc2_latent_proj = nn.Linear(
            config.moe_latent_size, config.hidden_size, bias=False
        )
        self.num_experts = config.num_experts

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        _, router_weights, selected_experts = self.gate(hidden_states_flat)
        latent = self.fc1_latent_proj(hidden_states_flat)
        expert_output = self.experts(latent, selected_experts, router_weights)
        output = self.fc2_latent_proj(expert_output)

        return output.view(batch_size, seq_len, hidden_dim)


class FakeLatentMoEModel(nn.Module):
    """A stack of nemotron_h-style latent MoE blocks."""

    def __init__(self, config: FakeLatentMoEConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [FakeLatentMoEBlock(config) for _ in range(config.num_hidden_layers)]
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
