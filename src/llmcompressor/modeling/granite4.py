import torch
import torch.nn as nn
from transformers.models.granitemoehybrid.configuration_granitemoehybrid import (
    GraniteMoeHybridConfig,
)
from transformers.models.granitemoehybrid.modeling_granitemoehybrid import (
    GraniteMoeHybridTopKGating,
    GraniteMoeHybridParallelExperts,
    GraniteMoeHybridMLP,
)

class SequentialGraniteMoeHybridMoE(nn.Module):
    """
    Sparsely gated Mixture-of-Experts (MoE) layer with optional calibration mode.

    When calibrate_all_experts=True, all experts process every token,
    but only top-k outputs (with their gates) contribute to the final output.
    This maintains exact numerical equivalence with the sparse routing result
    while exposing all experts for calibration (e.g., quantization stats collection).
    """

    def __init__(
        self, 
        config: GraniteMoeHybridConfig, 
        original: GraniteMoeHybridParallelExperts, 
        calibrate_all_experts: bool,
    ):
        super().__init__()
        self.input_size = config.hidden_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.calibrate_all_experts = calibrate_all_experts

        # Router determines which experts handle which tokens
        self.router = GraniteMoeHybridTopKGating(
            input_size=self.input_size,
            num_experts=self.num_experts,
            top_k=self.top_k,
        )

        # Per-expert MLPs
        self.experts = nn.ModuleList([GraniteMoeHybridMLP(config) for _ in range(self.num_experts)])

    def forward(self, layer_input: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, hidden_dim = layer_input.size()
        num_tokens = bsz * seq_len
        layer_input = layer_input.reshape(num_tokens, hidden_dim)

        # --- Routing ---
        _, batch_index, batch_gates, expert_size, logits = self.router(layer_input)
        layer_output = torch.zeros(
            (num_tokens, hidden_dim), dtype=layer_input.dtype, device=layer_input.device
        )

        if self.calibrate_all_experts:
            # ----------------------------------------------------
            # Calibration mode: all experts process all tokens
            # ----------------------------------------------------
            # Compute the top-k mask (same as sparse mode)
            top_k_logits, top_k_indices = logits.topk(self.top_k, dim=1)
            top_k_gates = torch.softmax(top_k_logits, dim=1).type_as(layer_input)

            # Create a dense gating tensor with zeros for non-top-k experts
            dense_gates = torch.zeros_like(logits, dtype=layer_input.dtype)
            dense_gates.scatter_(1, top_k_indices, top_k_gates)

            # Run all experts on all tokens
            for i, expert in enumerate(self.experts):
                outputs = expert(layer_input)                      # [num_tokens, hidden_dim]
                layer_output += outputs * dense_gates[:, i].unsqueeze(1)

        else:
            # ----------------------------------------------------
            # Sparse routing mode (normal inference)
            # ----------------------------------------------------
            expert_inputs = layer_input[batch_index]
            input_splits = expert_inputs.split(expert_size, dim=0)
            gate_splits = batch_gates.split(expert_size, dim=0)
            index_splits = batch_index.split(expert_size, dim=0)

            for i, expert in enumerate(self.experts):
                if expert_size[i] == 0:
                    continue
                outputs = expert(input_splits[i])
                outputs = outputs * gate_splits[i].unsqueeze(1)
                layer_output.index_add_(0, index_splits[i], outputs)

        # --- Restore shape [B, T, H] ---
        layer_output = layer_output.view(bsz, seq_len, hidden_dim)
        return layer_output

def replace(
    config: GraniteMoeHybridConfig, 
    module: GraniteMoeHybridParallelExperts, 
    calibrate_all_experts: bool,
):
    return SequentialGraniteMoeHybridMoE(
        config=config,
        original=module,
        calibrate_all_experts=calibrate_all_experts,
    )