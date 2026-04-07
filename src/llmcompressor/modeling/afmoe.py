import torch

from llmcompressor.modeling.moe_context import MoECalibrationModule


@MoECalibrationModule.register("AfmoeMoE")
class CalibrationAfmoeMoE(MoECalibrationModule):
    """
    Calibration version of AfmoeMoE that sends all tokens to all experts.

    During calibration, when calibrate_all_experts=True, all tokens are sent to
    all experts to ensure proper quantization statistics are collected for every
    expert, not just those activated by the calibration data routing.

    The Afmoe architecture uses:
    - Token-choice top-K routing with sigmoid/softmax scoring
    - Optional shared experts processed on all tokens
    - Learnable expert bias for routing control

    Note: AfmoeMoE is loaded dynamically from the model hub via trust_remote_code=True.
    The original module is passed as a parameter.
    """

    is_permanent = False

    def __init__(
        self,
        original: torch.nn.Module,
        config,
        calibrate_all_experts: bool = True,
    ):
        super().__init__()
        self.config = config
        self.router = original.router
        self.experts = original.experts
        self.shared_experts = original.shared_experts
        self.expert_bias = original.expert_bias
        self.calibrate_all_experts = calibrate_all_experts

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional calibration mode.

        When calibrate_all_experts=True:
            - All tokens are sent to all experts for calibration
            - Routing weights are still used for final output combination
            - This ensures all experts see calibration data
        When calibrate_all_experts=False:
            - Normal MoE routing behavior (only routed tokens go to each expert)
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        # Step 1: Get routing decisions
        top_scores, selected_experts = self.router(hidden_states, self.expert_bias)

        # Step 2: Process through shared experts
        if self.shared_experts is not None:
            shared_output = self.shared_experts(hidden_states_flat)
        else:
            shared_output = torch.zeros_like(hidden_states_flat)

        # Step 3: Create expert mask for routing - which tokens
        # were selected
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.config.num_experts
        ).permute(2, 1, 0)  # (num_experts, top_k, batch_size * seq_len)

        # Step 4: Process routed experts
        routed_output = torch.zeros_like(
            hidden_states_flat, dtype=hidden_states.dtype, device=hidden_states.device
        )

        for expert_idx, expert in enumerate(self.experts):
            # Get the indices of tokens routed to this expert
            idx, token_idx = torch.where(expert_mask[expert_idx])

            if self.calibrate_all_experts:
                # Pass all tokens through the expert but only outputs
                # for the selected tokens are extracted (i.e if this
                # expert was selected)
                expert_output = expert(hidden_states_flat)[token_idx]
            else:
                # Only pass routed tokens through the expert
                expert_output = expert(hidden_states_flat[token_idx])

            # If any tokens were routed to this expert, add their contribution
            if len(token_idx) > 0:
                weighted_output = expert_output * top_scores[token_idx, idx, None]
                # add weighted output to the final output for the routed tokens
                routed_output.index_add_(
                    0, token_idx, weighted_output.to(hidden_states.dtype)
                )

        # Step 5: Combine shared and routed expert output
        output = shared_output.to(hidden_states.dtype) + routed_output.to(
            hidden_states.dtype
        )
        return output.view(batch_size, seq_len, hidden_dim)

    def restore(self, original: torch.nn.Module) -> torch.nn.Module:
        """
        Restore the original module structure.

        Since is_permanent=False, this method is called when exiting
        the calibration context to restore the original MoE module.
        """
        return original
