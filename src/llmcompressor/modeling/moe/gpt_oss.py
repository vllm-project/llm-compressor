class GptOssExpertMLP(ExpertMLPWithGate):
    @classmethod
    def from_experts(cls, experts: FusedExpertsModule, expert_index: int, moe_intermediate_size: int, hidden_dim: int):
        assert experts.has_gate
        if experts.__class__._apply_gate is not _default_apply_gate:
            # assume that if a `_apply_gate` is implemented, then the weight is not valid for quantization (for example, might be interleaved)
            raise NotImplementedError(
                f"Linearization for {experts.__class__.__name__} has not been implemented yet"
            )

        with skip_weights_initialize():
            instance = cls(hidden_dim, moe_intermediate_size, experts.has_bias, experts._apply_gate)

        # load weights
        gate_weight = experts.gate_up_proj[expert_index, :moe_intermediate_size]
        up_weight = experts.gate_up_proj[expert_index, moe_intermediate_size:]
        down_weight = experts.down_proj[expert_index]

        if experts.is_transposed:
            gate_weight = gate_weight.T
            up_weight = up_weight.T
            down_weight = down_weight.T

        instance.gate_proj.weight.copy_(gate_weight)
        instance.up_proj.weight.copy_(up_weight)
        instance.down_proj.weight.copy_(down_weight)

        # load biases
        if experts.has_bias:
            gate_bias = experts.gate_up_proj_bias[expert_index, :moe_intermediate_size]
            up_bias = experts.gate_up_proj_bias[expert_index, moe_intermediate_size:]
            down_bias = experts.down_proj_bias[expert_index]

            instance.gate_proj.bias.copy_(gate_bias)
            instance.up_proj.bias.copy_(up_bias)
            instance.down_proj.bias.copy_(down_bias)
            
        return instance