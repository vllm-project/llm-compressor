"""
Development tests for REAP MoE expert pruning with LinearExperts2D.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.configuration_utils import PretrainedConfig

from llmcompressor.modeling.moe.linear_experts import LinearExperts2D, ExpertMLP
from llmcompressor.modifiers.pruning.reap.utils import MoEModelAttrs, prune_moe_layer, SOFTMAX


class FakeMoEConfig(PretrainedConfig):
    """Minimal PreTrainedConfig for testing LinearExperts2D."""

    model_type = "fake_moe"

    def __init__(
        self,
        num_experts: int = 8,
        hidden_size: int = 32,
        intermediate_size: int = 64,
        num_hidden_layers: int = 2,
        **kwargs
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
        """
        Apply SwiGLU gating: splits the concatenated tensor into gate and up parts,
        applies SiLU to the gate part, and multiplies with the up part.

        Args:
            gate_up_out: [batch, 2 * intermediate_size] concatenated tensor

        Returns:
            [batch, intermediate_size] gated output
        """
        gate, up = gate_up_out.chunk(2, dim=-1)  # Split into two halves
        return torch.nn.functional.silu(gate) * up  # SwiGLU: SiLU(gate) ⊙ up


class FakeRouter(nn.Module):
    """
    Top-K router for MoE layer.
    Based on standard MoE router implementations (e.g., Qwen3MoeTopKRouter).

    Uses nn.Parameter for weight to match the structure expected by prune_moe_layer.
    """

    def __init__(self, config: FakeMoEConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_dim = config.hidden_size
        self.out_features = config.num_experts

        # Router weight: [num_experts, hidden_dim]
        # Using nn.Parameter to match expected structure (not nn.Linear)
        self.weight = nn.Parameter(torch.randn(config.num_experts, config.hidden_size))

    def forward(self, hidden_states: torch.Tensor):
        """
        Route tokens to top-k experts.

        Args:
            hidden_states: [batch_size * seq_len, hidden_dim] or [seq_len, hidden_dim]

        Returns:
            router_logits: [seq_len, num_experts] - raw router scores
            router_weights: [seq_len, top_k] - normalized weights for selected experts
            selected_experts: [seq_len, top_k] - indices of selected experts
        """
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)

        # Compute router logits using F.linear (weight transposed for correct matmul)
        router_logits = F.linear(hidden_states, self.weight)  # [seq_len, num_experts]

        # Apply softmax to get probabilities
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)

        # Select top-k experts
        router_weights, selected_experts = torch.topk(
            router_probs, self.top_k, dim=-1
        )  # Both: [seq_len, top_k]

        # Normalize top-k weights if configured
        if self.norm_topk_prob:
            router_weights = router_weights / router_weights.sum(dim=-1, keepdim=True)

        # Convert back to original dtype
        router_weights = router_weights.to(router_logits.dtype)

        return router_logits, router_weights, selected_experts


class FakeMoEBlock(nn.Module):
    """
    Single MoE block with router and experts.
    Based on standard MoE block implementations (e.g., Qwen3MoeSparseMoeBlock).
    """

    def __init__(self, config: FakeMoEConfig):
        super().__init__()
        self.gate = FakeRouter(config)
        self.experts = FakeLinearExperts2D(config)
        self.num_experts = config.num_experts

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MoE block.

        Args:
            hidden_states: [batch_size, seq_len, hidden_dim]

        Returns:
            output: [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Flatten to [batch_size * seq_len, hidden_dim]
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        # Route tokens to experts
        router_logits, router_weights, selected_experts = self.gate(hidden_states_flat)

        # Process tokens through selected experts
        expert_output = self.experts(hidden_states_flat, selected_experts, router_weights)

        # Reshape back to [batch_size, seq_len, hidden_dim]
        output = expert_output.view(batch_size, seq_len, hidden_dim)

        return output


class FakeMoEModel(nn.Module):
    """
    Complete MoE model with multiple layers.
    Simplified model with only MoE blocks (no attention, embeddings, etc.).
    """

    def __init__(self, config: FakeMoEConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            FakeMoEBlock(config) for _ in range(config.num_hidden_layers)
        ])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all MoE layers.

        Args:
            hidden_states: [batch_size, seq_len, hidden_dim]

        Returns:
            output: [batch_size, seq_len, hidden_dim]
        """
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


def test_layer_forward_pass(model: FakeMoEModel, layer_name: str):
    """
    Test forward pass through a single MoE layer.

    Args:
        model: The FakeMoEModel instance
        layer_name: Name of the layer to test (e.g., "layers.0")
    """
    print("\n" + "="*70)
    print(f"Testing Layer Forward Pass: {layer_name}")
    print("="*70)

    config = model.config
    layer = model.get_submodule(layer_name)

    print(f"\nLayer configuration:")
    print(f"  Layer: {layer_name}")
    print(f"  Type: {type(layer).__name__}")
    print(f"  Number of experts: {layer.num_experts}")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Top-k routing: {config.num_experts_per_tok}")

    # Create test input
    batch_size = 4
    seq_len = 8
    num_tokens = batch_size * seq_len
    hidden_dim = config.hidden_size

    # Input for a single layer: [batch, seq, hidden]
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim, dtype=config.dtype)

    print(f"\nInput configuration:")
    print(f"  hidden_states shape: {hidden_states.shape}")

    # Perform forward pass
    print(f"\nPerforming forward pass...")
    with torch.no_grad():
        output = layer(hidden_states)

    print(f"  Output shape: {output.shape}")

    # Validation checks
    print(f"\nValidation checks:")

    # Check 1: Output shape matches input shape
    assert output.shape == hidden_states.shape, \
        f"Output shape {output.shape} doesn't match input shape {hidden_states.shape}"
    print(f"  ✓ Output shape matches input: {output.shape}")

    # Check 2: No NaN values
    has_nan = torch.isnan(output).any().item()
    assert not has_nan, "Output contains NaN values"
    print(f"  ✓ No NaN values in output")

    # Check 3: No Inf values
    has_inf = torch.isinf(output).any().item()
    assert not has_inf, "Output contains Inf values"
    print(f"  ✓ No Inf values in output")

    # Check 4: Output has non-zero values
    is_nonzero = (output.abs() > 1e-6).any().item()
    assert is_nonzero, "Output is all zeros - layer may not be working"
    print(f"  ✓ Output contains non-zero values")

    # Check 5: Output is different from input (transformation occurred)
    is_different = not torch.allclose(output, hidden_states, rtol=1e-3)
    assert is_different, "Output is identical to input - no transformation occurred"
    print(f"  ✓ Output differs from input (transformation applied)")

    # Check 6: Different inputs produce different outputs
    hidden_states_2 = torch.randn(batch_size, seq_len, hidden_dim, dtype=config.dtype)
    with torch.no_grad():
        output_2 = layer(hidden_states_2)
    is_deterministic = not torch.allclose(output, output_2, rtol=1e-3)
    assert is_deterministic, "Different inputs produced same output"
    print(f"  ✓ Different inputs produce different outputs")

    print("\n" + "="*70)
    print(f"All validation checks passed for {layer_name}!")
    print("="*70 + "\n")


def test_prune_moe_layer_unit():
    """
    Unit test for prune_moe_layer function.

    Tests that:
    1. Experts are correctly pruned to the retained set
    2. Router weights are correctly sliced
    3. num_experts attributes are updated
    4. Forward pass still works after pruning
    5. Model forward pass still works after pruning
    """
    print("\n" + "="*70)
    print("Testing prune_moe_layer Unit Test")
    print("="*70)

    # Set seed for reproducibility
    torch.manual_seed(42)

    # Create model
    config = FakeMoEConfig(
        num_experts=8,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2
    )
    model = FakeMoEModel(config)
    model.eval()

    # Define MoEModelAttrs for our fake model
    attrs = MoEModelAttrs(
        router_attr="gate",
        experts_attr="experts",
        num_experts_config_key="num_experts",
        routing_mode=SOFTMAX
    )

    # Test pruning first layer
    layer_name = "layers.0"
    retained = [0, 2, 3, 5, 7]  # Keep 5 out of 8 experts

    print(f"\nPruning configuration:")
    print(f"  Layer: {layer_name}")
    print(f"  Original experts: {config.num_experts}")
    print(f"  Retained experts: {retained}")
    print(f"  Number retained: {len(retained)}")

    # Get layer and store original state
    layer = model.get_submodule(layer_name)
    original_num_experts = layer.num_experts
    original_router_weight = layer.gate.weight.detach().clone()

    # Count non-expert modules (e.g., act_fn) that will be retained
    num_non_expert_modules = sum(
        1 for module in layer.experts.children()
        if not isinstance(module, ExpertMLP)
    )

    # Store references to original expert modules
    original_expert_modules = {i: layer.experts[i] for i in retained}

    print(f"\nBefore pruning:")
    print(f"  layer.num_experts: {layer.num_experts}")
    print(f"  layer.experts.num_experts: {layer.experts.num_experts}")
    print(f"  layer.gate.num_experts: {layer.gate.num_experts}")
    print(f"  router weight shape: {layer.gate.weight.shape}")
    print(f"  len(layer.experts): {len(layer.experts)}")

    # Perform pruning
    print(f"\nPerforming pruning...")
    result = prune_moe_layer(model, layer_name, retained, attrs)

    print(f"\nAfter pruning:")
    print(f"  layer.num_experts: {layer.num_experts}")
    print(f"  layer.experts.num_experts: {layer.experts.num_experts}")
    print(f"  layer.gate.num_experts: {layer.gate.num_experts}")
    print(f"  router weight shape: {layer.gate.weight.shape}")
    print(f"  len(layer.experts): {len(layer.experts)}")

    # Validation checks
    print(f"\nValidation checks:")

    # Check 1: Return value matches input
    assert result == retained, f"Expected return {retained}, got {result}"
    print(f"  ✓ Function returned correct retained list")

    # Check 2: Number of experts updated correctly
    assert layer.num_experts == len(retained), \
        f"layer.num_experts should be {len(retained)}, got {layer.num_experts}"
    assert layer.experts.num_experts == len(retained), \
        f"layer.experts.num_experts should be {len(retained)}, got {layer.experts.num_experts}"
    assert layer.gate.num_experts == len(retained), \
        f"layer.gate.num_experts should be {len(retained)}, got {layer.gate.num_experts}"
    print(f"  ✓ num_experts attributes updated to {len(retained)}")

    # Check 3: Module list has correct length (retained experts + non-expert modules)
    expected_module_count = len(retained) + num_non_expert_modules
    assert len(layer.experts) == expected_module_count, \
        f"Expected {expected_module_count} modules ({len(retained)} experts + {num_non_expert_modules} non-expert), got {len(layer.experts)}"
    print(f"  ✓ Module list length is {len(layer.experts)} ({len(retained)} experts + {num_non_expert_modules} non-expert modules)")

    # Check 4: Correct experts retained (by object identity)
    # Only check expert modules, not non-expert modules like act_fn
    expert_count = 0
    for module in layer.experts:
        if isinstance(module, ExpertMLP):
            original_idx = retained[expert_count]
            assert module is original_expert_modules[original_idx], \
                f"Expert at position {expert_count} is not the original expert {original_idx}"
            expert_count += 1
    assert expert_count == len(retained), \
        f"Expected {len(retained)} expert modules, found {expert_count}"
    print(f"  ✓ Correct expert modules retained (by object identity)")

    # Check 5: Router weights correctly sliced
    expected_router_weight = original_router_weight[retained]
    torch.testing.assert_close(
        layer.gate.weight,
        expected_router_weight,
        msg="Router weights not correctly sliced"
    )
    print(f"  ✓ Router weights correctly sliced")

    # Check 6: Router output features updated
    assert layer.gate.out_features == len(retained), \
        f"Router out_features should be {len(retained)}, got {layer.gate.out_features}"
    print(f"  ✓ Router out_features updated to {len(retained)}")

    # Check 7: Layer forward pass still works
    print(f"\n  Testing layer forward pass after pruning...")
    test_layer_forward_pass(model, layer_name)

    # Check 8: Full model forward pass still works
    print(f"\n  Testing full model forward pass after pruning...")
    test_model_forward_pass(model)

    # Test pruning second layer with different retained set
    print("\n" + "-"*70)
    print("Testing pruning second layer with different retained set")
    print("-"*70)

    layer_name_2 = "layers.1"
    retained_2 = [1, 3, 4, 6]  # Keep 4 out of 8 experts

    print(f"\nPruning configuration for second layer:")
    print(f"  Layer: {layer_name_2}")
    print(f"  Retained experts: {retained_2}")
    print(f"  Number retained: {len(retained_2)}")

    result_2 = prune_moe_layer(model, layer_name_2, retained_2, attrs)
    layer_2 = model.get_submodule(layer_name_2)

    # Count non-expert modules for second layer
    num_non_expert_modules_2 = sum(
        1 for module in layer_2.experts.children()
        if not isinstance(module, ExpertMLP)
    )

    assert result_2 == retained_2
    assert layer_2.num_experts == len(retained_2)
    assert layer_2.experts.num_experts == len(retained_2)
    assert len(layer_2.experts) == len(retained_2) + num_non_expert_modules_2
    print(f"  ✓ Second layer pruned correctly to {len(retained_2)} experts")

    # Final model forward pass with both layers pruned
    print(f"\n  Testing full model with both layers pruned...")
    test_model_forward_pass(model)

    print("\n" + "="*70)
    print("All prune_moe_layer unit tests passed!")
    print("="*70)
    print(f"\nSummary:")
    print(f"  Layer 0: {original_num_experts} → {len(retained)} experts")
    print(f"  Layer 1: {original_num_experts} → {len(retained_2)} experts")
    print(f"  All forward passes: ✓ Working")
    print(f"  Expert pruning: ✓ Correct")
    print(f"  Router pruning: ✓ Correct")
    print("="*70 + "\n")


def test_model_forward_pass(model: FakeMoEModel):
    """
    Test forward pass through the entire model.

    Args:
        model: The FakeMoEModel instance
    """
    print("\n" + "="*70)
    print("Testing Full Model Forward Pass")
    print("="*70)

    config = model.config

    print(f"\nModel configuration:")
    print(f"  Number of layers: {config.num_hidden_layers}")
    print(f"  Experts per layer: {config.num_experts}")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Top-k routing: {config.num_experts_per_tok}")

    # Create test input
    batch_size = 2
    seq_len = 8
    hidden_dim = config.hidden_size

    hidden_states = torch.randn(batch_size, seq_len, hidden_dim, dtype=config.dtype)

    print(f"\nInput configuration:")
    print(f"  hidden_states shape: {hidden_states.shape}")

    # Perform forward pass
    print(f"\nPerforming forward pass through {config.num_hidden_layers} layers...")
    with torch.no_grad():
        output = model(hidden_states)

    print(f"  Output shape: {output.shape}")

    # Validation checks
    print(f"\nValidation checks:")

    # Check 1: Output shape matches input shape
    assert output.shape == hidden_states.shape, \
        f"Output shape {output.shape} doesn't match input shape {hidden_states.shape}"
    print(f"  ✓ Output shape matches input: {output.shape}")

    # Check 2: No NaN values
    has_nan = torch.isnan(output).any().item()
    assert not has_nan, "Output contains NaN values"
    print(f"  ✓ No NaN values in output")

    # Check 3: No Inf values
    has_inf = torch.isinf(output).any().item()
    assert not has_inf, "Output contains Inf values"
    print(f"  ✓ No Inf values in output")

    # Check 4: Output has non-zero values
    is_nonzero = (output.abs() > 1e-6).any().item()
    assert is_nonzero, "Output is all zeros - model may not be working"
    print(f"  ✓ Output contains non-zero values")

    # Check 5: Output is different from input (transformation occurred)
    is_different = not torch.allclose(output, hidden_states, rtol=1e-3)
    assert is_different, "Output is identical to input - no transformation occurred"
    print(f"  ✓ Output differs from input (transformation applied)")

    # Check 6: Different inputs produce different outputs
    hidden_states_2 = torch.randn(batch_size, seq_len, hidden_dim, dtype=config.dtype)
    with torch.no_grad():
        output_2 = model(hidden_states_2)
    is_deterministic = not torch.allclose(output, output_2, rtol=1e-3)
    assert is_deterministic, "Different inputs produced same output"
    print(f"  ✓ Different inputs produce different outputs")

    print("\n" + "="*70)
    print("All validation checks passed for full model!")
    print("="*70 + "\n")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Instantiating Fake MoE Model")
    print("="*70)

    # Set seed for reproducibility
    torch.manual_seed(42)

    # Create config
    config = FakeMoEConfig(
        num_experts=8,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2
    )

    # Create model
    model = FakeMoEModel(config)
    model.eval()

    print(f"\nModel Configuration:")
    print(f"  Number of layers: {config.num_hidden_layers}")
    print(f"  Experts per layer: {config.num_experts}")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Intermediate size: {config.intermediate_size}")
    print(f"  Top-k routing: {config.num_experts_per_tok}")

    print(f"\nModel Structure:")
    for i, layer in enumerate(model.layers):
        print(f"  Layer {i}:")
        print(f"    Router: {type(layer.gate).__name__} ({layer.gate.num_experts} experts)")
        print(f"    Experts: {type(layer.experts).__name__} ({layer.experts.num_experts} experts)")

    print("\n" + "="*70)
    print("Model instantiated successfully!")
    print("="*70 + "\n")

    # Test full model forward pass
    test_model_forward_pass(model)

    # Test individual layer forward pass
    test_layer_forward_pass(model, "layers.0")

    # Test prune_moe_layer function
    test_prune_moe_layer_unit()
