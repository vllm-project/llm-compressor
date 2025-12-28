"""
Standalone script to verify that GLM4 MoE calibration module sends all tokens to all experts.

This replicates the logic from test_calib_replace_glm4moe_all_experts but without the cadence requirement.
"""

import contextlib
import sys
from functools import partial
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

# Add src to path if needed
repo_root = Path(__file__).parent.parent.parent
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from llmcompressor.modeling.glm4_moe import CalibrationGlm4MoeMoE
from llmcompressor.modeling.moe_context import moe_calibration_context
from llmcompressor.utils.dev import skip_weights_download
from llmcompressor.utils.helpers import calibration_forward_context


def verify_all_experts_activated(model_path_or_stub: str):
    """
    Verify that all experts in GLM4 MoE model receive calibration data.
    
    Args:
        model_path_or_stub: Either a local path or HuggingFace model stub
    """
    print("=" * 70)
    print("GLM4 MoE Expert Activation Verification")
    print("=" * 70)
    
    # Load model
    print(f"\n1. Loading model from: {model_path_or_stub}")
    try:
        with skip_weights_download():
            model = AutoModelForCausalLM.from_pretrained(
                model_path_or_stub,
                torch_dtype="auto",
                device_map="auto"
            )
        print(f"   ✓ Model loaded successfully")
        print(f"   Device: {next(model.parameters()).device}")
    except Exception as e:
        print(f"   ✗ Failed to load model: {e}")
        return False
    
    # Activate calibration contexts
    print("\n2. Activating calibration contexts...")
    with contextlib.ExitStack() as stack:
        stack.enter_context(calibration_forward_context(model))
        stack.enter_context(moe_calibration_context(model, calibrate_all_experts=True))
        print("   ✓ Calibration contexts activated")
        
        # Find GLM4 MoE layers
        print("\n3. Finding GLM4 MoE layers...")
        moe_layers = []
        for name, module in model.named_modules():
            if isinstance(module, CalibrationGlm4MoeMoE):
                moe_layers.append((name, module))
        
        if not moe_layers:
            print("   ✗ No GLM4 MoE calibration layers found!")
            print("   This means the calibration module wasn't registered or activated.")
            return False
        
        print(f"   ✓ Found {len(moe_layers)} MoE layer(s)")
        
        # Test the first MoE layer
        layer_name, moe_layer = moe_layers[0]
        print(f"\n4. Testing MoE layer: {layer_name}")
        print(f"   Number of experts: {len(moe_layer.experts)}")
        print(f"   calibrate_all_experts: {moe_layer.calibrate_all_experts}")
        
        # Set up expert tracking
        num_experts = len(moe_layer.experts)
        expert_triggered = [False] * num_experts
        
        # Define hook function
        def hook_fn(i, module, input, output):
            expert_triggered[i] = True
        
        # Attach hooks to all experts
        print("\n5. Attaching forward hooks to all experts...")
        hooks = []
        for i, expert in enumerate(moe_layer.experts):
            hook = expert.register_forward_hook(partial(hook_fn, i))
            hooks.append(hook)
        print(f"   ✓ Attached {len(hooks)} hooks")
        
        # Create test input
        print("\n6. Running forward pass...")
        hidden_dim = model.config.hidden_size
        batch, seq_len = 4, 32
        sample = torch.randn(
            batch, seq_len, hidden_dim,
            dtype=torch.float32,
            device=next(model.parameters()).device
        )
        print(f"   Input shape: {sample.shape}")
        
        # Run forward pass
        with torch.no_grad():
            _ = moe_layer(sample)
        print("   ✓ Forward pass completed")
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Verify results
        print("\n7. Verifying expert activation...")
        print(f"   Expert activation status: {expert_triggered}")
        
        triggered_count = sum(expert_triggered)
        print(f"\n   Results:")
        print(f"   - Experts triggered: {triggered_count}/{num_experts}")
        print(f"   - Experts NOT triggered: {num_experts - triggered_count}")
        
        if all(expert_triggered):
            print("\n" + "=" * 70)
            print("✓ SUCCESS: All experts received calibration data!")
            print("=" * 70)
            return True
        else:
            print("\n" + "=" * 70)
            print("✗ FAILURE: Not all experts received calibration data!")
            print("=" * 70)
            not_triggered = [i for i, triggered in enumerate(expert_triggered) if not triggered]
            print(f"   Experts NOT triggered: {not_triggered}")
            return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Verify GLM4 MoE calibration module activates all experts"
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to model (local path or HuggingFace model stub)"
    )
    
    args = parser.parse_args()
    
    success = verify_all_experts_activated(args.model_path)
    sys.exit(0 if success else 1)

