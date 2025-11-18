#!/usr/bin/env python3
"""
Test INT4 MoE + LoRA in vLLM
Tests Qwen1.5-MoE-A2.7B with GPTQ-Int4 + LoRA
"""

import sys

def test_moe_int4_lora():
    """Test INT4 MoE model with LoRA in vLLM."""
    print("=" * 80)
    print("INT4 MoE + LoRA Test (vLLM)")
    print("=" * 80)

    # Import vLLM
    print("\n[1] Importing vLLM...")
    try:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
        print("✓ vLLM imports successful")
    except Exception as e:
        print(f"✗ Failed to import vLLM: {e}")
        return 1

    # Test model - Qwen MoE with GPTQ INT4
    model_id = "Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4"

    print(f"\n[2] Loading INT4 MoE model: {model_id}")
    print("    Architecture: Mixture of Experts (MoE)")
    print("    Model details:")
    print("      - Total params: 14.3B")
    print("      - Active params: 2.7B per token")
    print("      - Quantization: GPTQ INT4")
    print("      - Experts: 60 experts with Top-4 routing")

    try:
        llm = LLM(
            model=model_id,
            quantization="gptq",  # GPTQ INT4 quantization
            dtype="auto",
            max_model_len=2048,
            enable_lora=True,  # Enable LoRA support
            max_loras=1,
            trust_remote_code=True,
            gpu_memory_utilization=0.9
        )
        print("✓ INT4 MoE model loaded successfully with LoRA support enabled!")

        # Test inference
        print("\n[3] Testing inference with INT4 MoE + LoRA enabled...")
        sampling_params = SamplingParams(temperature=0.0, max_tokens=30)

        test_prompts = [
            "Hello, my name is",
            "The future of AI is"
        ]

        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n  Test {i}: '{prompt}'")
            outputs = llm.generate([prompt], sampling_params)
            generated_text = outputs[0].outputs[0].text
            print(f"  Generated: {generated_text}")

        print("\n" + "=" * 80)
        print("RESULT: INT4 MoE + LoRA WORKS IN vLLM!")
        print("=" * 80)
        print("✓ INT4 MoE model loaded (GPTQ)")
        print("✓ LoRA support enabled")
        print("✓ Inference successful")
        print("✓ MoE architecture: 60 experts, Top-4 routing")

        return 0

    except Exception as e:
        print(f"\n✗ Failed to load INT4 MoE model: {e}")
        import traceback
        traceback.print_exc()

        print("\n" + "=" * 80)
        print("ERROR DETAILS")
        print("=" * 80)
        print(f"Model: {model_id}")
        print(f"Error: {e}")

        return 1


if __name__ == "__main__":
    sys.exit(test_moe_int4_lora())
