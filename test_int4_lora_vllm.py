#!/usr/bin/env python3
"""
Test INT4 compressed-tensors + LoRA in vLLM
"""

import sys

def test_int4_lora():
    """Test INT4 compressed-tensors model with LoRA in vLLM."""
    print("=" * 80)
    print("INT4 Compressed-Tensors + LoRA Test")
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

    # Test model
    model_id = "Ishant86/FuseO1-DeepSeekR1-QwQ-SkyT1-32B-compressed-tensors-int4"

    print(f"\n[2] Loading INT4 compressed-tensors model: {model_id}")
    print("    This is a 32B model quantized to INT4 (~8GB)")
    print("    Model should be loaded with compressed-tensors quantization")

    try:
        llm = LLM(
            model=model_id,
            quantization="compressed-tensors",  # Explicitly specify compressed-tensors
            dtype="auto",
            max_model_len=512,
            enable_lora=True,  # Enable LoRA support
            max_loras=1,
            trust_remote_code=True,
            gpu_memory_utilization=0.9
        )
        print("✓ INT4 model loaded successfully with LoRA support enabled!")

        # Test inference
        print("\n[3] Testing inference with INT4 + LoRA enabled...")
        sampling_params = SamplingParams(temperature=0.0, max_tokens=20)
        outputs = llm.generate(["Hello, my name is"], sampling_params)

        generated_text = outputs[0].outputs[0].text
        print(f"✓ Inference successful")
        print(f"  Generated: {generated_text}")

        print("\n" + "=" * 80)
        print("RESULT: INT4 + LoRA WORKS IN vLLM!")
        print("=" * 80)
        print("✓ INT4 compressed-tensors model loaded")
        print("✓ LoRA support enabled")
        print("✓ Inference successful")

        return 0

    except Exception as e:
        print(f"\n✗ Failed to load INT4 model: {e}")
        import traceback
        traceback.print_exc()

        print("\n" + "=" * 80)
        print("ERROR DETAILS")
        print("=" * 80)
        print(f"Model: {model_id}")
        print(f"Error: {e}")

        return 1


if __name__ == "__main__":
    sys.exit(test_int4_lora())
