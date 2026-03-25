import torch
from compressed_tensors.quantization import QuantizationScheme
from compressed_tensors.quantization.quant_args import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modeling.gpt_oss import convert_model_for_quantization_gptoss
from llmcompressor.modifiers.quantization import QuantizationModifier


def main():
    MODEL_ID = "openai/gpt-oss-20b"
    BASE_NAME = MODEL_ID.rstrip("/").split("/")[-1]
    OUTPUT_DIR = f"{BASE_NAME}-w4a8-channelwise"

    print(f"[GPT-OSS] Loading model: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    # ---- GPT-OSS MoE → linear experts conversion ----
    print("[GPT-OSS] Converting fused MoE experts to LinearExperts for quantization...")
    convert_model_for_quantization_gptoss(model)
    print("[GPT-OSS] Conversion completed.")

    # ---- Quantization config: W4A8 (int4 weights, int8 activations) ----

    # Weights: 4-bit, channelwise, symmetric, static
    weights_args = QuantizationArgs(
        num_bits=4,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.CHANNEL,
        symmetric=True,
        dynamic=False,
    )

    # Activations: 8-bit, per-token, asymmetric, dynamic
    activations_args = QuantizationArgs(
        num_bits=8,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.TOKEN,
        symmetric=False,
        dynamic=True,
        observer=None,
    )

    # Apply to all Linear layers, excluding lm_head
    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=weights_args,
        input_activations=activations_args,
    )

    recipe = QuantizationModifier(
        config_groups={"group_0": scheme},
        ignore=["lm_head"],
    )

    print(f"[GPT-OSS] Starting oneshot quantization → {OUTPUT_DIR}")
    oneshot(
        model=model,
        recipe=recipe,
        tokenizer=tokenizer,
        output_dir=OUTPUT_DIR,
        trust_remote_code_model=True,
    )
    print(f"[GPT-OSS] Quantization finished. Quantized model written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
