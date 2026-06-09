from compressed_tensors.quantization import QuantizationScheme
from compressed_tensors.quantization.quant_args import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils.dev import load_context


def main():
    MODEL_ID = "openai/gpt-oss-20b"
    BASE_NAME = MODEL_ID.rstrip("/").split("/")[-1]
    OUTPUT_DIR = f"{BASE_NAME}-w4a8-channelwise"

    print(f"[GPT-OSS] Loading model: {MODEL_ID}")
    with load_context():
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto",
        )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

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
    )
    print(f"[GPT-OSS] Quantization finished. Quantized model written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
