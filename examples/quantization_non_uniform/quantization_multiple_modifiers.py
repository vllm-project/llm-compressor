# NOTE: to use a custom dataset, see examples/custom_dataset_example.py
import argparse

from compressed_tensors.offload import dispatch_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq import AWQMapping, AWQModifier


def parse_args():
    parser = argparse.ArgumentParser(description="Quantization with multiple modifiers")
    parser.add_argument(
        "--independent",
        action="store_true",
        help="Add this flag if you'd like to run each modifier "
        "independently instead of in the same sequence",
    )
    return parser.parse_args()


# Select model and load it.
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Configure the quantization algorithm to run.
#   * quantize self_attn layers to W8A8 with GPTQ
#   * quantize mlp layers to W4A16 with AWQ
#       only include mappings pertaining to target layers
recipe = [
    GPTQModifier(targets=r"re:.*self_attn\.(k|q|o|v)_proj$", scheme="W8A8"),
    AWQModifier(
        mappings=[
            AWQMapping(
                "re:.*post_attention_layernorm$",
                ["re:.*gate_proj$", "re:.*up_proj$"],
            ),
            AWQMapping(
                "re:.*up_proj$",
                ["re:.*down_proj$"],
            ),
        ],
    ),
    QuantizationModifier(
        targets=r"re:.*mlp\.(down|gate|up)_proj$",
        scheme="W4A16",
    ),
]

if __name__ == "__main__":
    args = parse_args()

    # Apply algorithms.
    oneshot(
        model=model,
        dataset="perfectblend",
        splits="train[:512]",
        recipe=recipe,
        max_seq_length=2048,
        num_calibration_samples=512,
        pipeline="independent" if args.independent else "sequential",
    )

    # Confirm generations of the quantized model look sane.
    print("\n\n")
    print("========== SAMPLE GENERATION ==============")
    dispatch_model(model)
    sample = tokenizer("Hello my name is", return_tensors="pt")
    sample = {key: value.to(model.device) for key, value in sample.items()}
    output = model.generate(**sample, max_new_tokens=100)
    print(tokenizer.decode(output[0]))
    print("==========================================\n\n")

    # Save to disk compressed.
    SAVE_DIR = (
        model_id.rstrip("/").split("/")[-1] + "-gptq-w8a8-self_attn-awq-w4a16-mlp"
    )
    model.save_pretrained(SAVE_DIR, save_compressed=True)
    tokenizer.save_pretrained(SAVE_DIR)
