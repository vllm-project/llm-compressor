import argparse

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQMapping, AWQModifier
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.utils import dispatch_for_generation


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
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Select calibration dataset.
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"

# Select number of samples. 512 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048


def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
        )
    }


# Tokenize inputs.
def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


# Configure the quantization algorithm to run.
#   * quantize self_attn layers to W8A8 with GPTQ
#   * quantize mlp layers to W4A16 with AWQ
#       only include mappings pertaining to target layers
recipe = [
    GPTQModifier(targets=r"re:.*self_attn\.(k|q|o|v)_proj$", scheme="W8A8"),
    AWQModifier(
        targets=r"re:.*mlp\.(down|gate|up)_proj$",
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
        scheme="W4A16",
    ),
]

if __name__ == "__main__":
    args = parse_args()
    # Load dataset and preprocess.
    ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
    ds = ds.shuffle(seed=42)
    ds = ds.map(preprocess)
    ds = ds.map(tokenize, remove_columns=ds.column_names)

    # Apply algorithms.
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
        pipeline="independent" if args.independent else "sequential",
    )

    # Confirm generations of the quantized model look sane.
    print("\n\n")
    print("========== SAMPLE GENERATION ==============")
    dispatch_for_generation(model)
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
