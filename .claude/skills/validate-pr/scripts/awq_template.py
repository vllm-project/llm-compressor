import argparse

from compressed_tensors.offload import dispatch_model
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq import AWQModifier

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    args = parser.parse_args()

    MODEL_ID = args.model_id

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    DATASET_ID = "HuggingFaceH4/ultrachat_200k"
    DATASET_SPLIT = "train_sft"
    NUM_CALIBRATION_SAMPLES = 128
    MAX_SEQUENCE_LENGTH = 512

    ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
    ds = ds.shuffle(seed=42)

    def preprocess(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
            )
        }

    ds = ds.map(preprocess)

    recipe = [
        AWQModifier(duo_scaling="both"),
        QuantizationModifier(
            ignore=["lm_head"],
            scheme="W4A16_ASYM",
            targets=["Linear"],
        ),
    ]

    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    )

    print("\n\n")
    print("========== SAMPLE GENERATION ==============")
    dispatch_model(model)
    input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(
        model.device
    )
    output = model.generate(input_ids, max_new_tokens=100)
    print(tokenizer.decode(output[0]))
    print("==========================================\n\n")

    SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-awq-asym"
    model.save_pretrained(SAVE_DIR, save_compressed=True)
    tokenizer.save_pretrained(SAVE_DIR)


if __name__ == "__main__":
    main()
