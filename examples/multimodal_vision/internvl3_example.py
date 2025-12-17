import torch
from datasets import load_dataset
from transformers import AutoModelForImageTextToText, AutoProcessor

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

# Load model.
model_id = "OpenGVLab/InternVL3-8B-hf"
model = AutoModelForImageTextToText.from_pretrained(
    model_id, torch_dtype=torch.bfloat16
)
processor = AutoProcessor.from_pretrained(model_id)

# Load datasets
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"
NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 512
ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)


def preprocess_and_tokenize(example):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": example["messages"]},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    # remove extra dim added by multimodal processors
    inputs = {key: value[0] for key, value in inputs.items()}

    return inputs


ds = ds.map(preprocess_and_tokenize, remove_columns=ds.column_names)

# Recipe
recipe = GPTQModifier(
    targets="Linear",
    scheme="FP8",
    ignore=["re:.*lm_head", "re:.*vision_tower.*", "re:.*multi_modal_projector.*"],
)

# Perform oneshot
oneshot(
    model=model,
    tokenizer=model_id,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    trust_remote_code_model=True,
)

# Save to disk compressed.
SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-FP8"
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)
