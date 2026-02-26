import json
import os

import requests
import torch
from compressed_tensors.offload import dispatch_model
from PIL import Image
from transformers import (
    AutoProcessor,
    Mistral3ForConditionalGeneration,
    default_data_collator,
)

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

# Load model.
model_id = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
model = Mistral3ForConditionalGeneration.from_pretrained(model_id, dtype="auto")
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# Use a custom calibration chat template, rather than the overly-verbose default
file_path = os.path.join(os.path.dirname(__file__), "mistral3_chat_template.json")
with open(file_path, "r") as file:
    processor.chat_template = json.load(file)["chat_template"]

# Oneshot arguments
DATASET_ID = "flickr30k"
DATASET_SPLIT = "test"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048


# Patch: mismatch between processor and model dtype
def data_collator(features):
    for feature in features:
        feature["pixel_values"] = torch.tensor(
            feature["pixel_values"], dtype=model.dtype
        )
    return default_data_collator(features, return_tensors="pt")


# Recipe
recipe = [
    GPTQModifier(
        targets="Linear",
        scheme="W4A16",
        ignore=["re:.*lm_head", "re:.*vision_tower.*", "re:.*multi_modal_projector.*"],
    ),
]

# Perform oneshot
oneshot(
    model=model,
    tokenizer=model_id,
    dataset=DATASET_ID,
    splits={"calibration": f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]"},
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    trust_remote_code_model=True,
    data_collator=data_collator,
    sequential_targets=["MistralDecoderLayer"],
)

# Confirm generations of the quantized model look sane.
print("========== SAMPLE GENERATION ==============")
dispatch_model(model)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Please describe the animal in this image\n"},
            {"type": "image"},
        ],
    },
]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
image_url = "http://images.cocodataset.org/train2017/000000231895.jpg"
raw_image = Image.open(requests.get(image_url, stream=True).raw)

inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to(model.device)
inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)  # fix dtype
output = model.generate(**inputs, max_new_tokens=100)
print(processor.decode(output[0], skip_special_tokens=True))
print("==========================================")

# Save to disk compressed.
SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-W4A16-G128"
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)
