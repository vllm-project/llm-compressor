import requests
import torch
from PIL import Image
from transformers import AutoProcessor, Llama4ForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.utils.dev import skip_weights_download
from llmcompressor.utils.llama4 import linearize_moe

# Load model.
model_id = "meta-llama/Llama-4-Maverick-17B-128E-Instruct"
#with skip_weights_download(Llama4ForConditionalGeneration):
model = Llama4ForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.bfloat16  # load on cpu
)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

model = linearize_moe(model)

# Oneshot arguments
DATASET_ID = "flickr30k"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048
DATASET_SPLIT = {"calibration": f"test[:{NUM_CALIBRATION_SAMPLES}]"}


# Define a oneshot data collator for multimodal inputs.
def data_collator(batch):
    assert len(batch) == 1
    return {
        key: torch.tensor(value) if key != "pixel_values" else torch.tensor(value, dtype=torch.bfloat16).squeeze(0)
        for key, value in batch[0].items()
    }


# Recipe
recipe = [
    GPTQModifier(
        targets="Linear",
        scheme="W4A16",
        ignore=[
            "language_model.lm_head",
            "re:vision_model.*",
        ],
        #sequential_targets=["Llama4TextDecoderLayer"],
        sequential_targets=["Llama4TextAttention", "Llama4TextMLP", "Llama4TextMLP"],
    ),
]

# Perform oneshot
oneshot(
    model=model,
    tokenizer=model_id,
    dataset=DATASET_ID,
    splits=DATASET_SPLIT,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    trust_remote_code_model=True,
    data_collator=data_collator,
    oneshot_device="cuda:0",
)

# Confirm generations of the quantized model look sane.
print("========== SAMPLE GENERATION ==============")
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

inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=100)
print(processor.decode(output[0], skip_special_tokens=True))
print("==========================================")

# Save to disk compressed.
SAVE_DIR = model_id.split("/")[1] + "-W4A16-G128"
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)