import requests
import torch
from PIL import Image
from transformers import AutoProcessor

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.transformers.tracing import TraceableLlavaForConditionalGeneration

# Load model.
model_id = "mgoin/pixtral-12b"
model = TraceableLlavaForConditionalGeneration.from_pretrained(
    model_id, device_map="auto", torch_dtype="auto"
)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# Oneshot arguments
DATASET_ID = "flickr30k"
NUM_CALIBRATION_SAMPLES = 512
DATASET_SPLIT = {"calibration": f"test[:{NUM_CALIBRATION_SAMPLES}]"}
MAX_SEQUENCE_LENGTH = 2048


# Define a oneshot data collator for multimodal inputs.
# NOTE: for transformers<4.48.0, please squeeze the first dimension of `pixel_values`
# by appending `[0]` to the end of line 32
def data_collator(batch):
    assert len(batch) == 1
    return {
        "input_ids": torch.LongTensor(batch[0]["input_ids"]),
        "attention_mask": torch.tensor(batch[0]["attention_mask"]),
        "pixel_values": torch.tensor(batch[0]["pixel_values"]),
    }


# Recipe
recipe = [
    GPTQModifier(
        targets="Linear",
        scheme="W4A16",
        sequential_targets=["MistralDecoderLayer"],
        ignore=["re:.*lm_head", "re:vision_tower.*", "re:multi_modal_projector.*"],
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
