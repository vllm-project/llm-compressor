import requests
from compressed_tensors.offload import dispatch_model
from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

# Load model.
model_id = "google/gemma-3-4b-it"
model = Gemma3ForConditionalGeneration.from_pretrained(model_id, dtype="auto")
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# Oneshot arguments
BATCH_SIZE = 4
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048
DATASET_ID = "flickr30k"
DATASET_SPLIT = {"calibration": f"test[:{NUM_CALIBRATION_SAMPLES}]"}

# Recipe
recipe = [
    GPTQModifier(
        targets="Linear",
        scheme="W4A16",
        ignore=[
            "lm_head",
            r"re:model\.vision_tower.*",
            r"re:model\.multi_modal_projector.*",
        ],
    ),
]

# Perform oneshot
oneshot(
    model=model,
    processor=processor,
    dataset=DATASET_ID,
    splits=DATASET_SPLIT,
    recipe=recipe,
    batch_size=BATCH_SIZE,
    shuffle_calibration_samples=False,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
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

# Note: compile is disabled: https://github.com/huggingface/transformers/issues/38333
inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=100, disable_compile=True)
print(processor.decode(output[0], skip_special_tokens=True))
print("==========================================")

# Save to disk compressed.
SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-W4A16-G128"
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)
