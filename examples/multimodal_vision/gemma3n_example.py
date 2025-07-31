import requests
import torch
from PIL import Image
from transformers import AutoProcessor, Gemma3nForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.utils import dispatch_for_generation

# Load model.
model_id = "google/gemma-3n-E2B-it"
model = Gemma3nForConditionalGeneration.from_pretrained(model_id, torch_dtype="auto")
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# Oneshot arguments
DATASET_ID = "flickr30k"
DATASET_SPLIT = {"calibration": "test[:512]"}
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048


# Define a oneshot data collator for multimodal inputs.
def data_collator(batch):
    assert len(batch) == 1
    return {key: torch.tensor(value) for key, value in batch[0].items()}


# Recipe
recipe = [
    GPTQModifier(
        targets="Linear",
        scheme="W4A16",
        ignore=[
            "re:.*embed_audio.*",
            "re:.*embed_vision.*",
            "re:.*audio_tower.*",
            "re:.*vision_tower.*",
            "re:.*altup.*",
            "re:.*lm_head.*",
            "re:.*laurel.*",
            "re:model\.language_model\.layers\.\d+\.per_layer_input_gate",
            "re:model\.language_model\.layers\.\d+\.per_layer_projection",
            "model.language_model.per_layer_model_projection",
        ],
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
    # gemma3n has broken weight offloading which is required by the sequential pipeline
    pipeline="basic",
    # gemma3n does not support untying word embeddings
    tie_word_embeddings=True,
)

# Confirm generations of the quantized model look sane.
print("========== SAMPLE GENERATION ==============")
dispatch_for_generation(model)
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
inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=100, disable_compile=True)
print(processor.decode(output[0], skip_special_tokens=True))
print("==========================================")

# # Save to disk compressed.
# SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-W4A16-G128"
# model.save_pretrained(SAVE_DIR, save_compressed=True)
# processor.save_pretrained(SAVE_DIR)
