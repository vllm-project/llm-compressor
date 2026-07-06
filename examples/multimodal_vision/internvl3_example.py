import requests
from compressed_tensors.offload import dispatch_model
from datasets import load_dataset
from PIL import Image
from transformers import AutoProcessor, InternVLForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier

# Load model.
model_id = "OpenGVLab/InternVL3-8B-hf"
model = InternVLForConditionalGeneration.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

# Oneshot arguments
DATASET_ID = "lmms-lab/flickr30k"
DATASET_SPLIT = "test"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)


# Apply chat template and tokenize inputs.
def preprocess_and_tokenize(example):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What does this image show?"},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": " ".join(example["caption"])},
            ],
        },
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=False)
    inputs = processor(images=example["image"], text=prompt, return_tensors="pt")
    return {key: value[0] for key, value in inputs.items()}  # Unwrap batch dimension


ds = ds.map(preprocess_and_tokenize, remove_columns=ds.column_names)

# Recipe
recipe = GPTQModifier(
    targets="Linear",
    scheme="NVFP4",
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
    }
]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
image_url = "http://images.cocodataset.org/train2017/000000231895.jpg"
raw_image = Image.open(requests.get(image_url, stream=True).raw)

inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=100)
print(processor.decode(output[0], skip_special_tokens=True))
print("==========================================")

# Save to disk compressed.
SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-NVFP4"
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)
